# Local Model Evaluation & Version Comparison Service

一个本地运行的模型评估服务：提交少量样本（可带基准结果）与模型版本标识，
在本地计算可配置、**确定性**的评估指标，支持并发评估、任务取消与超时、
按租户/项目的配额隔离以及进程崩溃后的恢复。所有结果可复现。

基于 Python 3.14 + FastAPI + SQLite（标准库，无外部数据库依赖）。

---

## 1. 快速开始

```bash
# 安装依赖（需要 Python 3.14，uv 会自动获取）
uv sync

# 启动服务（默认监听 uvicorn 的 8000 端口）
uv run uvicorn main:app --reload

# 或运行测试
uv run pytest
```

可通过环境变量配置：

| 变量 | 默认值 | 含义 |
| --- | --- | --- |
| `EVAL_DB_PATH` | `eval.db` | SQLite 数据库路径（自动创建，启用 WAL） |
| `EVAL_MAX_WORKERS` | `4` | 评估任务最大并发度 |
| `EVAL_DEFAULT_QUOTA` | `100` | 未显式配置时的每租户默认配额 |
| `EVAL_DEFAULT_TIMEOUT` | `60` | 任务默认超时（秒）；请求可覆盖 |

启动后可访问交互式文档：<http://127.0.0.1:8000/docs>。

### 端到端示例

```bash
# 1) 提交评估（样本可直接带 prediction；也可只给 input 走本地确定性模型后端）
curl -s -X POST http://127.0.0.1:8000/api/v1/evaluations \
  -H 'Content-Type: application/json' \
  -d '{
    "tenant_id": "acme", "project_id": "search-rank",
    "model_version": "v2026-09-20",
    "samples": [
      {"id": "s1", "reference": "hello world", "prediction": "hello world"},
      {"id": "s2", "reference": "a b c",       "prediction": "a b d"}
    ],
    "metrics": ["exact_match", "token_f1"],
    "config": {"case_sensitive": true}
  }'
# -> {"task_id":"task-....","state":"pending","reused":false}

# 2) 查询状态 / 结果 / 样本级得分
curl -s "http://127.0.0.1:8000/api/v1/evaluations/task-xxxx?tenant_id=acme"
curl -s "http://127.0.0.1:8000/api/v1/evaluations/task-xxxx/result?tenant_id=acme"
curl -s "http://127.0.0.1:8000/api/v1/evaluations/task-xxxx/samples?tenant_id=acme"

# 3) 取消
curl -s -X POST "http://127.0.0.1:8000/api/v1/evaluations/task-xxxx/cancel?tenant_id=acme"
```

---

## 2. HTTP 接口

所有业务接口前缀：`/api/v1`。任务归属由请求体的 `tenant_id`（及可选
`project_id`）决定；读取/取消时以 query 参数 `tenant_id` 校验，跨租户访问
统一返回 `404`（不泄露任务是否存在）。

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| POST | `/evaluations` | 提交评估；重复提交返回同一 `task_id`（`reused:true`），不重复扣配额 |
| GET | `/evaluations/{id}` | 查询任务状态与（完成后的）结果 |
| GET | `/evaluations/{id}/result` | 仅成功任务可取结果，否则 `409` |
| GET | `/evaluations/{id}/samples` | 样本级得分与逐样本失败原因 |
| POST | `/evaluations/{id}/cancel` | 取消排队或运行中的任务 |
| GET | `/tenants/{tenant}/evaluations?state=` | 列出租户任务，可按状态过滤 |
| PUT | `/quotas` | 设置租户或项目配额（body：`tenant_id`/`project_id`/`limit`） |
| GET | `/quotas` | 查看各 scope 的限额与用量 |
| GET | `/metrics` | 已注册指标目录 |

### 提交体（`EvalRequest`）

```jsonc
{
  "tenant_id": "acme",            // 必填，租户隔离维度
  "project_id": "rank",           // 可选，项目隔离维度
  "model_version": "v1.2.3",      // 参与缓存指纹
  "samples": [                    // 每条样本
    {"id": "s1", "input": "...", "reference": "期望输出", "prediction": "可选：直接给定预测"}
  ],
  "metrics": ["exact_match", "token_f1"],  // 指标集合（顺序不影响指纹）
  "config": {"case_sensitive": false},     // 打分/后端配置，参与指纹
  "baseline": {"task_id": "task-..."}      // 或 {"metrics": {"exact_match": 0.8}}
  "timeout_seconds": 30,                    // 可选，覆盖默认超时
  "idempotency_key": "client-key-123"       // 可选，客户端幂等键
}
```

### 状态机

`pending → running → succeeded`，或终态 `failed` / `cancelled` / `timeout`。
非终态读取返回明确的 `state`、时间戳与（若有）`error`，保证可解释。

---

## 3. 指标与确定性打分

内置纯函数指标（注册在 `app/metrics.py`，可用 `register_metric` 扩展）：

| 指标 | 含义 |
| --- | --- |
| `exact_match` | 完全匹配得 1，否则 0；支持配置 `case_sensitive`（默认 true）、`strip`（默认 false） |
| `token_f1` | 基于空白分词的集合 F1（顺序、重复词不敏感） |
| `numeric_abs_error` | 数值预测与参考值的绝对误差；无法解析为数值时该样本记为失败 |

确定性保证：

* 指标是 `(prediction, reference, config)` 的纯函数；默认模型后端
  `DeterministicBackend` 以 `SHA-256(model_version, seed, input)` 产生预测，
  进程重启、跨机器结果一致。
* 请求指纹 = 规范化 JSON（字典键排序、紧凑分隔）的 SHA-256，覆盖
  租户/项目、模型版本、样本、指标集合、配置、基准与幂等键。
  **任何一项变化都得到不同任务，绝不复用彼此缓存**。
* `timeout_seconds` 是执行参数，**不**参与指纹（同一评估不因超时而分裂缓存）。
* 指标顺序不影响指纹；结果以规范化 JSON 落盘。非有限浮点（如自定义指标产生的
  `inf`）序列化为 `null`；无法聚合（无成功样本）的指标也是 `null`。

### 基准对比

`baseline` 支持两种形式：

* `{"metrics": {...}}`：直接给出基准指标值；
* `{"task_id": "..."}`：引用同租户下另一个**已成功**任务的结果（跨租户或未成功
  会令任务 `failed` 并给出原因）。

结果中 `comparison` 给出每个指标的 `current` / `baseline` / `delta`。

---

## 4. 并发、取消、超时与配额

### 并发（要求 6）

* 固定大小线程池（`EVAL_MAX_WORKERS`）提供有限并发度，超出任务排队；
  慢任务不会阻塞其他任务。
* 任务线程在样本边界协作检查取消信号；模拟延迟（配置 `latency` 秒）按 20ms
  切片可中断，因此取消/超时响应及时。

### 取消与超时（要求 3）

* 取消/超时意图先持久化到 `cancel_request` 表（**首个意图先赢**），再通知 worker。
  这消除了“取消与结果写入”的竞态：
  * 取消先发生 → 结果被丢弃，任务终态 `cancelled`/`timeout`，**绝不留下部分结果**；
  * 评估已完成 → 结果胜出，随后的取消返回保持 `succeeded`。
* 排队任务被取消会立即终态化，不会真正执行；运行中任务在下一样本边界中止。
* 超时由看门狗定时器触发（同时持久化 `timeout` 意图），终态为 `timeout`。

> 说明：线程不能被强杀。评估代码在样本间、以及可中断的后端调用间检查信号。
> 若接入会长时间阻塞的真实模型后端，请在其后端内部同样轮询取消事件，
> 或改用进程执行器。

### 配额（要求 4）

* 用量直接由 `task` 行派生（提交时一次性计入，不退款、不依赖可能漂移的计数器），
  因此**重启后绝不重复累计**。
* 项目级任务同时计入**租户总量**与**项目量**，两级限额都必须满足；
  配额耗尽时新提交返回 `429`（重复提交不受影响、不扣配额）。
* 调整限额**只影响之后的新提交**；已排队/运行中的任务继续执行，不会被撤销。

---

## 5. 崩溃恢复（要求 5）

服务启动时执行恢复：

* `succeeded/failed/cancelled/timeout` 等终态任务结果保持可读；
* 崩溃时处于 `running` 的任务回退为 `pending` 并按幂等原则重新执行；
  不新增任务行、不重复计配额、不产生重复结果；
* 崩溃前已记录取消/超时意图的运行中任务，直接终态化（不再重跑）；
* 去重索引（指纹 → 任务）持久化，恢复后重复提交仍命中同一任务。

任务标识本身就是指纹的派生值（`task-<指纹前16位>`，冲突在唯一约束下受控），
因此“同一请求 ⇒ 同一任务标识”在重启后依然成立。

---

## 6. 样本级失败（要求 7）

单个样本的预测或任一指标抛错时：

* 该样本记录 `error`（形如 `ValueError: ...`），**不会被静默计为成功**；
* 汇总指标只对成功样本求平均，`succeeded_samples` / `failed_samples`
  如实反映失败情况；某指标若没有任何成功样本，聚合值为 `null`；
* 样本级失败不会导致整个任务失败——任务仍可 `succeeded`，
  但结果中显式暴露全部失败。可通过 `/samples` 端点单独查看。

---

## 7. 代码结构

```
main.py              入口（create_app + 环境变量配置）
app/
  models.py          Pydantic 模型、规范化 JSON、请求指纹（SHA-256）
  metrics.py         确定性指标注册表（exact_match / token_f1 / numeric_abs_error）
  evaluator.py       确定性模型后端 + 逐样本评估核心 + 取消/超时检查点
  executor.py        有限并发线程池、取消令牌、超时看门狗
  storage.py         SQLite：任务/去重/配额/取消意图，原子事务与 CAS 状态转换
  quotas.py          租户+项目两级配额
  service.py         编排层：提交去重、执行、取消竞态、恢复、基准解析
  api.py             FastAPI 路由与错误码
tests/               51 个自动化测试（见下）
```

### 自动化测试（`uv run pytest`）

| 文件 | 覆盖场景 |
| --- | --- |
| `test_dedup.py` | 重复提交、40 并发重复提交只建一次、配额只扣一次、幂等键分区、并发不同提交精确受配额约束 |
| `test_determinism.py` | 同输入同配置结果一致、版本/配置/指标/基准变化导致缓存失效、跨重启复现、基准对比 |
| `test_cancel_timeout.py` | 运行中/排队取消、幂等取消、超时、无部分结果、晚到取消不覆盖成功、样本级失败 |
| `test_quota_isolation.py` | 租户/项目隔离、配额耗尽 429、两级配额、限额变更不影响存量任务、重启不重复计配额 |
| `test_recovery.py` | 完成结果可读、RUNNING 崩溃重跑幂等、PENDING 重排队、取消/超时意图跨崩溃、结果确定性 |
| `test_concurrency.py` | 并发度上限、慢任务不阻塞快任务、多租户并行推进 |
| `test_race_invariants.py` | 随机取消/完成竞态下终态自洽：成功必有结果、取消/超时必无结果且原因可解释 |
| `test_api.py` | 全 HTTP 流程、跨租户 404、429、取消/超时状态、配额管理接口、参数校验 |

---

## 8. 作为库嵌入

```python
from app.service import EvalService
from app.models import EvalRequest, Sample

with EvalService(db_path="eval.db", max_workers=4) as svc:
    resp = svc.submit(EvalRequest(
        tenant_id="t", model_version="v1",
        samples=[Sample(id="1", reference="x", prediction="x")],
        metrics=["exact_match"],
    ))
    svc.executor.wait(resp.task_id, timeout=30)
    print(svc.get_result(resp.task_id))
```

扩展指标 / 模型后端：

```python
from app.metrics import register_metric
from app.evaluator import register_backend

register_metric("length_accuracy", lambda pred, ref, cfg: float(len(pred) == len(ref)))

class MyBackend:
    def predict(self, model_version, text, config) -> str:
        ...  # 必须确定性
register_backend("my-model", MyBackend())
# 请求中 config={"backend": "my-model"}
```
