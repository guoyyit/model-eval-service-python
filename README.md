# model-eval-service-python

本地模型评估与版本对比服务。提交包含少量样本、可选基准结果与模型版本标识的评估请求，服务在本地以**确定性**方式计算指标，支持并发评估、任务取消、超时、租户配额与崩溃恢复，所有结果可复现。

## 快速开始

```bash
uv sync                 # 安装依赖
uv run uvicorn main:app --host 127.0.0.1 --port 8000
```

环境变量配置（均有默认值）：

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `EVAL_DB_PATH` | `./data/eval.db` | SQLite 持久化路径（`:memory:` 表示纯内存） |
| `EVAL_MAX_WORKERS` | `4` | 评估任务最大并发度 |
| `EVAL_DEFAULT_QUOTA` | `100` | 每租户默认任务配额 |
| `EVAL_DEFAULT_TIMEOUT` | `60` | 单任务默认超时（秒） |

## API 一览

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| POST | `/evaluations` | 提交评估；相同请求去重，返回同一 `task_id` |
| GET | `/evaluations?tenant=` | 列出任务（可按租户过滤） |
| GET | `/evaluations/{task_id}` | 查询任务状态与结果 |
| POST | `/evaluations/{task_id}/cancel` | 取消任务（幂等） |
| GET | `/tenants/{tenant}/quota` | 查询租户配额用量 |
| PUT | `/tenants/{tenant}/quota` | 调整租户配额（仅影响新提交） |
| GET | `/metrics` | 列出可用指标 |
| GET | `/health` | 健康检查 |

### 提交评估

```bash
curl -X POST localhost:8000/evaluations -H 'content-type: application/json' -d '{
  "tenant": "acme",
  "project": "chatbot",
  "model_version": "v2.1",
  "samples": [
    {"id": "s1", "input": "hello", "expected": "hello"},
    {"id": "s2", "input": "1.0",  "expected": "1.0"}
  ],
  "metrics": [{"name": "exact_match"}, {"name": "token_f1"}],
  "baseline": {"exact_match": 0.5},
  "timeout_seconds": 30
}'
```

响应：`{"task": {...}, "created": true}`。`created=false` 表示命中了相同请求的进行中/已完成任务（去重，不重复扣配额）。

### 查询结果

```bash
curl localhost:8000/evaluations/<task_id>
```

任务状态机：`queued → running → succeeded | failed | cancelled | timeout`。
终态任务的 `result` 包含：

- `sample_scores`：样本级得分，失败样本带 `error` 原因且不计入聚合；
- `aggregate`：各指标在**成功样本**上的均值；
- `failed_samples` / `failure_reasons`：失败样本及原因（不会被静默计为成功）；
- `delta_vs_baseline`：与请求中基准结果的差值（提供 `baseline` 时）。

## 内置指标

`exact_match`、`contains`、`length_ratio`、`numeric_close`（支持 `params.tolerance`）、`token_f1`。
指标是纯函数；模型输出由 `(model_version, input)` 确定性模拟，因此**相同输入与相同配置必然得到相同结果**。

## 设计要点

- **去重与缓存**：请求指纹 = SHA256(模型版本 + 样本 + 指标配置 + 基准)（规范化 JSON）。
  指纹不含租户与执行参数（超时等）。`(tenant, fingerprint)` 上的部分唯一索引保证：
  并发重复提交只创建一次执行；已成功任务即结果缓存；已取消/失败/超时的请求可重新提交为新任务。
  不同模型版本或指标配置指纹不同，绝不复用彼此缓存。
- **取消与超时不留部分结果**：结果只在“状态翻转为 `succeeded`”的同一条原子 UPDATE 中写入；
  取消/超时通过 CAS（compare-and-swap）与结果写入竞争，先到先得，败者无副作用。
- **租户配额**：提交时在**同一事务**内完成去重、配额扣减与任务创建；配额账本以
  `(tenant, task_id)` 为主键，重复扣减幂等。配额耗尽返回 429；已接受任务不受后续配额调整影响。
- **崩溃恢复**：SQLite WAL + FULL sync。重启时 `running` 任务重置为 `queued` 并幂等重跑，
  已完成结果直接可读，配额不重复累计，结果不重复产生。
- **有界并发**：固定大小线程池执行评估，慢任务不阻塞其他任务；取消事件在样本间即时生效。

## 测试

```bash
uv run pytest          # 40 个用例
```

覆盖：重复提交（含并发）、取消与超时（含取消/结果写入竞态）、缓存失效（版本/指标配置）、
崩溃恢复（结果可读、幂等重跑、配额不重复）、并发隔离（有界并发、慢任务不阻塞、租户隔离）、
指标确定性与失败样本记录、HTTP API 行为。

## 代码结构

```
evalsvc/
  models.py       # 请求/结果/任务等类型定义
  fingerprint.py  # 请求指纹（去重与缓存键）
  metrics.py      # 确定性指标注册表 + 模拟模型输出
  store.py        # SQLite 持久化：原子状态转换、配额账本、恢复
  quota.py        # 租户配额策略
  runner.py       # 确定性评估执行（取消/超时感知）
  service.py      # 提交去重、调度、取消、恢复编排
  app.py          # FastAPI 接线
main.py           # uvicorn 入口
tests/            # 自动化测试
```
