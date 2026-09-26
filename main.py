"""Entrypoint: run the evaluation service with uvicorn."""

from evalsvc.app import create_app

app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
