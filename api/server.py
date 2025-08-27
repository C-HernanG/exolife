"""
Simple FastAPI server for ExoLife.

This module defines a minimal HTTP API using FastAPI.  It exposes
a health endpoint at `/health` which returns a JSON payload
indicating the service status, and a root endpoint at `/` that
returns a welcome message.  To extend the API with real model
predictions, import your trained models from the ExoLife package
and expose additional endpoints.
"""

from fastapi import FastAPI

app = FastAPI(title="ExoLife API", version="0.1.0")


@app.get("/health", summary="Health check", tags=["system"])
async def health() -> dict[str, str]:
    """Return a simple health check status."""
    return {"status": "ok"}


@app.get("/", summary="Welcome", tags=["system"])
async def root() -> dict[str, str]:
    """Return a welcome message."""
    return {"message": "Welcome to the ExoLife API"}
