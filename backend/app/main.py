"""HTTP API for CardioLens predictions and SHAP contributions."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.app.engine import get_bundle


@asynccontextmanager
async def lifespan(_app: FastAPI):
    get_bundle()
    yield


app = FastAPI(
    title="CardioLens",
    summary="CAD and coronary stenosis estimates with SHAP explanations.",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    features: dict[str, Any] = Field(description="Clinical inputs. Do not include LAD, LCX, RCA, or Cath.")


@app.get("/health")
def health() -> dict[str, str]:
    bundle = get_bundle()
    return {"status": "ok", "model": bundle.metrics["selected_model"]}


@app.get("/bootstrap")
def bootstrap() -> dict[str, Any]:
    return get_bundle().bootstrap()


@app.get("/metrics")
def metrics() -> dict[str, Any]:
    return get_bundle().metrics


@app.post("/predict")
def predict(body: PredictRequest) -> dict[str, Any]:
    try:
        return get_bundle().predict(body.features)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
