"""Minimal HTTP interface for loaded local decision profiles."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

from reflexone import __version__
from reflexone.inference import SupportPredictor
from reflexone.schema import CategoricalDecision, PredictRequest, SupportDecision
from reflexone.tasks import TaskPredictor


class BatchRequest(BaseModel):
    requests: Annotated[list[PredictRequest], Field(min_length=1, max_length=32)]


def create_app(artifacts_root: Path | str = "artifacts") -> FastAPI:
    root = Path(artifacts_root)
    predictors = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if (root / "support" / "model.safetensors").exists():
            predictors["support"] = SupportPredictor(root / "support")
        for name in ("commands", "spam"):
            if (root / name / "model.joblib").exists():
                predictor = TaskPredictor(root / name)
                if predictor.task != name:
                    raise ValueError(
                        f"Expected profile '{name}' in directory, found '{predictor.task}'"
                    )
                predictors[name] = predictor
        yield
        predictors.clear()

    app = FastAPI(
        title="ReflexOne",
        version=__version__,
        lifespan=lifespan,
        description="Locally trained typed decisions. Output validity does not guarantee correctness.",
    )

    @app.get("/", include_in_schema=False)
    def index():
        return RedirectResponse("/docs")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "ready": bool(predictors), "loaded_profiles": sorted(predictors)}

    @app.get("/ready")
    def readiness(response: Response) -> dict:
        missing = sorted({"support", "commands", "spam"} - predictors.keys())
        if missing:
            response.status_code = 503
        return {
            "ready": not missing,
            "loaded_profiles": sorted(predictors),
            "missing_profiles": missing,
        }

    @app.get("/profiles")
    def profiles() -> dict:
        return {
            "loaded": sorted(predictors),
            "support_domains": ["banking", "hotels"] if "support" in predictors else [],
            "max_batch_size": 32,
            "max_text_characters": 4000,
        }

    def predict_one(request: PredictRequest) -> SupportDecision | CategoricalDecision:
        if request.profile not in predictors:
            raise HTTPException(
                503, f"Profile '{request.profile}' is not loaded; train or download its artifact"
            )
        predictor = predictors[request.profile]
        if request.profile == "support":
            return predictor.predict(request.text, request.domain)
        return CategoricalDecision.model_validate(predictor.predict(request.text))

    @app.post("/predict", response_model=SupportDecision | CategoricalDecision)
    def predict(request: PredictRequest):
        return predict_one(request)

    @app.post("/predict/batch", response_model=list[SupportDecision | CategoricalDecision])
    def predict_batch(batch: BatchRequest):
        missing = sorted({r.profile for r in batch.requests if r.profile not in predictors})
        if missing:
            raise HTTPException(503, f"Profiles are not loaded: {', '.join(missing)}")
        return [predict_one(request) for request in batch.requests]

    return app
