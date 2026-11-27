"""FastAPI application. Run from the project root:  uvicorn api.main:app --reload"""
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from .model_service import ModelService
from .schemas import BatchRequest, BatchResponse, CustomerFeatures, Prediction

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["service"] = ModelService()   # fails fast if the model files are missing
    yield
    state.clear()


app = FastAPI(
    title="Customer Churn Prediction API",
    version="1.0.0",
    description="Predicts the PROBABILITY that a customer will churn. Predictions are model estimates, "
                "not observed outcomes.",
    lifespan=lifespan,
)


def get_service() -> ModelService:
    service = state.get("service")
    if service is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return service


@app.get("/health")
def health():
    service = state.get("service")
    return {
        "status": "ok" if service else "unavailable",
        "model_loaded": service is not None,
        "model_name": service.model_name if service else None,
        "model_version": service.model_version if service else None,
    }


@app.get("/model-info")
def model_info():
    return get_service().info()


@app.post("/predict", response_model=Prediction)
def predict(customer: CustomerFeatures, explain: bool = True):
    """Score one customer. With explain=true the response lists the main factors raising/lowering the risk."""
    record = customer.model_dump()
    return get_service().predict([record], explain=explain, endpoint="predict")[0]


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(request: BatchRequest):
    """Score up to 1000 customers at once (no per-customer explanations)."""
    records = [c.model_dump() for c in request.customers]
    return {"predictions": get_service().predict(records, explain=False, endpoint="predict_batch")}
