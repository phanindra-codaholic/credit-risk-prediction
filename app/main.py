import logging
import os
import json
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, Depends, HTTPException, status, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from sqlalchemy import text

from app.config import settings
from app.database import init_db, get_db, PredictionHistory
from app.schemas import (
    CreditApplicationRequest,
    PredictionResponse,
    PredictionRecordResponse
)
from app.predictor import predictor

# Setup Structured Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("credit_risk_api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle hook for startup and shutdown procedures."""
    logger.info("Starting up Credit Risk Prediction API...")
    try:
        init_db()
        logger.info("Database initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")

    try:
        if predictor.pipeline is None:
            predictor.load_model()
        logger.info("Machine learning model verified.")
    except Exception as e:
        logger.warning(f"Could not load ML model on startup: {e}")

    yield
    logger.info("Shutting down Credit Risk Prediction API.")


app = FastAPI(
    title="Credit Risk Prediction API",
    description="A production-grade REST API serving a trained Machine Learning model to evaluate loan default risk, backed by PostgreSQL.",
    version=settings.APP_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for frontend or client integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get(
    "/",
    tags=["General"],
    summary="Root Endpoint",
    description="Returns a welcome message, API version, and a link to the interactive documentation."
)
def root():
    return {
        "message": "Welcome to the Credit Risk Prediction API",
        "version": settings.APP_VERSION,
        "docs_url": "/docs",
        "status": "running"
    }


@app.get(
    "/health",
    tags=["System"],
    summary="Health Check",
    description="Verifies the operational status of the API, ML model, and database connection."
)
def health_check(db: Session = Depends(get_db)):
    db_status = "connected"
    try:
        db.execute(text("SELECT 1"))
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        db_status = "disconnected"

    model_status = "loaded" if predictor.pipeline is not None else "not_loaded"

    overall_healthy = (db_status == "connected") and (model_status == "loaded")
    status_code = status.HTTP_200_OK if overall_healthy else status.HTTP_503_SERVICE_UNAVAILABLE

    return {
        "status": "healthy" if overall_healthy else "degraded",
        "database": db_status,
        "model": model_status,
        "version": settings.APP_VERSION
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Inference"],
    summary="Predict Credit Risk",
    description="Evaluates a loan applicant's details and returns the default prediction, probability, and logs the result to PostgreSQL."
)
def predict_credit_risk(
    application: CreditApplicationRequest,
    db: Session = Depends(get_db)
):
    logger.info(f"Received prediction request for applicant age {application.person_age}, amount ${application.loan_amnt}")
    
    # 1. Prepare feature dictionary for model inference
    applicant_dict = application.model_dump()

    # 2. Run inference using the pre-loaded ML pipeline
    try:
        inference_result = predictor.predict(applicant_dict)
    except Exception as e:
        logger.error(f"Inference error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while generating credit risk prediction."
        )

    # 3. Persist prediction transaction to the database
    prediction_record = PredictionHistory(
        person_age=application.person_age,
        person_income=application.person_income,
        person_home_ownership=application.person_home_ownership,
        person_emp_length=application.person_emp_length,
        loan_intent=application.loan_intent,
        loan_grade=application.loan_grade,
        loan_amnt=application.loan_amnt,
        loan_int_rate=application.loan_int_rate,
        loan_percent_income=applicant_dict["loan_percent_income"],
        cb_person_default_on_file=application.cb_person_default_on_file,
        cb_person_cred_hist_length=application.cb_person_cred_hist_length,
        prediction=inference_result["prediction"],
        probability=inference_result["probability"]
    )

    try:
        db.add(prediction_record)
        db.commit()
        db.refresh(prediction_record)
        logger.info(f"Stored prediction result in database with ID: {prediction_record.id}")
        prediction_id = prediction_record.id
    except Exception as e:
        db.rollback()
        logger.error(f"Database save error: {e}", exc_info=True)
        # Even if DB fails, we still return the prediction to the client with prediction_id = None
        prediction_id = None

    return PredictionResponse(
        prediction=inference_result["prediction"],
        probability=inference_result["probability"],
        default_risk=inference_result["default_risk"],
        prediction_id=prediction_id
    )


@app.get(
    "/predictions",
    response_model=List[PredictionRecordResponse],
    tags=["History"],
    summary="List Prediction History",
    description="Retrieves stored credit risk predictions from the database with pagination."
)
def get_prediction_history(
    limit: int = Query(10, ge=1, le=100, description="Maximum records to retrieve"),
    offset: int = Query(0, ge=0, description="Offset record index"),
    db: Session = Depends(get_db)
):
    records = (
        db.query(PredictionHistory)
        .order_by(PredictionHistory.id.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )
    return records


@app.get(
    "/predictions/{prediction_id}",
    response_model=PredictionRecordResponse,
    tags=["History"],
    summary="Get Prediction By ID",
    description="Retrieves a specific prediction record from the database by its unique identifier."
)
def get_prediction_by_id(
    prediction_id: int,
    db: Session = Depends(get_db)
):
    record = db.query(PredictionHistory).filter(PredictionHistory.id == prediction_id).first()
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Prediction record with ID {prediction_id} not found."
        )
    return record


@app.get(
    "/model/metrics",
    tags=["System"],
    summary="Get Model Performance Metrics",
    description="Returns cross-validation and test set evaluation metrics for the deployed model."
)
def get_model_metrics():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    metrics_path = os.path.join(project_root, "models", "model_metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            return json.load(f)
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="Model metrics file not found."
    )
