import json
import logging
import os
from contextlib import asynccontextmanager
from typing import List, Literal, Optional

from fastapi import Depends, FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.config import settings
from app.database import PredictionHistory, get_db, init_db
from app.predictor import predictor
from app.schemas import (
    CreditApplicationRequest,
    PredictionRecordResponse,
    PredictionResponse,
    PredictionReviewUpdate,
    PredictionStatsResponse,
)
from app.sql_queries import (
    delete_prediction_sql,
    fetch_prediction_stats,
    filter_predictions_sql,
    update_prediction_review_sql,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("credit_risk_api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Credit Risk Prediction API...")
    try:
        init_db()
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")

    try:
        if predictor.pipeline is None:
            predictor.load_model()
    except Exception as e:
        logger.warning(f"Failed to load ML model on startup: {e}")

    yield
    logger.info("Shutting down Credit Risk Prediction API.")


app = FastAPI(
    title="Credit Risk Prediction API",
    description="REST API serving a Scikit-learn credit risk model with SQL persistence.",
    version=settings.APP_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["General"], summary="Root Endpoint")
def root():
    return {
        "message": "Welcome to the Credit Risk Prediction API",
        "version": settings.APP_VERSION,
        "docs_url": "/docs",
        "status": "running",
    }


@app.get("/health", tags=["System"], summary="Health Check")
def health_check(db: Session = Depends(get_db)):
    db_status = "connected"
    try:
        db.execute(text("SELECT 1"))
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        db_status = "disconnected"

    model_status = "loaded" if predictor.pipeline is not None else "not_loaded"
    overall_healthy = (db_status == "connected") and (model_status == "loaded")
    code = status.HTTP_200_OK if overall_healthy else status.HTTP_503_SERVICE_UNAVAILABLE

    return JSONResponse(
        status_code=code,
        content={
            "status": "healthy" if overall_healthy else "degraded",
            "database": db_status,
            "model": model_status,
            "version": settings.APP_VERSION,
        },
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Inference"],
    summary="Predict Credit Risk",
)
def predict_credit_risk(
    application: CreditApplicationRequest,
    db: Session = Depends(get_db),
):
    applicant_dict = application.model_dump()

    try:
        inference_result = predictor.predict(applicant_dict)
    except Exception as e:
        logger.error(f"Inference error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate credit risk prediction.",
        )

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
        probability=inference_result["probability"],
    )

    try:
        db.add(prediction_record)
        db.commit()
        db.refresh(prediction_record)
        prediction_id = prediction_record.id
    except Exception as e:
        db.rollback()
        logger.error(f"Database save error: {e}", exc_info=True)
        prediction_id = None

    return PredictionResponse(
        prediction=inference_result["prediction"],
        probability=inference_result["probability"],
        default_risk=inference_result["default_risk"],
        prediction_id=prediction_id,
    )


@app.get(
    "/predictions",
    response_model=List[PredictionRecordResponse],
    tags=["History"],
    summary="List Prediction History",
)
def get_prediction_history(
    prediction: Optional[Literal["Low Risk", "High Risk"]] = Query(None),
    loan_grade: Optional[Literal["A", "B", "C", "D", "E", "F", "G"]] = Query(None),
    review_status: Optional[Literal["PENDING", "APPROVED", "REJECTED"]] = Query(None),
    limit: int = Query(10, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
):
    return filter_predictions_sql(
        db=db,
        prediction=prediction,
        loan_grade=loan_grade,
        review_status=review_status,
        limit=limit,
        offset=offset,
    )


@app.get(
    "/predictions/stats",
    response_model=PredictionStatsResponse,
    tags=["Analytics"],
    summary="Portfolio Risk Statistics",
)
def get_prediction_statistics(db: Session = Depends(get_db)):
    return fetch_prediction_stats(db)


@app.get(
    "/predictions/{prediction_id}",
    response_model=PredictionRecordResponse,
    tags=["History"],
    summary="Get Prediction By ID",
)
def get_prediction_by_id(prediction_id: int, db: Session = Depends(get_db)):
    record = db.query(PredictionHistory).filter(PredictionHistory.id == prediction_id).first()
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Prediction record with ID {prediction_id} not found.",
        )
    return record


@app.patch(
    "/predictions/{prediction_id}",
    response_model=PredictionRecordResponse,
    tags=["History"],
    summary="Update Prediction Review Status",
)
def update_prediction_review(
    prediction_id: int,
    review: PredictionReviewUpdate,
    db: Session = Depends(get_db),
):
    updated = update_prediction_review_sql(
        db=db,
        prediction_id=prediction_id,
        review_status=review.review_status,
        reviewer_notes=review.reviewer_notes,
    )
    if not updated:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Prediction record with ID {prediction_id} not found.",
        )
    return updated


@app.delete(
    "/predictions/{prediction_id}",
    tags=["History"],
    summary="Delete Prediction Record",
)
def delete_prediction_by_id(prediction_id: int, db: Session = Depends(get_db)):
    deleted = delete_prediction_sql(db=db, prediction_id=prediction_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Prediction record with ID {prediction_id} not found.",
        )
    return {"message": f"Prediction record {prediction_id} deleted successfully."}


@app.get("/model/metrics", tags=["System"], summary="Get Model Performance Metrics")
def get_model_metrics():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    metrics_path = os.path.join(project_root, "models", "model_metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            return json.load(f)
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="Model metrics file not found.",
    )
