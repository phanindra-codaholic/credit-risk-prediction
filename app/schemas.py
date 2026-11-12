from pydantic import BaseModel, Field, model_validator, ConfigDict
from typing import Optional, Literal, List
from datetime import datetime

class CreditApplicationRequest(BaseModel):
    person_age: int = Field(..., ge=18, le=100, description="Age of the loan applicant in years (18-100)", json_schema_extra={"example": 29})
    person_income: float = Field(..., ge=0.0, description="Annual income of the applicant in USD", json_schema_extra={"example": 65000.0})
    person_home_ownership: Literal["RENT", "OWN", "MORTGAGE", "OTHER"] = Field(
        ..., description="Home ownership status", json_schema_extra={"example": "RENT"}
    )
    person_emp_length: Optional[float] = Field(
        None, ge=0.0, le=100.0, description="Length of employment in years", json_schema_extra={"example": 4.0}
    )
    loan_intent: Literal["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"] = Field(
        ..., description="Intent/purpose of the loan", json_schema_extra={"example": "EDUCATION"}
    )
    loan_grade: Literal["A", "B", "C", "D", "E", "F", "G"] = Field(
        ..., description="Credit risk grade assigned to the loan", json_schema_extra={"example": "B"}
    )
    loan_amnt: float = Field(..., gt=0.0, description="Requested loan amount in USD", json_schema_extra={"example": 10000.0})
    loan_int_rate: Optional[float] = Field(
        None, ge=0.0, le=100.0, description="Interest rate as a percentage (e.g., 11.5 for 11.5%)", json_schema_extra={"example": 11.26}
    )
    loan_percent_income: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Ratio of loan amount to annual income (auto-calculated if omitted)", json_schema_extra={"example": 0.15}
    )
    cb_person_default_on_file: Literal["Y", "N"] = Field(
        ..., description="Historical default on credit bureau record ('Y' or 'N')", json_schema_extra={"example": "N"}
    )
    cb_person_cred_hist_length: int = Field(
        ..., ge=0, le=100, description="Credit history length in years", json_schema_extra={"example": 4}
    )

    @model_validator(mode="after")
    def compute_percent_income_if_missing(self):
        if self.loan_percent_income is None:
            if self.person_income > 0:
                self.loan_percent_income = min(1.0, round(float(self.loan_amnt / self.person_income), 4))
            else:
                self.loan_percent_income = 1.0
        return self


class PredictionResponse(BaseModel):
    prediction: str = Field(..., description="'Low Risk' (Non-Default) or 'High Risk' (Default)", json_schema_extra={"example": "Low Risk"})
    probability: float = Field(..., description="Estimated probability of default (0.0 to 1.0)", json_schema_extra={"example": 0.1245})
    default_risk: str = Field(..., description="Descriptive risk outcome", json_schema_extra={"example": "Likely Non-Default"})
    prediction_id: Optional[int] = Field(None, description="Database record ID for the saved prediction", json_schema_extra={"example": 1})


class PredictionRecordResponse(BaseModel):
    id: int
    created_at: datetime
    person_age: int
    person_income: float
    person_home_ownership: str
    person_emp_length: Optional[float]
    loan_intent: str
    loan_grade: str
    loan_amnt: float
    loan_int_rate: Optional[float]
    loan_percent_income: float
    cb_person_default_on_file: str
    cb_person_cred_hist_length: int
    prediction: str
    probability: float
    review_status: str = "PENDING"
    reviewer_notes: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


class PredictionReviewUpdate(BaseModel):
    review_status: Literal["PENDING", "APPROVED", "REJECTED"] = Field(
        ..., description="Underwriter review decision", json_schema_extra={"example": "APPROVED"}
    )
    reviewer_notes: Optional[str] = Field(
        None, max_length=500, description="Optional notes from underwriter review", json_schema_extra={"example": "Verified income statements."}
    )


class GradeStats(BaseModel):
    loan_grade: str
    count: int
    high_risk_count: int
    avg_probability: float


class IntentStats(BaseModel):
    loan_intent: str
    count: int
    high_risk_count: int
    avg_probability: float


class PredictionStatsResponse(BaseModel):
    total_predictions: int
    high_risk_count: int
    low_risk_count: int
    high_risk_rate_pct: float
    avg_probability: float
    avg_loan_amount: float
    avg_person_income: float
    by_loan_grade: List[GradeStats]
    by_loan_intent: List[IntentStats]
