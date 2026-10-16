from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import Optional, Literal
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

    @field_validator("loan_percent_income", mode="before")
    def compute_percent_income_if_missing(cls, v, info):
        if v is None and "loan_amnt" in info.data and "person_income" in info.data:
            income = info.data["person_income"]
            amnt = info.data["loan_amnt"]
            if income > 0:
                return round(float(amnt / income), 4)
            return 1.0
        return v


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

    model_config = ConfigDict(from_attributes=True)
