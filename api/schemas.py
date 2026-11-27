"""Request / response schemas of the churn API.

JSON keys are snake_case (tenure_months, ...); the service maps them back to the
column names used during training ("Tenure Months", ...).
"""
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

YesNo = Literal["Yes", "No"]
AddOn = Literal["Yes", "No", "No internet service"]
ADDON_FIELDS = ["online_security", "online_backup", "device_protection", "tech_support",
                "streaming_tv", "streaming_movies"]


class CustomerFeatures(BaseModel):
    """One customer, described with the attributes available at prediction time."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"example": {
            "customer_id": "0001-DEMO",
            "senior_citizen": "No", "partner": "No", "dependents": "No",
            "tenure_months": 2, "phone_service": "Yes", "multiple_lines": "No",
            "internet_service": "Fiber optic", "online_security": "No", "online_backup": "No",
            "device_protection": "No", "tech_support": "No", "streaming_tv": "No",
            "streaming_movies": "No", "contract": "Month-to-month", "paperless_billing": "Yes",
            "payment_method": "Electronic check", "monthly_charges": 70.7, "total_charges": 151.65,
        }},
    )

    customer_id: Optional[str] = Field(None, description="Optional identifier, only echoed back and logged.")
    senior_citizen: YesNo
    partner: YesNo
    dependents: YesNo
    tenure_months: int = Field(ge=0, le=120)
    phone_service: YesNo
    multiple_lines: Literal["Yes", "No", "No phone service"]
    internet_service: Literal["DSL", "Fiber optic", "No"]
    online_security: AddOn
    online_backup: AddOn
    device_protection: AddOn
    tech_support: AddOn
    streaming_tv: AddOn
    streaming_movies: AddOn
    contract: Literal["Month-to-month", "One year", "Two year"]
    paperless_billing: YesNo
    payment_method: Literal["Electronic check", "Mailed check",
                            "Bank transfer (automatic)", "Credit card (automatic)"]
    monthly_charges: float = Field(gt=0, le=500)
    total_charges: Optional[float] = Field(
        None, ge=0, description="If omitted, estimated as tenure_months x monthly_charges.")

    @model_validator(mode="after")
    def check_business_rules(self):
        # Combinations that cannot occur in the training data are rejected rather than silently scored.
        if self.phone_service == "No" and self.multiple_lines != "No phone service":
            raise ValueError("multiple_lines must be 'No phone service' when phone_service is 'No'")
        if self.phone_service == "Yes" and self.multiple_lines == "No phone service":
            raise ValueError("multiple_lines cannot be 'No phone service' when phone_service is 'Yes'")
        addons = [getattr(self, f) for f in ADDON_FIELDS]
        if self.internet_service == "No" and any(a != "No internet service" for a in addons):
            raise ValueError("all add-on services must be 'No internet service' when internet_service is 'No'")
        if self.internet_service != "No" and any(a == "No internet service" for a in addons):
            raise ValueError("add-on services cannot be 'No internet service' when the customer has internet")
        return self


class RiskFactor(BaseModel):
    feature: str
    value: str
    impact: float = Field(description="Change in churn probability compared with an average customer "
                                      "(positive = raises the risk, negative = lowers it).")


class Prediction(BaseModel):
    """A PREDICTED risk. The customer's actual outcome is unknown until it is observed."""

    customer_id: Optional[str] = None
    churn_probability: float = Field(description="Predicted probability of churn, between 0 and 1.")
    predicted_churn: bool = Field(description="True when churn_probability >= the model's decision threshold.")
    risk_level: Literal["low", "medium", "high"]
    threshold: float
    model_name: str
    model_version: str
    top_risk_factors: list[RiskFactor] = []
    top_protective_factors: list[RiskFactor] = []


class BatchRequest(BaseModel):
    customers: list[CustomerFeatures] = Field(min_length=1, max_length=1000)


class BatchResponse(BaseModel):
    predictions: list[Prediction]
