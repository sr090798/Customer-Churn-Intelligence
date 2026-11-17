import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from src.models.predict import load_model, predict_churn
import logging

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

class CustomerData(BaseModel):
    # Numerical features
    tenure_months: float = Field(ge=0, le=72)
    monthly_charges: float = Field(ge=0)
    total_charges: float = Field(ge=0)

    # Categorical features
    gender: str
    senior_citizen: str
    partner: str
    dependents: str
    phone_service: str
    multiple_lines: str
    internet_service: str
    online_security: str
    online_backup: str
    device_protection: str
    tech_support: str
    streaming_tv: str
    streaming_movies: str
    contract: str
    paperless_billing: str
    payment_method: str

class BatchCustomerData(BaseModel):
    customers: list[CustomerData] = Field(min_length=1, max_length=100)

class PredictionResponse(BaseModel):
    churn_probability: float
    churn_prediction: int

class BatchPredictionResponse(BaseModel):
    total_customers: int
    predictions: list[PredictionResponse]

app = FastAPI(
    title="Customer Churn Prediction API",
    description="""
    ## Overview
    An ML-powered API for predicting customer churn.

    ## Capabilities
    - Predict churn probability for an individual customer.
    - Generate predictions for multiple customers in a single request.
    - Validate customer input before making predictions.

    ## Prediction Output
    - `churn_probability`: Estimated probability of churn.
    - `churn_prediction`: Predicted class (0 = No Churn, 1 = Churn).

    ## Model
    Random Forest Classifier trained on the Telco Customer Churn dataset.
    """,
    version="1.0.0"
)

model = load_model()

logger.info("Churn prediction model loaded successfully")

API_TO_MODEL_COLUMNS = {
    "tenure_months": "Tenure Months",
    "monthly_charges": "Monthly Charges",
    "total_charges": "Total Charges",
    "gender": "Gender",
    "senior_citizen": "Senior Citizen",
    "partner": "Partner",
    "dependents": "Dependents",
    "phone_service": "Phone Service",
    "multiple_lines": "Multiple Lines",
    "internet_service": "Internet Service",
    "online_security": "Online Security",
    "online_backup": "Online Backup",
    "device_protection": "Device Protection",
    "tech_support": "Tech Support",
    "streaming_tv": "Streaming TV",
    "streaming_movies": "Streaming Movies",
    "contract": "Contract",
    "paperless_billing": "Paperless Billing",
    "payment_method": "Payment Method"
}

def customers_to_dataframe(customers):
    """
    Convert customer data into a DataFrame
    with the original model feature names.
    """

    customers_list = [
        customer.model_dump()
        for customer in customers
    ]

    customers_list = [
        {
            API_TO_MODEL_COLUMNS[key]: value
            for key, value in customer.items()
        }
        for customer in customers_list
    ]

    return pd.DataFrame(customers_list)

@app.get("/")
def home():
    return {
        "message": "Customer Churn Prediction API is running!"
    }

@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict churn for a single customer",
    description="""
    Accepts customer information and returns the predicted churn
    probability and churn classification.

    The request must contain all 19 required customer features.

    **Output interpretation:**
    - `churn_probability`: Probability of churn, between 0 and 1.
    - `churn_prediction`: 0 = No Churn, 1 = Churn.

    **Possible responses:**
    - `200`: Prediction generated successfully.
    - `422`: Invalid customer input.
    - `500`: Unexpected internal server error.
    """,
    tags=["Predictions"],
    responses={
        200: {
            "description": "Prediction generated successfully",
            "content": {
                "application/json": {
                    "example": {
                        "churn_probability": 0.87,
                        "churn_prediction": 1
                    }
                }
            }
        }
    }
)

def predict(customer: CustomerData):

    logger.info("Received single-customer prediction request")

    try:
        customer_df = customers_to_dataframe([customer])
        result = predict_churn(model, customer_df)
        logger.info("Single-customer prediction completed successfully")

    except ValueError as e:
        logger.warning("Prediction rejected: %s", str(e))
        raise HTTPException(
            status_code=422,
            detail=str(e)
        )

    except Exception:
        logger.exception("Unexpected error during single-customer prediction")

        raise

    return {
        "churn_probability": float(result["Churn_Probability"].iloc[0]),
        "churn_prediction": int(result["Churn_Prediction"].iloc[0])
    }

@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    summary="Predict churn for multiple customers",
    description="""
    Accepts a batch of customer records and returns churn predictions
    for each customer.

    **Input requirements:**
    - Each customer must contain all 19 required features.
    - Batch size must be between 1 and 100 customers.
    - All customer records must pass input validation.

    **Output:**
    - `total_customers`: Number of customers processed.
    - `predictions`: List of individual churn probabilities and classifications.

    **Possible responses:**
    - `200`: Batch predictions generated successfully.
    - `422`: Invalid input or batch size.
    - `500`: Unexpected internal server error.
    """,
    tags=["Predictions"],
    responses={
    200: {
        "description": "Batch predictions generated successfully",
        "content": {
            "application/json": {
                "example": {
                    "total_customers": 2,
                    "predictions": [
                        {
                            "churn_probability": 0.87,
                            "churn_prediction": 1
                        },
                        {
                            "churn_probability": 0.12,
                            "churn_prediction": 0
                        }
                    ]
                }
            }
        }
    }
})
def predict_batch(batch: BatchCustomerData):

    logger.info(
        "Received batch prediction request for %d customers",
        len(batch.customers)
    )

    try:
        customers_df = customers_to_dataframe(batch.customers)
        result = predict_churn(model, customers_df)
        logger.info(
            "Batch prediction completed successfully for %d customers",
            len(batch.customers)
        )

    except ValueError as e:
        logger.warning("Batch prediction rejected: %s", str(e))
        raise HTTPException(status_code=422, detail=str(e))

    except Exception:
        logger.exception("Unexpected error during batch prediction")
        raise

    predictions = [
        {
            "churn_probability": float(row["Churn_Probability"]),
            "churn_prediction": int(row["Churn_Prediction"])
        }
        for _, row in result.iterrows()
    ]

    return {
        "total_customers": len(predictions),
        "predictions": predictions
    }
