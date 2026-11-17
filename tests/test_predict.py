import numpy as np
import pandas as pd
import pytest

from src.models.predict import load_model, predict_churn


def test_classifier_rejects_invalid_classes():

    model = load_model()

    customer_data = pd.DataFrame({
        "Tenure Months": [12],
        "Monthly Charges": [75.5],
        "Total Charges": [900.0],
        "Gender": ["Female"],
        "Senior Citizen": ["No"],
        "Partner": ["No"],
        "Dependents": ["No"],
        "Phone Service": ["Yes"],
        "Multiple Lines": ["No"],
        "Internet Service": ["Fiber optic"],
        "Online Security": ["No"],
        "Online Backup": ["Yes"],
        "Device Protection": ["No"],
        "Tech Support": ["No"],
        "Streaming TV": ["Yes"],
        "Streaming Movies": ["No"],
        "Contract": ["Month-to-month"],
        "Paperless Billing": ["Yes"],
        "Payment Method": ["Electronic check"]
    })

    model.named_steps["classifier"].classes_ = np.array([0])

    with pytest.raises(
        ValueError,
        match="classifier must be fitted with binary classes 0 and 1"
    ):
        predict_churn(model, customer_data)