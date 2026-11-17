from fastapi.testclient import TestClient

from api.main import app

from unittest.mock import patch

client = TestClient(
    app,
    raise_server_exceptions=False
)

valid_customer = {
    "tenure_months": 12,
    "monthly_charges": 75.5,
    "total_charges": 900.0,
    "gender": "Female",
    "senior_citizen": "No",
    "partner": "No",
    "dependents": "No",
    "phone_service": "Yes",
    "multiple_lines": "No",
    "internet_service": "Fiber optic",
    "online_security": "No",
    "online_backup": "Yes",
    "device_protection": "No",
    "tech_support": "No",
    "streaming_tv": "Yes",
    "streaming_movies": "No",
    "contract": "Month-to-month",
    "paperless_billing": "Yes",
    "payment_method": "Electronic check"
}


def test_home_endpoint():

    response = client.get("/")

    assert response.status_code == 200

    assert response.json() == {
        "message": "Customer Churn Prediction API is running!"
    }

def test_single_customer_prediction():

    customer = {
        "tenure_months": 12,
        "monthly_charges": 75.5,
        "total_charges": 900.0,
        "gender": "Female",
        "senior_citizen": "No",
        "partner": "No",
        "dependents": "No",
        "phone_service": "Yes",
        "multiple_lines": "No",
        "internet_service": "Fiber optic",
        "online_security": "No",
        "online_backup": "Yes",
        "device_protection": "No",
        "tech_support": "No",
        "streaming_tv": "Yes",
        "streaming_movies": "No",
        "contract": "Month-to-month",
        "paperless_billing": "Yes",
        "payment_method": "Electronic check"
    }

    response = client.post("/predict", json=customer)

    assert response.status_code == 200

    result = response.json()

    assert "churn_probability" in result
    assert "churn_prediction" in result

    assert 0 <= result["churn_probability"] <= 1
    assert result["churn_prediction"] in [0, 1]

# Test invalid customer data
def test_invalid_category():

    customer = {
        "tenure_months": 12,
        "monthly_charges": 75.5,
        "total_charges": 900.0,
        "gender": "Female",
        "senior_citizen": "No",
        "partner": "No",
        "dependents": "No",
        "phone_service": "Yes",
        "multiple_lines": "No",
        "internet_service": "Fiber optic",
        "online_security": "No",
        "online_backup": "Yes",
        "device_protection": "No",
        "tech_support": "No",
        "streaming_tv": "Yes",
        "streaming_movies": "No",
        "contract": "Quarterly",
        "paperless_billing": "Yes",
        "payment_method": "Electronic check"
    }

    response = client.post("/predict", json=customer)

    assert response.status_code == 422

    assert "Invalid categories" in response.json()["detail"]

# Test missing input fields
def test_missing_input_field():

    customer = {
        "tenure_months": 12,
        "monthly_charges": 75.5,
        # total_charges is intentionally missing
        "gender": "Female",
        "senior_citizen": "No",
        "partner": "No",
        "dependents": "No",
        "phone_service": "Yes",
        "multiple_lines": "No",
        "internet_service": "Fiber optic",
        "online_security": "No",
        "online_backup": "Yes",
        "device_protection": "No",
        "tech_support": "No",
        "streaming_tv": "Yes",
        "streaming_movies": "No",
        "contract": "Month-to-month",
        "paperless_billing": "Yes",
        "payment_method": "Electronic check"
    }

    response = client.post("/predict", json=customer)

    assert response.status_code == 422

    assert any(
        error["loc"][-1] == "total_charges"
        for error in response.json()["detail"]
    )

# Test batch prediction
def test_batch_prediction():

    customers = {
        "customers": [
            {
                "tenure_months": 12,
                "monthly_charges": 75.5,
                "total_charges": 900.0,
                "gender": "Female",
                "senior_citizen": "No",
                "partner": "No",
                "dependents": "No",
                "phone_service": "Yes",
                "multiple_lines": "No",
                "internet_service": "Fiber optic",
                "online_security": "No",
                "online_backup": "Yes",
                "device_protection": "No",
                "tech_support": "No",
                "streaming_tv": "Yes",
                "streaming_movies": "No",
                "contract": "Month-to-month",
                "paperless_billing": "Yes",
                "payment_method": "Electronic check"
            },
            {
                "tenure_months": 60,
                "monthly_charges": 45.0,
                "total_charges": 2700.0,
                "gender": "Male",
                "senior_citizen": "No",
                "partner": "Yes",
                "dependents": "Yes",
                "phone_service": "Yes",
                "multiple_lines": "No",
                "internet_service": "DSL",
                "online_security": "Yes",
                "online_backup": "Yes",
                "device_protection": "Yes",
                "tech_support": "Yes",
                "streaming_tv": "No",
                "streaming_movies": "No",
                "contract": "Two year",
                "paperless_billing": "No",
                "payment_method": "Bank transfer (automatic)"
            }
        ]
    }

    response = client.post("/predict/batch", json=customers)

    assert response.status_code == 200

    result = response.json()

    assert result["total_customers"] == 2
    assert len(result["predictions"]) == 2

    for prediction in result["predictions"]:
        assert 0 <= prediction["churn_probability"] <= 1
        assert prediction["churn_prediction"] in [0, 1]

# Test batch and individual prediction consistency
def test_batch_individual_consistency():

    customer = {
        "tenure_months": 12,
        "monthly_charges": 75.5,
        "total_charges": 900.0,
        "gender": "Female",
        "senior_citizen": "No",
        "partner": "No",
        "dependents": "No",
        "phone_service": "Yes",
        "multiple_lines": "No",
        "internet_service": "Fiber optic",
        "online_security": "No",
        "online_backup": "Yes",
        "device_protection": "No",
        "tech_support": "No",
        "streaming_tv": "Yes",
        "streaming_movies": "No",
        "contract": "Month-to-month",
        "paperless_billing": "Yes",
        "payment_method": "Electronic check"
    }

    # Individual prediction
    single_response = client.post("/predict", json=customer)

    # Batch prediction containing the same customer
    batch_response = client.post(
        "/predict/batch",
        json={"customers": [customer]}
    )

    assert single_response.status_code == 200
    assert batch_response.status_code == 200

    single_result = single_response.json()
    batch_result = batch_response.json()["predictions"][0]

    assert abs(
        single_result["churn_probability"]
        - batch_result["churn_probability"]
    ) < 1e-10

    assert (
        single_result["churn_prediction"]
        == batch_result["churn_prediction"]
    )

def test_empty_batch_rejected():

    response = client.post(
        "/predict/batch",
        json={"customers": []}
    )

    assert response.status_code == 422

    errors = response.json()["detail"]

    assert any(
        error["loc"][-1] == "customers"
        for error in errors
    )

def test_negative_tenure_rejected():

    customer = valid_customer.copy()
    customer["tenure_months"] = -5

    response = client.post("/predict", json=customer)

    assert response.status_code == 422

    errors = response.json()["detail"]

    assert any(
        error["loc"][-1] == "tenure_months"
        for error in errors
    )

def test_negative_monthly_charges_rejected():

    customer = valid_customer.copy()
    customer["monthly_charges"] = -100

    response = client.post("/predict", json=customer)

    assert response.status_code == 422

    errors = response.json()["detail"]

    assert any(
        error["loc"][-1] == "monthly_charges"
        for error in errors
    )

def test_negative_total_charges_rejected():

    customer = valid_customer.copy()
    customer["total_charges"] = -500

    response = client.post("/predict", json=customer)

    assert response.status_code == 422

    errors = response.json()["detail"]

    assert any(
        error["loc"][-1] == "total_charges"
        for error in errors
    )

def test_batch_size_limit():

    customers = [
        valid_customer.copy()
        for _ in range(101)
    ]

    response = client.post(
        "/predict/batch",
        json={"customers": customers}
    )

    assert response.status_code == 422

    errors = response.json()["detail"]

    assert any(
        error["loc"][-1] == "customers"
        for error in errors
    )

def test_unexpected_prediction_error():

    with patch(
        "api.main.predict_churn",
        side_effect=RuntimeError("Unexpected internal error")
    ):
        response = client.post(
            "/predict",
            json=valid_customer
        )

    assert response.status_code == 500
    assert response.text == "Internal Server Error"

def test_invalid_category_in_batch():
    invalid_customer = valid_customer.copy()
    invalid_customer["contract"] = "Quarterly"

    response = client.post(
        "/predict/batch",
        json={
            "customers": [
                valid_customer,
                invalid_customer
            ]
        }
    )

    assert response.status_code == 422
    assert "Invalid categories" in response.json()["detail"]

def test_unexpected_batch_prediction_error():
    with patch(
        "api.main.predict_churn",
        side_effect=RuntimeError("Unexpected internal error")
    ):
        response = client.post(
            "/predict/batch",
            json={"customers": [valid_customer]}
        )

    assert response.status_code == 500

def test_invalid_batch_generates_warning_log(caplog):
    invalid_customer = valid_customer.copy()
    invalid_customer["contract"] = "Quarterly"

    with caplog.at_level("WARNING", logger="api.main"):
        response = client.post(
            "/predict/batch",
            json={"customers": [valid_customer, invalid_customer]}
        )

    assert response.status_code == 422

    assert any(
        "Batch prediction rejected" in record.message
        and record.levelname == "WARNING"
        for record in caplog.records
    )

def test_unexpected_batch_error_generates_error_log(caplog):
    with patch(
        "api.main.predict_churn",
        side_effect=RuntimeError("Unexpected internal error")
    ):
        with caplog.at_level("ERROR", logger="api.main"):
            response = client.post(
                "/predict/batch",
                json={"customers": [valid_customer]}
            )

    assert response.status_code == 500

    assert any(
        "Unexpected error during batch prediction" in record.message
        and record.levelname == "ERROR"
        and record.exc_info is not None
        for record in caplog.records
    )