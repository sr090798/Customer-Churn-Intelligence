'''
Path: Handles filesystem paths in a platform-independent way.
joblib: Loads our serialized machine learning model.

__file__ represents the location of the current Python file. (customer-churn-intelligence/src/models/predict.py)

parents[0] → src/models
parents[1] → src
parents[2] → customer-churn-intelligence
'''

'''
Defining load_model()
The function:
1. Accepts an optional model path.
2. Checks whether the file exists.
3. Loads the serialized model.
4. Returns the loaded model.
We also use a default argument so that calling load_model() automatically loads our project's saved Random Forest.
'''


from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from src.data.preprocess import (
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES
)


# --------------------------------------------------
# 1. Model loading
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT / "artifacts" / "churn_random_forest.joblib"
)


def load_model(model_path: str | Path = MODEL_PATH):
    """
    Load a serialized machine learning model.

    Parameters
    ----------
    model_path : str or Path
        Location of the saved model.

    Returns
    -------
    model
        Loaded model object.
    """

    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}"
        )

    return joblib.load(model_path)


# --------------------------------------------------
# 2. Expected input schema
# --------------------------------------------------

EXPECTED_FEATURES = (
    NUMERICAL_FEATURES + CATEGORICAL_FEATURES
)


# --------------------------------------------------
# 3. Churn prediction
# --------------------------------------------------

def predict_churn(
    model: Pipeline,
    customer_data: pd.DataFrame,
    threshold: float = 0.5
) -> pd.DataFrame:
    """
    Predict customer churn probabilities and classes.

    Parameters
    ----------
    model : Pipeline
        Fitted scikit-learn pipeline containing
        a preprocessor and classifier.

    customer_data : pd.DataFrame
        Customer records containing the expected
        numerical and categorical features.

    threshold : float, default=0.5
        Probability threshold for classifying churn.

    Returns
    -------
    pd.DataFrame
        Churn probabilities and predicted classes.
    """

    # 1. Validate threshold
    if not 0 <= threshold <= 1:
        raise ValueError(
            "Threshold must be between 0 and 1."
        )

    # 2. Validate input type
    if not isinstance(customer_data, pd.DataFrame):
        raise TypeError(
            "Customer data must be a pandas DataFrame."
        )

    if customer_data.empty:
        raise ValueError(
            "Customer data cannot be empty."
        )

    # 3. Validate duplicate column names
    if customer_data.columns.duplicated().any():

        duplicate_columns = (
            customer_data.columns[
                customer_data.columns.duplicated()
            ].tolist()
        )

        raise ValueError(
            f"Duplicate column names detected: "
            f"{duplicate_columns}"
        )

    # 4. Validate feature schema
    missing_features = (
        set(EXPECTED_FEATURES) - set(customer_data.columns)
    )

    extra_features = (
        set(customer_data.columns) - set(EXPECTED_FEATURES)
    )

    if missing_features:
        raise ValueError(
            f"Missing features: {sorted(missing_features)}"
        )

    if extra_features:
        raise ValueError(
            f"Unexpected features: {sorted(extra_features)}"
        )

    # 5. Validate model type and structure
    if not isinstance(model, Pipeline):
        raise TypeError(
            "Model must be a scikit-learn Pipeline."
        )

    required_steps = {"preprocessor", "classifier"}

    available_steps = set(model.named_steps.keys())

    if not required_steps.issubset(available_steps):
        raise ValueError(
            "Model pipeline must contain "
            "'preprocessor' and 'classifier' steps."
        )

    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

    if not isinstance(preprocessor, ColumnTransformer):
        raise TypeError(
            "The preprocessor must be a ColumnTransformer."
        )

    if not callable(getattr(classifier, "predict_proba", None)):
        raise TypeError(
            "The classifier must support predict_proba()."
        )

    # 6. Validate fitted state
    try:
        check_is_fitted(preprocessor)
        check_is_fitted(classifier)

    except NotFittedError as e:
        raise ValueError(
            "The supplied model pipeline is not fitted."
        ) from e

    # 7. Validate categorical encoder structure
    categorical_pipeline = (
        preprocessor.named_transformers_.get("cat")
    )

    if not isinstance(categorical_pipeline, Pipeline):
        raise TypeError(
            "The categorical transformer must be a Pipeline."
        )

    encoder = (
        categorical_pipeline.named_steps.get("encoder")
    )

    if not isinstance(encoder, OneHotEncoder):
        raise TypeError(
            "The categorical pipeline must contain "
            "an OneHotEncoder named 'encoder'."
        )

    check_is_fitted(encoder)

    # 8. Create a copy
    customer_data = customer_data.copy()

    # 9. Normalize categorical whitespace
    for feature in CATEGORICAL_FEATURES:

        customer_data[feature] = customer_data[feature].map(
            lambda value: value.strip()
            if isinstance(value, str)
            else value
        )

    # 10. Convert numerical features
    for feature in NUMERICAL_FEATURES:

        original_values = customer_data[feature]

        converted_values = pd.to_numeric(
            original_values,
            errors="coerce"
        )

        invalid_values = (
            original_values.notna()
            & converted_values.isna()
        )

        if invalid_values.any():
            raise ValueError(
                f"Cannot convert values in '{feature}' "
                f"to numeric: "
                f"{original_values[invalid_values].tolist()}"
            )

        customer_data[feature] = converted_values

        # Check for infinite values
        non_finite_values = (
            customer_data[feature].notna()
            & ~np.isfinite(customer_data[feature])
        )

        if non_finite_values.any():
            raise ValueError(
                f"Non-finite values detected in '{feature}'."
            )

    # 11. Validate missing values
    missing_value_features = (
        customer_data[EXPECTED_FEATURES]
        .columns[
            customer_data[EXPECTED_FEATURES]
            .isna()
            .any()
        ]
        .tolist()
    )

    if missing_value_features:
        raise ValueError(
            f"Missing values detected in features: "
            f"{missing_value_features}"
        )

    # 12. Validate categorical features
    for feature, valid_categories in zip(
        CATEGORICAL_FEATURES,
        encoder.categories_
    ):

        supplied_categories = set(
            customer_data[feature].unique()
        )

        invalid_categories = (
            supplied_categories - set(valid_categories)
        )

        if invalid_categories:
            raise ValueError(
                f"Invalid categories in '{feature}': "
                f"{sorted(invalid_categories, key=str)}. "
                f"Expected one of: {list(valid_categories)}"
            )

    # 13. Generate predictions
    churn_probability = classifier.predict_proba(
        preprocessor.transform(customer_data)
    )[:, 1]

    churn_prediction = (
        churn_probability >= threshold
    ).astype(int)

    # 14. Return results
    results = pd.DataFrame(
        {
            "Churn_Probability": churn_probability,
            "Churn_Prediction": churn_prediction
        },
        index=customer_data.index
    )

    return results
