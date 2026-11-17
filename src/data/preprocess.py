NUMERICAL_FEATURES = [
    "Tenure Months",
    "Monthly Charges",
    "Total Charges"
]

CATEGORICAL_FEATURES = [
    "Gender",
    "Senior Citizen",
    "Partner",
    "Dependents",
    "Phone Service",
    "Multiple Lines",
    "Internet Service",
    "Online Security",
    "Online Backup",
    "Device Protection",
    "Tech Support",
    "Streaming TV",
    "Streaming Movies",
    "Contract",
    "Paperless Billing",
    "Payment Method"
]


from sklearn.pipeline import Pipeline

# Numerical pipeline
from sklearn.preprocessing import StandardScaler
numerical_pipeline = Pipeline(
    steps=[
        ("scaler", StandardScaler())
    ]
)

# One-Hot Encoding pipeline
from sklearn.preprocessing import OneHotEncoder
categorical_pipeline = Pipeline(
    steps=[
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore")
        )
    ]
)

'''
Why handle_unknown="ignore"?
Suppose our training data contains:
- Month-to-month
- One year
- Two year
But a new customer record contains a category that wasn't present during training.

Without handle_unknown="ignore", the encoder raises an error.

With this setting, it handles the unseen category by producing zeros for that feature's encoded columns.
'''

# Combine them using ColumnTransformer
from sklearn.compose import ColumnTransformer

'''
The transformers parameter accepts a list of tuples. Each tuple contains three elements:
(name, transformer, columns)
For our numerical features: ("num", numerical_pipeline, NUMERICAL_FEATURES)
'''

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_pipeline, NUMERICAL_FEATURES),
        ("cat", categorical_pipeline, CATEGORICAL_FEATURES)
    ],
    remainder="drop"
)

'''
What does remainder="drop" mean?
Any columns in the input that aren't explicitly included in either feature list will be dropped.
This is useful because we want only our selected 19 predictors to enter the model.
'''

# Create group identifiers
import pandas as pd

'''
creating group and giving them group ID while not excluding the ones with missing values.
'''

def create_groups(X: pd.DataFrame) -> pd.Series:
    return X.groupby(
        NUMERICAL_FEATURES + CATEGORICAL_FEATURES,
        dropna=False
    ).ngroup()

# Import the splitter
from sklearn.model_selection import StratifiedGroupKFold

def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    n_splits: int = 5,
    test_fold: int = 4,
    random_state: int = 42
):
    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state
    )

    for fold_idx, (train_idx, test_idx) in enumerate(
        splitter.split(X, y, groups)
    ):
        if fold_idx == test_fold:
            return (
                X.iloc[train_idx],
                X.iloc[test_idx],
                y.iloc[train_idx],
                y.iloc[test_idx]
            )