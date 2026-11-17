from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from src.data.preprocess import preprocessor

from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

def build_model_pipeline():
    model_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    random_state=42
                )
            )
        ]
    )

    return model_pipeline

from sklearn.model_selection import StratifiedGroupKFold, cross_validate
def evaluate_model_cv(
    X_train,
    y_train,
    groups_train,
    n_splits=5,
    random_state=42
):
    cv = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state
    )

    model_pipeline = build_model_pipeline()

    scoring = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc"
    ]

    results = cross_validate(
        estimator=model_pipeline,
        X=X_train,
        y=y_train,
        cv=cv,
        groups=groups_train,
        scoring=scoring,
        return_train_score=False
    )

    return results

from sklearn.model_selection import (
    StratifiedGroupKFold,
    cross_validate,
    cross_val_predict
)

def generate_oof_probabilities(
    X_train,
    y_train,
    groups_train,
    n_splits=5,
    random_state=42
):
    cv = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state
    )

    model_pipeline = build_model_pipeline()

    y_proba_oof = cross_val_predict(
        estimator=model_pipeline,
        X=X_train,
        y=y_train,
        cv=cv,
        groups=groups_train,
        method="predict_proba"
    )

    return y_proba_oof


# Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
def build_random_forest_pipeline():
    model_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    random_state=42,
                    n_jobs=1
                )
            )
        ]
    )

    return model_pipeline

'''
One important difference from Logistic Regression: 
Random Forest does not require feature scaling. 
However, we'll retain the same preprocessing pipeline for now, including scaling, 
so that the categorical encoding and overall workflow remain consistent.
'''

param_distributions = {
    "classifier__n_estimators": randint(200, 800),
    "classifier__max_depth": [None, 5, 10, 15, 20, 30],
    "classifier__min_samples_split": randint(2, 20),
    "classifier__min_samples_leaf": randint(1, 10),
    "classifier__max_features": ["sqrt", "log2"],
    "classifier__class_weight": [None, "balanced", "balanced_subsample"]
}

'''
A few important details:
1. classifier__ prefix: Our Random Forest is the classifier step inside an sklearn Pipeline. The double underscore tells sklearn which estimator's parameter we want to tune.
2. randint(200, 800): Samples integers from 200 to 799. The upper limit is exclusive.
3. None for max_depth: Allows a tree to grow without an explicit depth limit.
4. class_weight: Tests whether assigning greater importance to the minority churn class improves performance.
'''

def tune_random_forest(
    X_train,
    y_train,
    groups_train,
    n_iter=50,
    n_splits=5,
    random_state=42
):
    cv = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state
    )

    model_pipeline = build_random_forest_pipeline()

    random_search = RandomizedSearchCV(
        estimator=model_pipeline,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring="roc_auc",
        cv=cv,
        random_state=random_state,
        n_jobs=-1,
        verbose=1,
        return_train_score=True
    )

    random_search.fit(
        X_train,
        y_train,
        groups=groups_train
    )

    return random_search

def evaluate_random_forest_cv(
    X_train,
    y_train,
    groups_train,
    n_splits=5,
    random_state=42
):
    cv = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state
    )

    model_pipeline = build_random_forest_pipeline()

    scoring = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc"
    ]

    results = cross_validate(
        estimator=model_pipeline,
        X=X_train,
        y=y_train,
        cv=cv,
        groups=groups_train,
        scoring=scoring,
        return_train_score=False,
        n_jobs=-1
    )

    return results

from sklearn.calibration import CalibratedClassifierCV
def build_calibrated_rf_pipeline(calibration_splits):
    calibrated_model = CalibratedClassifierCV(
        estimator=build_random_forest_pipeline(),
        method="sigmoid",
        cv=calibration_splits
    )

    return calibrated_model