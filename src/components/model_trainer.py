from pathlib import Path

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.utils.common import save_json
from src.utils.logger import get_logger

logger = get_logger(__name__)


def train_model(X_train, X_test, y_train, y_test, config: dict, params: dict):
    model_cfg = params["model"]
    model_name = model_cfg["name"]

    if model_name != "logistic_regression":
        raise ValueError(f"Unsupported model for now: {model_name}")

    max_iter = int(model_cfg.get("max_iter", 1000))

    logger.info("Training model: %s", model_name)

    model = LogisticRegression(
        max_iter=max_iter, random_state=params["split"]["random_state"]
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model_name": model_name,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "classification_report": classification_report(
            y_test, y_pred, output_dict=True
        ),
    }

    model_path = Path(config["artifacts"]["model_path"])
    metrics_path = Path(config["artifacts"]["metrics_path"])

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    save_json(metrics, metrics_path)

    logger.info("Model saved to %s", model_path)
    logger.info("Metrics saved to %s", metrics_path)
    logger.info(
        "Evaluation metrics | accuracy=%.4f precision=%.4f recall=%.4f f1=%.4f roc_auc=%.4f",
        metrics["accuracy"],
        metrics["precision"],
        metrics["recall"],
        metrics["f1"],
        metrics["roc_auc"],
    )

    return model, metrics
