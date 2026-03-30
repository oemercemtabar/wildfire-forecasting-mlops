from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

from src.utils.common import save_json
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _extract_feature_importance(model_name, model, feature_names):
    if not hasattr(model, "feature_importances_"):
        return None

    importances = model.feature_importances_
    pairs = [
        {"feature": feature, "importance": float(importance)}
        for feature, importance in zip(feature_names, importances)
    ]
    pairs = sorted(pairs, key=lambda x: x["importance"], reverse=True)

    return {
        "model_name": model_name,
        "feature_importance": pairs,
    }


def _evaluate_model(model_name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    return {
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


def _build_models(params: dict):
    benchmark_cfg = params["benchmark"]
    random_state = params["split"]["random_state"]

    models = {}

    lr_cfg = benchmark_cfg["models"]["logistic_regression"]
    if lr_cfg.get("enabled", False):
        models["logistic_regression"] = LogisticRegression(
            max_iter=int(lr_cfg.get("max_iter", 1000)),
            random_state=random_state,
        )

    rf_cfg = benchmark_cfg["models"]["random_forest"]
    if rf_cfg.get("enabled", False):
        models["random_forest"] = RandomForestClassifier(
            n_estimators=int(rf_cfg.get("n_estimators", 200)),
            max_depth=rf_cfg.get("max_depth"),
            random_state=random_state,
            n_jobs=-1,
        )

    xgb_cfg = benchmark_cfg["models"]["xgboost"]
    if xgb_cfg.get("enabled", False):
        models["xgboost"] = XGBClassifier(
            n_estimators=int(xgb_cfg.get("n_estimators", 200)),
            max_depth=int(xgb_cfg.get("max_depth", 6)),
            learning_rate=float(xgb_cfg.get("learning_rate", 0.1)),
            subsample=float(xgb_cfg.get("subsample", 0.8)),
            colsample_bytree=float(xgb_cfg.get("colsample_bytree", 0.8)),
            eval_metric=xgb_cfg.get("eval_metric", "logloss"),
            random_state=random_state,
            use_label_encoder=False,
        )

    return models


def train_model(X_train, X_test, y_train, y_test, config: dict, params: dict):
    models = _build_models(params)

    if not models:
        raise ValueError("No benchmark models are enabled in params.yaml")

    benchmark_results = {}
    best_model_name = None
    best_model = None
    best_score = float("-inf")

    for model_name, model in models.items():
        logger.info("Training model: %s", model_name)
        model.fit(X_train, y_train)

        metrics = _evaluate_model(model_name, model, X_test, y_test)
        benchmark_results[model_name] = metrics

        logger.info(
            "%s | accuracy=%.4f precision=%.4f recall=%.4f f1=%.4f roc_auc=%.4f",
            model_name,
            metrics["accuracy"],
            metrics["precision"],
            metrics["recall"],
            metrics["f1"],
            metrics["roc_auc"],
        )

        if metrics["roc_auc"] > best_score:
            best_score = metrics["roc_auc"]
            best_model_name = model_name
            best_model = model

    if best_model is None:
        raise RuntimeError("Failed to select a best model during benchmarking.")

    model_path = Path(config["artifacts"]["model_path"])
    metrics_path = Path(config["artifacts"]["metrics_path"])
    benchmark_metrics_path = Path(config["artifacts"]["benchmark_metrics_path"])

    model_path.parent.mkdir(parents=True, exist_ok=True)

    best_metrics = benchmark_results[best_model_name]
    best_metrics["selected_as_best_model"] = True

    benchmark_payload = {
        "best_model_name": best_model_name,
        "selection_metric": "roc_auc",
        "models": benchmark_results,
    }

    feature_names = list(X_train.columns)
    feature_importance_payload = _extract_feature_importance(
        best_model_name,
        best_model,
        feature_names,
    )

    feature_importance_path = Path(config["artifacts"]["feature_importance_path"])

    if feature_importance_payload is not None:
        save_json(feature_importance_payload, feature_importance_path)
        logger.info("Feature importance saved to %s", feature_importance_path)

    joblib.dump(best_model, model_path)
    save_json(best_metrics, metrics_path)
    save_json(benchmark_payload, benchmark_metrics_path)

    logger.info("Best model selected: %s", best_model_name)
    logger.info("Best model saved to %s", model_path)
    logger.info("Best-model metrics saved to %s", metrics_path)
    logger.info("Benchmark metrics saved to %s", benchmark_metrics_path)

    return best_model, benchmark_payload
