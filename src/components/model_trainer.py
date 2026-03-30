from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import mlflow.xgboost
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
            y_test, y_pred, output_dict=True, zero_division=0
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
        )

    return models


def _log_common_params_to_mlflow(params: dict):
    mlflow.log_param("split_test_size", params["split"]["test_size"])
    mlflow.log_param("split_random_state", params["split"]["random_state"])

    benchmark_cfg = params.get("benchmark", {}).get("models", {})
    for model_name, model_cfg in benchmark_cfg.items():
        for key, value in model_cfg.items():
            mlflow.log_param(f"{model_name}_{key}", value)


def _log_model_metrics_to_mlflow(model_name: str, metrics: dict):
    mlflow.log_metric(f"{model_name}_accuracy", metrics["accuracy"])
    mlflow.log_metric(f"{model_name}_precision", metrics["precision"])
    mlflow.log_metric(f"{model_name}_recall", metrics["recall"])
    mlflow.log_metric(f"{model_name}_f1", metrics["f1"])
    mlflow.log_metric(f"{model_name}_roc_auc", metrics["roc_auc"])


def _log_best_model_to_mlflow(best_model_name: str, best_model, X_train):
    input_example = X_train.head(5)

    if best_model_name == "xgboost":
        mlflow.xgboost.log_model(
            xgb_model=best_model,
            artifact_path="best_model",
            input_example=input_example,
        )
    else:
        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="best_model",
            input_example=input_example,
        )


def train_model(X_train, X_test, y_train, y_test, config: dict, params: dict):
    models = _build_models(params)

    if not models:
        raise ValueError("No benchmark models are enabled in params.yaml")

    artifacts_cfg = config["artifacts"]
    model_path = Path(artifacts_cfg["model_path"])
    metrics_path = Path(artifacts_cfg["metrics_path"])
    benchmark_metrics_path = Path(artifacts_cfg["benchmark_metrics_path"])
    feature_importance_path = Path(artifacts_cfg["feature_importance_path"])

    model_path.parent.mkdir(parents=True, exist_ok=True)

    mlflow_cfg = params.get("mlflow", {})
    tracking_uri = artifacts_cfg.get("mlflow_tracking_uri", "file:./mlruns")
    experiment_name = mlflow_cfg.get("experiment_name", "wildfireops-benchmarking")
    log_model = bool(mlflow_cfg.get("log_model", True))

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    benchmark_results = {}
    best_model_name = None
    best_model = None
    best_score = float("-inf")

    with mlflow.start_run():
        _log_common_params_to_mlflow(params)

        for model_name, model in models.items():
            logger.info("Training model: %s", model_name)
            model.fit(X_train, y_train)

            metrics = _evaluate_model(model_name, model, X_test, y_test)
            benchmark_results[model_name] = metrics
            _log_model_metrics_to_mlflow(model_name, metrics)

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

        if feature_importance_payload is not None:
            save_json(feature_importance_payload, feature_importance_path)
            logger.info("Feature importance saved to %s", feature_importance_path)

        joblib.dump(best_model, model_path)
        save_json(best_metrics, metrics_path)
        save_json(benchmark_payload, benchmark_metrics_path)

        mlflow.set_tag("best_model", best_model_name)
        mlflow.log_artifact(str(metrics_path))
        mlflow.log_artifact(str(benchmark_metrics_path))

        if feature_importance_payload is not None:
            mlflow.log_artifact(str(feature_importance_path))

        if log_model:
            _log_best_model_to_mlflow(best_model_name, best_model, X_train)

        logger.info("Best model selected: %s", best_model_name)
        logger.info("Best model saved to %s", model_path)
        logger.info("Best-model metrics saved to %s", metrics_path)
        logger.info("Benchmark metrics saved to %s", benchmark_metrics_path)

        return best_model, benchmark_payload
