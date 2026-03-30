from src.components.data_ingestion import ingest_data
from src.components.data_validation import validate_data
from src.components.dataset_builder import build_positive_fire_base
from src.components.negative_sampler import sample_negative_examples
from src.components.training_dataset_builder import build_training_dataset
from src.components.weather_enricher import enrich_with_weather
from src.components.feature_eng import build_features
from src.components.model_trainer import train_model
from src.utils.config import load_yaml
from src.utils.logger import get_logger

logger = get_logger(__name__)


def run_training_pipeline():
    logger.info("Starting training pipeline")

    config = load_yaml("configs/config.yaml")
    params = load_yaml("configs/params.yaml")

    df = ingest_data(config)
    validated_df, validation_report = validate_data(df, config, params)
    positive_df = build_positive_fire_base(validated_df, config)
    _ = sample_negative_examples(positive_df, config, params)
    training_df = build_training_dataset(config)
    enriched_df = enrich_with_weather(config, params)
    X_train, X_test, y_train, y_test = build_features(config, params)
    model, metrics = train_model(X_train, X_test, y_train, y_test, config, params)

    logger.info("Training pipeline completed successfully")
    logger.info("Final training dataset shape: %s", training_df.shape)
    logger.info("Weather-enriched dataset shape: %s", enriched_df.shape)
    logger.info("Final metrics: %s", metrics)

    return {
        "validation_report": validation_report,
        "training_shape": training_df.shape,
        "enriched_shape": enriched_df.shape,
        "metrics": metrics,
    }


if __name__ == "__main__":
    run_training_pipeline()