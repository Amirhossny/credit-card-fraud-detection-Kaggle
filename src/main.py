import argparse
import logging
from pathlib import Path
from src.training import ModelTrainer
from src.eval import best_threshold, evaluate
from src.Data_utils import (
    preprocessing,
    build_clipping_transformer,
    prepare_data
)

from .helper_fun import (
    setup_logging,
    load_config,
    get_model_path,
    save_model,
    load_model
)



def train_and_validate(config, model_name, splits):
    logging.info(f"Trainig & validation for model: {model_name}")
    
    X_train, y_train = splits["train"]
    X_val, y_val = splits["val"]

    preprocessor = preprocessing()
    clipping_transformer = build_clipping_transformer()
    
    trainer = ModelTrainer(
        config=config,
        preprocessor=preprocessor,
        clipping_transformer=clipping_transformer
    )

    if model_name in ["logistic_regression", "random_forest"]:
        grid = trainer.train(
            X=X_train,
            y=y_train,
            model_name=model_name
        )
        model = grid.best_estimator_

    elif model_name == "voting":
        lr = trainer._build_model("logistic_regression")
        rf = trainer._build_model("random_forest")

        voting_model = trainer.simple_voting(lr, rf)
        model = trainer._build_pipeline(voting_model)
        model.fit(X_train, y_train)

    else:
        raise ValueError("Unsupported model")

    threshold = best_threshold(model, X_val, y_val)
    evaluate(model, X_val, y_val, threshold, logger=logging.getLogger())

    model_path = get_model_path(model_name)
    save_model(model=model, threshold=threshold, model_path=model_path)
    


def test_model(model_name, splits):
    
    logging.info(f"Testing model: {model_name}")
    
    X_test, y_test = splits["test"]
  
    model_path = get_model_path(model_name)
    model, threshold = load_model(model_path=model_path)
   
    evaluate(model, X_test, y_test, threshold, logger=logging.getLogger())
    
def main():
    parser = argparse.ArgumentParser(description="Training, Validation & Testing Pipeline")
    parser.add_argument("--model",
                        type=str, required=True, choices=["logistic_regression", "random_forest", "voting"])
    parser.add_argument("--mode", type=str, required=True, choices=["train", "test"])
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")

    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent.parent
    setup_logging(log_file=base_dir / "training.log")
    config_path = base_dir / "src" /args.config
    config = load_config(config_path)

    splits = prepare_data(config, base_dir)

    if args.mode == "train":
        train_and_validate(config, args.model, splits)
    elif args.mode == "test":
        test_model(args.model, splits)


if __name__ == "__main__":
    main()