import yaml
import joblib
import logging
from pathlib import Path
# -------- Paths --------
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

# -------- Naming --------
MODEL_NAME_MAP = {
    "logistic_regression": "lr.joblib",
    "random_forest": "rf.joblib",
    "voting": "voting.joblib"
}

def get_model_path(model_name: str) -> Path:
    try:
        return MODELS_DIR / MODEL_NAME_MAP[model_name]
    except KeyError:
        raise ValueError(f"Unsupported model: {model_name}")

   
def save_model(model, threshold, model_path):
    model_dictionary = {
        "model": model,
        "threshold": threshold
    }
    joblib.dump(model_dictionary, model_path)


def load_model(model_path):
    model_dictionary = joblib.load(model_path)
    return model_dictionary["model"], model_dictionary["threshold"]


def setup_logging(log_file="logging.log"):
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="a"),
            logging.StreamHandler()
        ]
    )

def load_config(path="config.yaml"):
    
    with open(path, "r") as f:
        return yaml.safe_load(f)
