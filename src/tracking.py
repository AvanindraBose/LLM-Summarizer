import mlflow
import os

BASE_DIR = os.getcwd()
MLFLOW_DIR = os.path.join(BASE_DIR, "mlruns")

os.makedirs(MLFLOW_DIR, exist_ok=True)

mlflow.set_tracking_uri("file:///" + MLFLOW_DIR.replace("\\", "/"))
mlflow.set_experiment("llm_summarization")