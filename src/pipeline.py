import mlflow
from src.chunking import chunk_text
from src.prompts import PROMPTS
from src.llm_client import call_llm
import src.tracking  # initializes MLflow


def summarize_article(article, prompt_version="v1"):

    with mlflow.start_run(run_name="summarization_run"):

        mlflow.log_param("prompt_version", prompt_version)
        mlflow.log_metric("input_length", len(article))

        # -------- PREPROCESS --------
        with mlflow.start_run(run_name="preprocessing", nested=True):
            chunks = chunk_text(article)
            mlflow.log_metric("num_chunks", len(chunks))

        summaries = []

        # -------- LLM CALLS --------
        for i, chunk in enumerate(chunks):
            with mlflow.start_run(run_name=f"llm_call_{i}", nested=True):
                prompt = PROMPTS[prompt_version].format(article=chunk)
                summary, latency = call_llm(prompt)

                mlflow.log_metric("latency", latency)
                mlflow.log_text(summary, f"summary_{i}.txt")

                summaries.append(summary)

        # -------- POST PROCESS --------
        with mlflow.start_run(run_name="postprocessing", nested=True):
            final_input = " ".join(summaries)
            final_prompt = f"Combine into a 200-word summary:\n\n{final_input}"

            final_summary, latency = call_llm(final_prompt)

            mlflow.log_metric("final_latency", latency)
            mlflow.log_text(final_summary, "final_summary.txt")

        return final_summary