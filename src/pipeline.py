import time
import mlflow
from src.chunking import chunk_text
from src.prompts import PROMPTS
from src.llm_client import call_llm
import src.tracking  # initializes MLflow


# -------- TOKEN ESTIMATION --------
def estimate_tokens(text):
    return int(len(text) / 4)


# -------- EVALUATION (LLM AS JUDGE) --------
def evaluate_summary(article, summary):
    prompt = f"""
    Rate this summary from 1-10 based on:
    - Accuracy
    - Coherence

    Return ONLY a number.

    Article:
    {article[:1000]}

    Summary:
    {summary}
    """

    score, _ = call_llm(prompt)
    return score


# -------- MAIN PIPELINE --------
def summarize_article(article, prompt_version="v1", model_name="gemini-pro"):

    start_total = time.time()

    with mlflow.start_run(run_name="summarization_run"):

        # -------- INPUT LOGGING --------
        mlflow.log_param("prompt_version", prompt_version)
        mlflow.log_param("model", model_name)
        mlflow.log_metric("input_length", len(article))
        mlflow.log_metric("input_tokens_est", estimate_tokens(article))

        # -------- PREPROCESS --------
        with mlflow.start_run(run_name="preprocessing", nested=True):
            t0 = time.time()
            chunks = chunk_text(article)
            mlflow.log_metric("num_chunks", len(chunks))
            mlflow.log_metric("preprocess_time", time.time() - t0)

        summaries = []
        total_tokens = 0

        # -------- LLM CALLS --------
        for i, chunk in enumerate(chunks):
            with mlflow.start_run(run_name=f"llm_call_{i}", nested=True):

                prompt = PROMPTS[prompt_version].format(article=chunk)

                t0 = time.time()
                summary, latency = call_llm(prompt)
                duration = time.time() - t0

                tokens = estimate_tokens(prompt + summary)
                total_tokens += tokens

                mlflow.log_metric("latency", latency)
                mlflow.log_metric("duration", duration)
                mlflow.log_metric("tokens_est", tokens)

                mlflow.log_text(prompt[:1000], f"prompt_{i}.txt")
                mlflow.log_text(summary, f"summary_{i}.txt")

                summaries.append(summary)

        # -------- POST PROCESS --------
        with mlflow.start_run(run_name="postprocessing", nested=True):

            final_input = " ".join(summaries)
            final_prompt = f"Combine into a 200-word summary:\n\n{final_input}"

            final_summary, latency = call_llm(final_prompt)

            final_tokens = estimate_tokens(final_prompt + final_summary)

            mlflow.log_metric("final_latency", latency)
            mlflow.log_metric("total_tokens_est", total_tokens + final_tokens)
            mlflow.log_text(final_summary, "final_summary.txt")

            # -------- EVALUATION TRACE --------
            with mlflow.start_run(run_name="evaluation", nested=True):

                score = evaluate_summary(article, final_summary)

                # Try to extract numeric score
                try:
                    numeric_score = float(score.strip().split()[0])
                    mlflow.log_metric("quality_score", numeric_score)
                except:
                    mlflow.log_text(score, "quality_score_raw.txt")

                mlflow.set_tag("evaluation_method", "llm_as_judge")

        # -------- TOTAL TRACE --------
        total_time = time.time() - start_total
        mlflow.log_metric("total_time", total_time)

        return final_summary