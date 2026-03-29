import pandas as pd
from src.pipeline import summarize_article
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# -------- LOAD PREPROCESSED DATA --------
df = pd.read_csv("data/processed/preprocessed.csv")

print("Total articles:", len(df))


df = df.head(10)

# -------- PROMPT VARIANTS --------
prompt_versions = ["v1", "v2", "v3"]

# -------- RUN EXPERIMENTS --------
for idx, row in df.iterrows():
    article = row["cleaned"]

    for prompt in prompt_versions:
        print(f"\nRunning article {idx} with {prompt}")

        try:
            summary = summarize_article(
                article=article,
                prompt_version=prompt
            )

            print("Summary:", summary[:150])

        except Exception as e:
            print(f"Error on article {idx}, prompt {prompt}: {e}")