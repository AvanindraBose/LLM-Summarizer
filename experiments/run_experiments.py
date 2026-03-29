import pandas as pd
from src.pipeline import summarize_article

articles = [
    open("data/sample.txt").read(),
    "AI is rapidly evolving..." * 20,
]

prompt_versions = ["v1", "v2", "v3"]

for article in articles:
    for prompt in prompt_versions:
        summarize_article(article, prompt_version=prompt)