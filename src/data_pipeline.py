import os
import re
import pandas as pd


# -------- LOAD --------
def load_data(file_path="data/raw/articles.txt"):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    articles = []
    split_articles = content.split("===ARTICLE_")

    for art in split_articles:
        if art.strip():
            # remove "1===" prefix
            cleaned = art.split("===", 1)[-1].strip()
            articles.append(cleaned)

    return pd.DataFrame({"article": articles})


# -------- CLEAN --------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# -------- FILTER --------
def filter_length(df, min_len=20, max_len=5000):
    df["length"] = df["article"].apply(len)
    return df[(df["length"] >= min_len) & (df["length"] <= max_len)]


# -------- MAIN PIPELINE --------
def preprocess_and_save():
    df = load_data()

    df["cleaned"] = df["article"].apply(clean_text)
    df = filter_length(df)

    os.makedirs("data/processed", exist_ok=True)
    df.to_csv("data/processed/preprocessed.csv", index=False)

    print("Saved to data/processed/preprocessed.csv")
    print("Total articles:", len(df))

    return df

if __name__ == "__main__":
    preprocess_and_save()