import streamlit as st
import time
from src.pipeline import summarize_article

st.title("LLM Summarizer")

text = st.text_area("Paste Article")

prompt = st.selectbox("Prompt", ["v1", "v2", "v3"])

if st.button("Summarize"):
    start = time.time()

    summary = summarize_article(text, prompt)

    latency = time.time() - start

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original")
        st.write(text[:2000])

    with col2:
        st.subheader("Summary")
        st.write(summary)

    st.info(f"Latency: {latency:.2f} sec")