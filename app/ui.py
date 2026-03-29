import streamlit as st
from src.pipeline import summarize_article

st.set_page_config(page_title="LLM Summarizer")

st.title("🧠 LLM Article Summarizer")

text = st.text_area("Paste Article", height=300)

prompt_version = st.selectbox("Prompt Version", ["v1", "v2", "v3"])

if st.button("Summarize"):
    if text:
        with st.spinner("Generating summary..."):
            summary = summarize_article(text, prompt_version)
        st.subheader("Summary")
        st.write(summary)