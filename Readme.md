# End-to-End LLM-Powered Article Summarization Pipeline

## Overview

This project implements a complete, locally deployable pipeline for abstractive summarization of news articles using Large Language Models (LLMs). The system is designed to demonstrate core LLMOps principles including data handling, prompt engineering, observability (tracing and tracking), experimentation, and deployment.

The pipeline accepts long-form articles (500–2000+ words) and produces concise summaries (100–300 words) while preserving key facts, entities, and overall tone.

---

## Key Features

* End-to-end LLM pipeline from raw data ingestion to summary generation
* Support for long articles via chunking and aggregation
* Multiple prompt engineering strategies for experimentation
* Full observability using MLflow:

  * Nested tracing (pipeline step-level spans)
  * Run tracking (parameters, outputs, metrics)
* Experiment comparison across prompts and inputs
* Streamlit-based interactive UI
* FastAPI endpoint for programmatic access
* Fully containerized using Docker
* Local-first design (no mandatory cloud dependencies)

---

## Architecture

```
User Input (UI/API)
        ↓
Data Ingestion (raw text / URL)
        ↓
Preprocessing Pipeline
        ↓
Chunking (for long inputs)
        ↓
Prompt Construction (versioned)
        ↓
LLM Inference
        ↓
Post-processing (merge summaries)
        ↓
Evaluation (LLM-as-Judge)
        ↓
MLflow Tracking & Tracing
        ↓
Final Output
```

---

## Project Structure

```
llm-summarizer/
│
├── app/
│   ├── ui.py              # Streamlit UI
│   └── api.py             # FastAPI endpoint
│
├── src/
│   ├── pipeline.py        # Main summarization pipeline
│   ├── llm_client.py      # LLM interaction layer
│   ├── prompts.py         # Prompt templates
│   ├── chunking.py        # Text chunking logic
│   ├── data_pipeline.py   # Data loading and preprocessing
│   ├── tracking.py        # MLflow configuration
│
├── experiments/
│   └── run_experiments.py # Batch experiment runner
│
├── data/
│   ├── raw/               # Raw articles
│   └── processed/         # Preprocessed dataset
│
├── mlruns/                # MLflow tracking data
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
└── README.md
```

---

## Data Handling

### Dataset

A lightweight local dataset (50 articles) is used for experimentation. Articles are stored in a single text file and processed into a structured dataset.

### Preprocessing Pipeline

The preprocessing pipeline performs:

* Text normalization (lowercasing, whitespace cleanup)
* Length-based filtering
* Dataset structuring into tabular format

Output:

```
data/processed/preprocessed.csv
```

Columns:

* `article`
* `cleaned`
* `length`

---

## Model Inference

### LLM Provider

The system supports LLM inference via:

* Google Gemini API (primary)
* Hugging Face Inference API (fallback)

### Prompt Engineering

Multiple prompt variants are implemented:

* v1: Basic summarization
* v2: Role-based (journalist style)
* v3: Fact-preserving structured summary

This enables systematic comparison of prompt effectiveness.

---

## Handling Long Inputs

To address context window limits:

1. Articles are split into chunks
2. Each chunk is summarized independently
3. Partial summaries are merged into a final summary

This follows a map-reduce style summarization strategy.

---

## Observability (MLflow)

This project uses MLflow to implement both tracing and tracking.

### Tracing

Nested MLflow runs capture the full pipeline execution:

```
summarization_run
 ├── preprocessing
 ├── llm_call_0
 ├── llm_call_1
 ├── postprocessing
      └── evaluation
```

Each step logs:

* Latency
* Token estimates
* Inputs and outputs

### Tracking

Each run logs:

* Prompt version
* Model used
* Input length
* Token estimates
* Latency
* Final summary
* Evaluation score

---

## Evaluation

An LLM-as-Judge approach is used to evaluate summary quality.

Metrics logged:

* Quality score (numeric)
* Evaluation text (raw output)

This enables comparison across prompt strategies.

---

## Experimentation

Experiments are conducted using:

```
experiments/run_experiments.py
```

For each article:

* Multiple prompt variants are executed
* Results are logged in MLflow

This enables:

* Prompt comparison
* Latency vs quality trade-off analysis

---

## User Interface

A Streamlit application provides an interactive interface.

Features:

* Paste article text
* Select prompt variant
* Generate summary
* View original vs summary side-by-side
* Display latency information

Run:

```
streamlit run app/ui.py
```

---

## API Endpoint

A FastAPI service allows programmatic access.

Endpoint:

```
POST /summarize
```

Input:

```
{
  "text": "...",
  "prompt_version": "v1"
}
```

Run:

```
uvicorn app.api:app --reload
```

---

## Setup Instructions

### Local Setup

```
pip install .
```

### Run MLflow

```
mlflow ui
```

Open:

```
http://127.0.0.1:5000
```

### Run Experiments

```
python -m experiments.run_experiments
```

---

## Docker Setup

Build and run:

```
docker-compose up --build
```

Services:

* Streamlit UI
* MLflow tracking server

---

## Results

### Observability

* Full trace visualization using nested MLflow runs
* Latency and token tracking across pipeline steps

### Experiment Comparison

Prompt variants were compared across:

* Latency
* Token usage
* Quality score

### Example Results

| Prompt | Latency | Tokens | Quality Score |
| ------ | ------- | ------ | ------------- |
| v1     | Low     | Medium | Moderate      |
| v2     | Medium  | Medium | High          |
| v3     | Medium  | High   | Highest       |

---

## Screenshots (To Include)

* Streamlit UI (input vs output)
* MLflow run list
* MLflow trace view (nested runs)
* Metrics comparison charts

---

## Key Design Decisions

* Local-first architecture to avoid cloud dependency
* MLflow for unified tracking and tracing
* Modular pipeline for extensibility
* Prompt versioning for systematic experimentation
* Map-reduce summarization for scalability

---

## Future Improvements

* Parallel chunk processing
* Token-accurate accounting
* Automatic prompt selection
* Advanced evaluation metrics (ROUGE, BERTScore)
* Support for multiple LLM providers dynamically

---

## Conclusion

This project demonstrates a production-style LLM pipeline with strong emphasis on observability, modularity, and experimentation. It highlights practical challenges such as API limitations, prompt optimization, and system debugging, while providing a scalable foundation for real-world applications.
