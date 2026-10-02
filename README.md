# 🤖 Autonomous Self-Healing Data Analyst Agent

An enterprise-grade autonomous data analysis agent that accepts **CSV, Excel (.xlsx, .xls), or JSON** files, finds patterns, and generates professional interactive charts and insights. 

It writes and executes Python code in an isolated sandbox, leverages a **Retrieval-Augmented Generation (RAG)** engine over official Python/Pandas/Plotly documentation to recover from runtime errors, and **self-heals** iteratively until clean, verified output is produced.

---

## 🌟 Key Features

1. **Multi-Format Ingestion**:
   - **CSV**: Auto-detects delimiters (comma, tab, semicolon), encodings (UTF-8, Latin1), and handles messy headers.
   - **Excel**: Reads multi-sheet `.xlsx` and `.xls` files with sheet switching and validation.
   - **JSON**: Ingests flat arrays or nested record envelopes with automated `json_normalize`.

2. **Automated 5-Stage Data Analysis Pipeline**:
   - **Stage 1 (Ingestion & Profiling)**: Detects column types (numeric, categorical, datetime, ID, text), calculates missing rates, duplicates, and assigns a Data Health Score (0-100%).
   - **Stage 2 (Smart Cleansing)**: Trims column whitespace, eliminates duplicate rows, parses ISO datetimes, standardizes formatted currency/percent strings, and performs intelligent median/mode imputation.
   - **Stage 3 (Statistical Pattern Discovery)**: Computes Pearson correlation matrices, flags IQR outliers, distribution skews, and segment performance.
   - **Stage 4 (Visual Dashboard Generation)**: Generates interactive Plotly figures (time-series trends, distribution histograms with marginal box plots, segment comparisons, scatter relationships, and correlation heatmaps).
   - **Stage 5 (Executive Delivery)**: Formulates strategic takeaways and executive recommendations.

3. **Autonomous Code Writing & Execution Sandbox**:
   - Writes clean Python code using `pandas`, `numpy`, and `plotly`.
   - Executes code in an isolated in-process sandbox with standard output redirection, variable tracking, and figure interception.

4. **RAG-Powered Error Recovery & Self-Healing Loop**:
   - Intercepts Python exceptions (`KeyError`, `SettingWithCopyWarning`, `TypeError`, `ValueError`, `AttributeError`, `ZeroDivisionError`).
   - Retrieves relevant official documentation chunks, antipattern warnings, and verified syntax fixes.
   - Self-healing prompt loop iteratively repairs code, re-runs it, and logs the healing audit trace until successful.

5. **Non-Technical User Interface (Streamlit)**:
   - File drag-and-drop & one-click sample datasets.
   - Conversational Analyst for plain-English natural language Q&A.
   - Live Self-Healing Inspector showing real-time code synthesis, error interception, RAG retrieval, and diffs.
   - 1-Click Export for Standalone HTML Executive Reports, Markdown Summaries, and Cleaned Datasets.

---

## 🏗️ Architecture

```
                          ┌────────────────────────┐
                          │   CSV / Excel / JSON   │
                          └───────────┬────────────┘
                                      │
                                      ▼
                          ┌────────────────────────┐
                          │     DataLoader         │
                          └───────────┬────────────┘
                                      │
                 ┌────────────────────┴──────────────────┐
                 ▼                                       ▼
    ┌────────────────────────┐              ┌────────────────────────┐
    │  DataProfiler (EDA)    │              │ AnalysisPipeline       │
    │  - Health Score (0-100)│              │ - Auto Data Cleansing  │
    │  - Outliers & Skewness │              │ - Automated Visuals    │
    │  - Correlation Matrix  │              │ - Executive Insights   │
    └────────────────────────┘              └───────────┬────────────┘
                                                        │
                                                        ▼
                                            ┌────────────────────────┐
                                            │ Executive HTML Report  │
                                            └────────────────────────┘

    ─── Conversational Self-Healing Loop ─────────────────────────────
    
    User Query ──► AnalystAgent (Synthesizes Code)
                         │
                         ▼
                   CodeSandbox (Executes Python)
                     /       \
           [Success]           [Error / Exception]
              │                        │
              ▼                        ▼
        Render Visuals         RAG Engine (Official Docs)
        & Insights                     │
              ▲                        ▼
              │            Self-Healing Code Prompt
              └────────────────────────┘
```

---

## 🚀 Quick Start

### 1. Prerequisites
Python 3.10+ installed.

### 2. Launch the Application
Double-click `start_app.bat` or run:
```bash
streamlit run app.py
```

### 3. Run Test Suite
```bash
python -m unittest discover tests
```

---

## 📁 Project Structure

- `app.py`: Streamlit interactive web application with tabs, metrics, and chat.
- `backend/data_loader.py`: Ingestion logic for CSV, Excel (multi-sheet), and JSON.
- `backend/data_profiler.py`: Automated statistical profiling, health score, and correlation engine.
- `backend/rag_engine.py`: Fast semantic retrieval engine indexing official docs.
- `backend/rag_knowledge_base.py`: Curated official Python, Pandas, and Plotly documentation chunks.
- `backend/code_sandbox.py`: Isolated Python code execution sandbox.
- `backend/agent.py`: Self-healing autonomous data analyst agent.
- `backend/pipeline.py`: 5-stage automated pipeline runner.
- `backend/report_generator.py`: Standalone HTML and Markdown executive report builder.
- `sample_datasets/`: Ready-to-use sample datasets (Customer Churn, Sales Performance, Web Traffic).
- `tests/`: Unit and integration test suite.
