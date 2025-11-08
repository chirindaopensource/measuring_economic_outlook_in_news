# **`README.md`**

# Replication of "*Measuring economic outlook in the news timely and efficiently*"

<!-- PROJECT SHIELDS -->
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![arXiv](https://img.shields.io/badge/arXiv-2511.04299-b31b1b.svg)](https://arxiv.org/abs/2511.04299)
[![Year](https://img.shields.io/badge/Year-2025-purple)](https://github.com/chirindaopensource/measuring_economic_outlook_in_news)
[![Discipline](https://img.shields.io/badge/Discipline-Computational%20Economics-00529B)](https://github.com/chirindaopensource/measuring_economic_outlook_in_news)
[![Data Source](https://img.shields.io/badge/Data%20Source-Swissdox%40LiRI-003299)](https://www.liri.uzh.ch/en/services/swissdox.html)
[![Core Method](https://img.shields.io/badge/Method-LLM--Based%20Sentiment%20Analysis-orange)](https://github.com/chirindaopensource/measuring_economic_outlook_in_news)
[![Analysis](https://img.shields.io/badge/Analysis-Time%20Series%20Forecasting-red)](https://github.com/chirindaopensource/measuring_economic_outlook_in_news)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Type Checking: mypy](https://img.shields.io/badge/type%20checking-mypy-blue)](http://mypy-lang.org/)
[![Anthropic](https://img.shields.io/badge/Anthropic-Claude%203.5%20Sonnet-D97A53?logo=anthropic&logoColor=white)](https://www.anthropic.com/news/claude-3-5-sonnet)
[![SentenceTransformers](https://img.shields.io/badge/SentenceTransformers-jina--embeddings--v2-2E4053)](https://huggingface.co/jinaai/jina-embeddings-v2-base-de)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=flat&logo=TensorFlow&logoColor=white)](https://www.tensorflow.org/)
[![Scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Statsmodels](https://img.shields.io/badge/Statsmodels-150458-blue)](https://www.statsmodels.org/stable/index.html)
[![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=flat&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=flat&logo=numpy&logoColor=white)](https://numpy.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-%23F37626.svg?style=flat&logo=Jupyter&logoColor=white)](https://jupyter.org/)

**Repository:** `https://github.com/chirindaopensource/measuring_economic_outlook_in_news`

**Owner:** 2025 Craig Chirinda (Open Source Projects)

This repository contains an **independent**, professional-grade Python implementation of the research methodology from the 2025 paper entitled **"Measuring economic outlook in the news timely and efficiently"** by:

*   Elliot Beck
*   Franziska Eckert
*   Linus Kühne
*   Helge Liebert
*   Rina Rosenblatt-Wisch

The project provides a complete, end-to-end computational framework for replicating the paper's findings. It delivers a modular, auditable, and extensible pipeline that executes the entire research workflow: from rigorous data validation and cleansing to large-scale embedding, model training, indicator construction, and the final econometric evaluation.

## Table of Contents

- [Introduction](#introduction)
- [Theoretical Background](#theoretical-background)
- [Features](#features)
- [Methodology Implemented](#methodology-implemented)
- [Core Components (Notebook Structure)](#core-components-notebook-structure)
- [Key Callable: `run_complete_neos_study`](#key-callable-run_complete_neos_study)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Input Data Structure](#input-data-structure)
- [Usage](#usage)
- [Output Structure](#output-structure)
- [Project Structure](#project-structure)
- [Customization](#customization)
- [Contributing](#contributing)
- [Recommended Extensions](#recommended-extensions)
- [License](#license)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)

## Introduction

This project provides a Python implementation of the analytical framework presented in Beck et al. (2025). The core of this repository is the iPython Notebook `measuring_economic_outlook_in_news_draft.ipynb`, which contains a comprehensive suite of functions to replicate the paper's findings. The pipeline is designed as a robust and scalable system for constructing a high-frequency economic sentiment indicator (NEOS) from a large corpus of news articles.

The paper's central contribution is a novel, resource-efficient methodology for sentiment analysis that is suitable for institutions with data privacy constraints. This codebase operationalizes the paper's experimental design, allowing users to:
-   Rigorously validate and manage the entire experimental configuration via a single `config.yaml` file.
-   Execute a multi-stage pipeline to cleanse, prepare, and generate high-dimensional embeddings for a large news corpus.
-   Train a weakly supervised neural network to filter for economics-relevant articles.
-   Programmatically generate a synthetic, high-quality labeled dataset for sentiment analysis using an LLM (Claude 3.5 Sonnet), avoiding the need for manual labeling and preserving data privacy.
-   Train a regularized logistic regression model to score the sentiment of millions of articles efficiently.
-   Construct the final monthly NEOS indicator, its early-release variants, and a traditional lexicon-based baseline.
-   Perform a comprehensive econometric evaluation using a pseudo-out-of-sample (POOS) forecasting exercise to test the indicator's predictive power for GDP growth.
-   Run Diebold-Mariano tests with HAC-robust errors to assess the statistical significance of the findings.
-   Conduct a full suite of robustness checks to test the sensitivity of the results to key methodological choices.

## Theoretical Background

The implemented methods are grounded in principles from natural language processing, machine learning, and time-series econometrics.

**1. LLM-based Synthetic Data Generation:**
The core innovation is the use of a powerful LLM to generate a small, perfectly labeled training set for sentiment classification. This avoids the costly and time-consuming process of manual annotation and, crucially, allows a sentiment model to be trained without exposing any proprietary source data to external APIs.

**2. High-Dimensional Classification with Regularization:**
The sentiment classifier is a logistic regression model trained on high-dimensional embeddings where the number of features ($p=1024$) exceeds the number of samples ($n=256$). To prevent overfitting, L2 (Ridge) regularization is essential. The model minimizes the regularized negative log-likelihood:
$$
\mathcal{L}(\boldsymbol{\beta}) = -\sum_{i=1}^{n} \left[ y_i \log(\sigma(\mathbf{x}_i^T \boldsymbol{\beta})) + (1-y_i) \log(1-\sigma(\mathbf{x}_i^T \boldsymbol{\beta})) \right] + \lambda ||\boldsymbol{\beta}||_2^2
$$

**3. Pseudo-Out-of-Sample (POOS) Forecast Evaluation:**
To rigorously assess the indicator's predictive value, the project implements a POOS forecasting exercise. This involves simulating a real-time forecasting process by iterating through time, using an expanding window of historical data to estimate the forecasting models at each step. This method strictly avoids look-ahead bias. The core forecasting model is:
$$
y_{t+h} = \alpha + \beta y_{t-1} + \gamma x_t^{(m)} + \varepsilon_t \quad \quad (1)
$$
This is compared against a benchmark AR(1) model where $\gamma=0$.

**4. Diebold-Mariano Test with HAC Errors:**
To test if the improvement in forecast accuracy (measured by RMSE) is statistically significant, the Diebold-Mariano (DM) test is used. The implementation is modified to use a Heteroskedasticity and Autocorrelation Consistent (HAC) variance estimator (Newey-West), which is critical for handling the serial correlation present in multi-step-ahead forecast errors ($h>0$).

## Features

The provided iPython Notebook (`measuring_economic_outlook_in_news_draft.ipynb`) implements the full research pipeline, including:

-   **Modular, Multi-Task Architecture:** The entire pipeline is broken down into 23 distinct, modular tasks, each with its own orchestrator function.
-   **Configuration-Driven Design:** All study parameters are managed in an external `config.yaml` file.
-   **Scalable Data Processing:** Includes efficient, batch-based processing for large-scale embedding and inference, with out-of-core storage using HDF5.
-   **Resumable Pipeline:** The main orchestrator implements checkpointing, allowing the pipeline to be stopped and resumed without re-running expensive completed steps.
-   **Robust Model Training:** Implements best practices for both neural network and classical model training, including temporal validation splits, early stopping, and cross-validated hyperparameter tuning.
-   **Rigorous Econometric Analysis:** Implements the full POOS forecasting loop and DM-HAC significance tests with high fidelity.
-   **Complete Replication and Robustness:** A single top-level function call can execute the entire study, including a comprehensive suite of sensitivity analyses.
-   **Full Provenance:** The pipeline generates a detailed log file and a final `reproducibility_manifest.json` that captures all configurations, library versions, and artifact paths for a given run.

## Methodology Implemented

The core analytical steps directly implement the methodology from the paper:

1.  **Validation & Cleansing (Tasks 1-5):** Ingests and validates all raw inputs, cleanses the news corpus according to the paper's scope, and adds temporal features.
2.  **Embedding (Task 6):** Generates 1024-dimensional `jina-embeddings-v2` for the entire corpus.
3.  **Relevance Filtering (Tasks 7-9):** Trains a weakly supervised MLP to identify economics-related articles and filters the corpus.
4.  **Sentiment Model Training (Tasks 10-12):** Generates a synthetic training set with Claude 3.5 Sonnet, embeds it, and trains an L2-regularized logistic regression classifier.
5.  **Indicator Construction (Tasks 13-14):** Scores all relevant articles for sentiment and aggregates the scores into monthly baseline and early-release indicators.
6.  **Econometric Evaluation (Tasks 15-20):** Aligns all indicators to a quarterly frequency, runs the full POOS forecasting exercise, computes RMSE ratios, performs DM-HAC tests, and generates correlation tables.
7.  **Visualization & Robustness (Tasks 21-23):** Generates all charts from the paper and runs a full suite of sensitivity analyses on key methodological choices.

## Core Components (Notebook Structure)

The `measuring_economic_outlook_in_news_draft.ipynb` notebook is structured as a logical pipeline with modular orchestrator functions for each of the 23 major tasks. All functions are self-contained, fully documented with type hints and docstrings, and designed for professional-grade execution.

## Key Callable: `run_complete_neos_study`

The project is designed around a single, top-level user-facing interface function:

-   **`run_complete_neos_study`:** This master orchestrator function, located in the final section of the notebook, runs the entire automated research pipeline from end-to-end, including the baseline analysis and all robustness checks. A single call to this function reproduces the entire computational portion of the project.

## Prerequisites

-   Python 3.9+
-   An Anthropic API key.
-   Sufficient disk space for embeddings (~8GB per million articles).
-   A GPU is highly recommended for the embedding and relevance model training steps.
-   Core dependencies: `pandas`, `numpy`, `pyyaml`, `pyarrow`, `tensorflow`, `scikit-learn`, `statsmodels`, `sentence-transformers`, `h5py`, `joblib`, `anthropic`, `umap-learn`, `matplotlib`, `seaborn`, `tqdm`, `faker`.

## Installation

1.  **Clone the repository:**
    ```sh
    git clone https://github.com/chirindaopensource/measuring_economic_outlook_in_news.git
    cd measuring_economic_outlook_in_news
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install Python dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

4.  **Set your Anthropic API Key:**
    ```sh
    export ANTHROPIC_API_KEY='your-key-here'
    ```

## Input Data Structure

The pipeline requires several input DataFrames with specific schemas, which are rigorously validated. A synthetic data generator is included in the notebook for a self-contained demonstration.
1.  **`raw_news_data_df`**: The large-scale news article corpus.
2.  **`raw_macro_data_df`**: Quarterly macroeconomic data (GDP, etc.).
3.  **`monthly_indicator_data_df`**: Monthly comparator indicators (PMI, KOF).
4.  **`release_calendar_df`**: Metadata on indicator release dates.
5.  **`evaluation_windows_df`**: Metadata on valid evaluation periods for certain indicators.
6.  A translated German sentiment **lexicon file** (CSV).

All other parameters are controlled by the `config.yaml` file.

## Usage

The `measuring_economic_outlook_in_news_draft.ipynb` notebook provides a complete, step-by-step guide. The primary workflow is to execute the final cell of the notebook, which demonstrates how to use the top-level `run_complete_neos_study` orchestrator:

```python
# Final cell of the notebook

# This block serves as the main entry point for the entire project.
if __name__ == '__main__':
    # 1. Load configuration from the YAML file.
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # 2. Define paths.
    ROOT_OUTPUT_DIRECTORY = './neos_study_output'
    LEXICON_PATH = './dummy_lexicon_de.csv'
    
    # 3. Generate a full set of synthetic data files for the demonstration.
    # (The data generation functions are defined earlier in the notebook)
    raw_news_data_df = create_synthetic_news_data(1000, config)
    # ... create all other synthetic DataFrames ...
    
    # 4. Execute the entire replication study.
    final_results = run_complete_neos_study(
        raw_news_data_df=raw_news_data_df,
        raw_macro_data_df=raw_macro_data_df,
        monthly_indicator_data_df=monthly_indicator_data_df,
        release_calendar_df=release_calendar_df,
        evaluation_windows_df=evaluation_windows_df,
        fused_master_input_specification=config,
        root_output_directory=ROOT_OUTPUT_DIRECTORY,
        lexicon_path=LEXICON_PATH
    )
    
    # 5. Inspect final results.
    print("--- Baseline Run Status ---")
    print(final_results['baseline_run_results']['status'])
```

## Output Structure

The pipeline generates a structured output directory:
-   **`output/baseline_run/`**: Contains all artifacts from the main pipeline run.
    -   `data/`: Intermediate data files (embeddings, scores, etc.).
    -   `models/`: Trained model files.
    -   `results/`: Final result tables (CSV) and charts (PNG).
    -   `pipeline_run.log`: A detailed log file for the run.
    -   `reproducibility_manifest.json`: A complete record of the run.
-   **`output/robustness_checks/`**: Contains a subdirectory for each sensitivity analysis, with the same internal structure as `baseline_run`.
-   **`output/robustness_checks/robustness_summary_results.csv`**: A master table comparing key results across all robustness checks.

## Project Structure

```
measuring_economic_outlook_in_news/
│
├── measuring_economic_outlook_in_news_draft.ipynb
├── config.yaml
├── requirements.txt
│
├── neos_study_output/
│   ├── baseline_run/
│   │   ├── data/
│   │   ├── models/
│   │   └── results/
│   └── robustness_checks/
│       ├── sensitivity_tau_0.4/
│       └── ...
│
├── LICENSE
└── README.md
```

## Customization

The pipeline is highly customizable via the `config.yaml` file. Users can modify all study parameters, including model names, API settings, filtering thresholds, and file paths, without altering the core Python code.

## Contributing

Contributions are welcome. Please fork the repository, create a feature branch, and submit a pull request with a clear description of your changes. Adherence to PEP 8, type hinting, and comprehensive docstrings is required.

## Recommended Extensions

Future extensions could include:
-   **Alternative Embedding Models:** The modular design allows for easy substitution of the `model_name` in the config to test other embedding models.
-   **Different Classifiers:** The sentiment model could be replaced with other classifiers (e.g., SVM, Gradient Boosting) to test for performance differences.
-   **Advanced Econometric Models:** The forecasting exercise could be extended to include more complex models, such as VARs or models with dynamic variable selection.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Citation

If you use this code or the methodology in your research, please cite the original paper:

```bibtex
@article{beck2025measuring,
  title={Measuring economic outlook in the news timely and efficiently},
  author={Beck, Elliot and Eckert, Franziska and K{\"u}hne, Linus and Liebert, Helge and Rosenblatt-Wisch, Rina},
  journal={arXiv preprint arXiv:2511.04299},
  year={2025}
}
```

For the implementation itself, you may cite this repository:
```
Chirinda, C. (2025). A Production-Grade Replication of "Measuring economic outlook in the news timely and efficiently".
GitHub repository: https://github.com/chirindaopensource/measuring_economic_outlook_in_news
```

## Acknowledgments

-   Credit to **Elliot Beck, Franziska Eckert, Linus Kühne, Helge Liebert, and Rina Rosenblatt-Wisch** for the foundational research that forms the entire basis for this computational replication.
-   This project is built upon the exceptional tools provided by the open-source community. Sincere thanks to the developers of the scientific Python ecosystem, including **Pandas, NumPy, TensorFlow, Scikit-learn, Statsmodels, Sentence-Transformers, and Anthropic**.

--

*This README was generated based on the structure and content of the `measuring_economic_outlook_in_news_draft.ipynb` notebook and follows best practices for research software documentation.*
