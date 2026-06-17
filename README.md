# FinLLM Financial NLP Tasks: Classification & Summarization

## Overview

This repository contains datasets and baseline experiments for two Financial NLP tasks from the **FinLLM Challenge (IJCAI 2024)**:

1. **Task 1: Financial Classification (Claim vs Premise)**
2. **Task 2: Financial Text Summarization**

These tasks evaluate the capability of Large Language Models (LLMs) in understanding, reasoning, and generating financial text.

---

# Task 1: Financial Classification

## Problem Statement

The objective is to classify a financial sentence from earnings conference calls into one of two categories:

* **Premise** → Supporting evidence, explanation, reasoning, or factual statement.
* **Claim** → Opinion, conclusion, prediction, or viewpoint.

### Example

**Input**

```text
Revenue increased by 25% compared to last year.
```

**Output**

```text
premise
```

**Input**

```text
The opportunity for our shareholders has never been better.
```

**Output**

```text
claim
```

---

## Dataset

**Dataset Name**

```text
TheFinAI/finarg-ecc-auc_train
```

### Dataset Structure

| Column  | Description                        |
| ------- | ---------------------------------- |
| id      | Unique sample identifier           |
| query   | Instruction prompt                 |
| text    | Financial sentence                 |
| answer  | Ground truth label                 |
| choices | Available labels                   |
| gold    | Numeric label (0=Premise, 1=Claim) |

---

## Task Type

```text
Binary Text Classification
```

### Input

```text
Financial sentence
```

### Output

```text
premise
or
claim
```

---

## Evaluation Metrics

* F1 Score (Primary Metric)
* Accuracy

---

## Applications

* Earnings Call Analysis
* Financial Argument Mining
* Investment Research Automation
* Financial Intelligence Systems
* Financial RAG Pipelines

---

# Task 2: Financial Text Summarization

## Problem Statement

The objective is to generate a concise and informative summary from a financial news article while preserving key information.

### Example

**Input**

```text
Apple reported quarterly revenue of $95 billion, exceeding analyst expectations and driven by strong iPhone sales.
```

**Output**

```text
Apple surpassed expectations with strong quarterly revenue growth fueled by iPhone sales.
```

---

## Dataset

**Dataset Name**

```text
TheFinAI/edtsum_train
```

---

## Task Type

```text
Abstractive Text Summarization
```

### Input

```text
Financial news article
```

### Output

```text
Short financial summary
```

---

## Evaluation Metrics

* ROUGE-1 (Primary Metric)
* ROUGE-2
* ROUGE-L
* BERTScore

---

## Applications

* Financial News Summarization
* Market Intelligence
* Research Report Generation
* Investor Briefings
* Financial Content Automation

---

# Project Structure

```text
.
├──
│   ├── finarg_ecc_auc_train.csv
│   └── edtsum_train.csv
│
├
│
├──
├── 
├── README.md
└── 
```

---

# Technologies

* Python
* Pandas
* Hugging Face Datasets
* Transformers
* PyTorch
* Scikit-Learn
* FinBERT
* BERT / RoBERTa / DeBERTa

---

# Objectives

### Task 1

Identify whether a financial statement is:

* Claim
* Premise

### Task 2

Generate concise and accurate summaries of financial news articles.

---

# Future Work

* FinBERT Fine-Tuning
* LoRA / PEFT Training
* Financial RAG Integration
* Multi-task Learning
* Financial LLM Benchmarking

---

## Author


