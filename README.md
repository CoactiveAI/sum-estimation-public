# SumEstimation

**SumEstimation** is a framework for estimating the sum of scores across large-scale embedding datasets using a variety of sampling strategies. It is designed to support both approximate and hybrid estimation techniques across multiple similarity functions (KDE, Softmax) and dataset types (image or text embeddings).

## Motivation

Computing exact sums over large embedding datasets (millions of vectors) is computationally expensive. This repo explores sampling-based estimators that can reliably approximate the total sum with fewer computations, enabling scalable deployment in ranking, evaluation, and retrieval workflows.

## Supported Methods

- **TopK**: Uses nearest neighbors by similarity.
- **Random**: Uniform random sampling of dataset points.
- **OurAlgorithm**: An adaptive sampler that selects a budgeted number of items per query.
- **Combined**: Combines TopK and Random sampling (hybrid).

Each method supports per-query evaluation and produces score estimates under a variety of hyperparameter and estimator configurations.

---

## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/your-org/sum-estimation-ext.git
cd sum-estimation-ext
pip install -r requirements.txt
```

---

## Citation
If you use this repository in your work, please cite the paper or contact us via the repository.

## Contact
For questions, feedback, or contributions:

Steve Mussmann
steve@coactive.ai

Mehul Smriti Raje
mehul@coactive.ai


