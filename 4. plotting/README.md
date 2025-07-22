# 📊 4.plotting

This directory contains scripts to **combine result files** and **generate visualizations** for sum estimation experiments.

---

## ✅ Step 1: Combine Results

Run the following script to merge raw result `.parquet` files into one DataFrame per `sum_problem_setting`:

```bash
python 4.plotting/combine_dfs.py


python 4.plotting/plot_recalls.py       # Plots Qdrant and Exact recall metrics
python 4.plotting/plot_results.py       # Plots estimation curves vs true sums
python 4.plotting/plot_synthetic.py     # Plots on synthetic settings (if applicable)
