import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from config import settings

experiment_results_path = settings.EXPERIMENT_RESULTS_PATH
plot_format = "png"


for combination in settings.TASK_DATA_COMBINATIONS:
    task = combination['task']
    data = combination['data']

    qidlevel_to_true_topk = {}
    qidlevel_maxk = {}
    df = pd.read_parquet(f"{experiment_results_path}/{data}_{task}_exact_recall.parquet")
    print(df.columns)
    for row in df.itertuples():
        qid = row[1] # 0th column is index
        
        k = row[2]
        level = int(row[3])
        truelist = row[4]

        qidlevel = f"{qid}_at_{level}"
        qidlevel_to_true_topk[qidlevel] = truelist
        qidlevel_maxk[qidlevel] = k


    klevel_to_recall = {} 
    df = pd.read_parquet(f"{experiment_results_path}/{data}_{task}_qdrant_recall.parquet")
    print(df.columns)
    for row in df.itertuples():
        qid = row[1] # 0th column is index
        k = row[2]
        level = int(row[3])
        if level >= 0 and data=="image" and (task == "softmax" or task == "kde"):
            level += 1 #mismatch in qdrant vs exact indexing
        approxlist = row[4]
        qidlevel = f"{qid}_at_{level}"
        assert(k <= qidlevel_maxk[qidlevel])
        truelist = qidlevel_to_true_topk[qidlevel][:k]

        if len(truelist) == k: 
            recall = len(set(approxlist).intersection(truelist)) / len(truelist)
        else:
            print(len(approxlist),len(truelist))
            # assert(len(approxlist) == len(truelist))
            recall = np.nan
        
        klevel = (k,level)
        if klevel not in klevel_to_recall:
            klevel_to_recall[klevel] = []
        klevel_to_recall[klevel].append(recall)

    k_to_mean_series = {}
    k_to_std_series = {}
    k_to_count_series = {}
    for k in [25,50,100,200]:
        k_to_mean_series[k] = np.full((40,), np.nan)
        k_to_std_series[k] = np.full((40,), np.nan)
        k_to_count_series[k] = np.full((40,), np.nan)

    for k, level in klevel_to_recall.keys():
        recalls = klevel_to_recall[(k,level)]
        if k <= 200:
            if not np.isnan(recalls).all():
                k_to_mean_series[k][level] = np.nanmean(recalls)
                k_to_std_series[k][level] = np.nanstd(recalls)
                k_to_count_series[k][level] = np.sum(np.logical_not(np.isnan(recalls)))
        else:
            assert(level==-1)
            recalls = recalls[:100]
            print(k,round(np.mean(recalls),4), round(2*np.std(recalls)/np.sqrt(len(recalls)),4))

    plt.figure()
    for k in [25,50,100,200]:
        plt.plot(np.arange(40),k_to_mean_series[k],"o-",label=f"k={k}")
        plt.fill_between(np.arange(40), 
                        k_to_mean_series[k]-2*k_to_std_series[k]/np.sqrt(k_to_count_series[k]), 
                        k_to_mean_series[k]+2*k_to_std_series[k]/np.sqrt(k_to_count_series[k]), alpha=0.3)
    plt.legend()
    plt.xlabel("Level")
    plt.ylabel("Recall")
    plt.xticks(ticks=np.arange(1,19), labels=[str(i) for i in np.arange(1,19)])
    plt.savefig(f"plots/{data}_{task}_recalls.{plot_format}", format=plot_format, bbox_inches="tight")
    