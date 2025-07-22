import pandas as pd
import numpy as np
from config import settings
from matplotlib import pyplot as plt

experiment_results_path = settings.EXPERIMENT_RESULTS_PATH
plot_format = "pdf"


# TODO: make this into a function
alg_to_color = {"our": "b",
                   "topk": "r",
                   "random": "g",
                   "combined": "0.25"}
alg_to_legend_name = {"our": "Our Algorithm",
                        "topk": "TopK",
                        "random": "Random",
                        "combined": "Combined"}
task_to_param_name = {"kde": "Bandwidth",
                      "softmax": "Temperature",
                      "ball_counting": "Radius"}
task_to_task_name = {"kde": "KDE",
                     "softmax": "Softmax",
                     "ball_counting": "Ball Counting"}
data_task_to_ylim_upper = {"image_kde": 0.2,
                     "image_ball_counting": 0.2,
                     "image_softmax": 0.2,
                     "text_kde": 0.2,
                     "text_ball_counting": 0.2,
                     }
data_task_to_time_upper = {"image_kde": None,
                     "image_ball_counting": 6.0,
                     "image_softmax": 3.0,
                     "text_kde": 2.0,
                     "text_ball_counting": 2.0,
                     }
data_task_to_param_lower= {"image_kde": None,
                     "image_ball_counting": 5.0,
                     "image_softmax": None,
                     "text_kde": None,
                     "text_ball_counting": 1,
                     }
data_task_to_param_upper = {"image_kde": None,
                     "image_ball_counting": 30,
                     "image_softmax": None,
                     "text_kde": None,
                     "text_ball_counting": 10,
                     }


for combination in settings.TASK_DATA_COMBINATIONS:
    task = combination['task']
    data = combination['data']

    qid_to_true_sum = {}
    df = pd.read_parquet(f"{experiment_results_path}/{data}_{task}_true_sum.parquet")
    task_param_values = np.array([float(value) for value in df.columns[2:]])

    for row in df.itertuples():
        qid = row[2] # 0th column is index
        values = np.array(row[3:])
        assert(len(values) == len(task_param_values))
        qid_to_true_sum[qid] = values


    method_to_qid_to_sum_estimates = {} 
    df = pd.read_parquet(f"{experiment_results_path}/{data}_{task}_sum_estimates.parquet")
    assert((task_param_values == np.array([float(value) for value in df.columns[2:]])).all())
    methods = list(df["method"].unique())
    for method in methods:
        method_to_qid_to_sum_estimates[method] = {}

    for row in df.itertuples():
        method = row[1] # 0th column is index
        qid = row[2]
        values = np.array(row[3:])
        assert(len(values) == len(task_param_values))
        method_to_qid_to_sum_estimates[method][qid] = values
    print(len(method_to_qid_to_sum_estimates.keys()))


    method_to_rel_err_matrix = {}
    for method in method_to_qid_to_sum_estimates.keys():
        rows = []
        qid_to_sum_estimates = method_to_qid_to_sum_estimates[method]
        for qid in qid_to_sum_estimates.keys():
            denom = np.where(qid_to_true_sum[qid] > 0, qid_to_true_sum[qid], -1)
            rows.append(np.where(qid_to_true_sum[qid] > 0,np.divide(np.abs(qid_to_sum_estimates[qid] - qid_to_true_sum[qid]),denom),np.nan))

        method_to_rel_err_matrix[method] = np.array(rows)
    print(len(method_to_rel_err_matrix.keys()))


    method_to_times = {}
    df = pd.read_parquet(f"{experiment_results_path}/{data}_{task}_time_estimates.parquet")
    assert(set(df["method"].unique()) == set(methods))
    for method in methods:
        method_to_times[method] = []

    for row in df.itertuples():
        method = row[1] # 0th column is index
        qid = row[2]
        time = row[3]
        method_to_times[method].append(time)

    #########################################
    # plot quality plot
    plt.figure()
    algs = []
    for method in sorted(method_to_rel_err_matrix.keys()): #hack to make combined come first
        print(method)
        alg = method[:method.index("_")]
        if alg not in algs:
            legend_name = alg_to_legend_name[alg]
            algs.append(alg)
        else:
            legend_name = None

        y = np.nanmedian(method_to_rel_err_matrix[method],axis=0)
        y[np.mean(np.isnan(method_to_rel_err_matrix[method]),axis=0) > 0.75] = np.nan
        plt.plot(task_param_values, 
                y, 
                color=alg_to_color[alg],
                label=legend_name,
                alpha=0.8,
                linewidth=1.5
                )

    plt.xlabel(task_to_param_name[task])
    plt.ylabel('Median Relative Error')
    plt.xscale('log')
    plt.xlim(data_task_to_param_lower[f"{data}_{task}"], data_task_to_param_upper[f"{data}_{task}"])
    plt.ylim(-0.01, data_task_to_ylim_upper[f"{data}_{task}"])
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(f"plots/{data}_{task}_quality.{plot_format}", format=plot_format, bbox_inches="tight")

    #########################################
    # plot trade-off
    plt.figure()
    alg_to_plot_tuples = {}

    for method in sorted(method_to_rel_err_matrix.keys()): #hack to make combined come first
        alg = method[:method.index("_")]
        if alg not in alg_to_plot_tuples:
            alg_to_plot_tuples[alg] = []

        x_center = np.median(method_to_times[method])
        assert(len(method_to_times[method])==100)
        x_lower = x_center - np.sort(method_to_times[method])[40] #Pr(Binomial(100,0.5) <= 40) = 2.844%
        x_upper = np.sort(method_to_times[method])[60] - x_center #Pr(Binomial(100,0.5) >= 61) = 1.76%

        rel_err_matrix = np.nan_to_num(method_to_rel_err_matrix[method])
        chosen_task_param_idx = np.argmax(np.median(rel_err_matrix,axis=0))
        rel_errs = rel_err_matrix[:,chosen_task_param_idx]
        y_center = np.median(rel_errs)
        assert(len(rel_errs)==100)
        y_lower = y_center - np.sort(rel_errs)[40] #Pr(Binomial(100,0.5) <= 40) = 2.844%
        y_upper = np.sort(rel_errs)[60] - y_center #Pr(Binomial(100,0.5) >= 61) = 1.76%

        annotation = method[method.index("_")+1:]

        alg_to_plot_tuples[alg].append( (x_center, x_lower, x_upper, y_center, y_lower, y_upper, annotation) )

    for alg, plot_tuples in sorted(list(alg_to_plot_tuples.items())): #hack to make combined plotted first
        plt.errorbar(
            np.array([plot_tuple[0] for plot_tuple in plot_tuples]), 
            np.array([plot_tuple[3] for plot_tuple in plot_tuples]), 
            xerr=np.array([[plot_tuple[1] for plot_tuple in plot_tuples],[plot_tuple[2] for plot_tuple in plot_tuples]]),
            yerr=np.array([[plot_tuple[4] for plot_tuple in plot_tuples],[plot_tuple[5] for plot_tuple in plot_tuples]]),
            fmt='o', 
            label=alg_to_legend_name[alg], 
            capsize=4, 
            color=alg_to_color[alg]
        )

        for plot_tuple in plot_tuples:
            plt.annotate(plot_tuple[6], (plot_tuple[0], plot_tuple[3]), fontsize=10, color=alg_to_color[alg])

    plt.xlabel("Median Inference Time")
    plt.ylabel(f"Median Relative Error at Worst {task_to_param_name[task]}")
    plt.xlim(0, data_task_to_time_upper[f"{data}_{task}"])
    plt.ylim(-0.01, data_task_to_ylim_upper[f"{data}_{task}"])
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(f"plots/{data}_{task}_tradeoff.{plot_format}", format=plot_format, bbox_inches="tight")