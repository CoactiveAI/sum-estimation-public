"""Figure styling: the colours, names and axis limits the paper's figures use."""

ALG_TO_COLOR = {
    "our": "b",
    "topk": "r",
    "random": "g",
    "combined": "0.25",
}

ALG_TO_LEGEND_NAME = {
    "our": "Our Algorithm",
    "topk": "TopK",
    "random": "Random",
    "combined": "Combined",
}

TASK_TO_PARAM_NAME = {
    "kde": "Bandwidth",
    "softmax": "Temperature",
    "ball_counting": "Radius",
}

TASK_TO_TASK_NAME = {
    "kde": "KDE",
    "softmax": "Softmax",
    "ball_counting": "Ball Counting",
}

# Axis limits per (data, task). None means "let matplotlib decide".
YLIM_UPPER = {
    "image_kde": 0.2,
    "image_ball_counting": 0.2,
    "image_softmax": 0.2,
    "text_kde": 0.2,
    "text_ball_counting": 0.2,
}

TIME_UPPER = {
    "image_kde": None,
    "image_ball_counting": 6.0,
    "image_softmax": 3.0,
    "text_kde": 2.0,
    "text_ball_counting": 2.0,
}

PARAM_LOWER = {
    "image_kde": None,
    "image_ball_counting": 5.0,
    "image_softmax": None,
    "text_kde": None,
    "text_ball_counting": 1,
}

PARAM_UPPER = {
    "image_kde": None,
    "image_ball_counting": 30,
    "image_softmax": None,
    "text_kde": None,
    "text_ball_counting": 10,
}


def algorithm_of(method: str) -> str:
    """'our_25' -> 'our'; the method name carries its parameters after the first _."""
    return method[:method.index("_")] if "_" in method else method


def params_of(method: str) -> str:
    """'combined_100_200' -> '100_200', used to annotate the trade-off points."""
    return method[method.index("_") + 1:] if "_" in method else ""
