class Settings:

    # TODO: change to a public url
    EXPERIMENT_RESULTS_PATH = "s3://coactive-ml-rnd/SumEstimationExperiments/github_results_july22_run3"
    TASK_DATA_COMBINATIONS = [
        {'task': 'kde', 'data': 'image'},
        {'task': 'softmax', 'data': 'image'},
        {'task': 'ball_counting', 'data': 'image'},
        {'task': 'kde', 'data': 'text'},
        {'task': 'ball_counting', 'data': 'text'},
    ]


settings = Settings()