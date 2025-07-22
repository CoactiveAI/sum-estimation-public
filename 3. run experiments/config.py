class Settings:

    QDRANT_HOST = "http://10.0.142.100:6333"
    QDRANT_API_KEY = 'bVkKP7rBAV73if3rQfo7IdJUfNt6ixAxR7UrSKxNdGQBpdjuJX'
    QDRANT_PORT = 443
    QDRANT_TIMEOUT = 10000

    COLLECTION_NAME = {
        "open-images_resnet-50": "SEPub_OI_train8M_resnet_50",
        "open-images_clip_vit_l14_336": "SEPub_OI_train8M_clip_vit_normalised",
        "amazon-reviews_distilbert": "SEPub_AR_10M_bert"
    }

    RESULTS_PATH = "s3://coactive-ml-rnd/SumEstimationExperiments/github_results_july22_run3"


settings = Settings()