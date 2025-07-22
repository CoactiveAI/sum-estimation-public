import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed

from qdrant_data_classes import EmbeddingObject
from config import settings


class Dataset:
    def __init__(self, spark, sc):
        self._spark = spark
        self._sc = sc
        self.query_embedding_objects = self._get_query_embedding_objects(spark)
        self.dataset_embedding_objects = self._get_dataset_embedding_objects(spark, sc)
        self.setting_params = self._get_setting_params()
        self.true_estimates = self._get_true_estimates()
        self.exact_recall = self._get_exact_recall()

    def _get_query_embedding_objects(self) -> list[EmbeddingObject]:
        pass

    def _get_dataset_embedding_objects(self) -> list[EmbeddingObject]:
        pass

    def _get_setting_params(self) -> list[float]:
        return self.setting_params

    def _get_true_estimates(self) -> pd.DataFrame:
        return pd.read_parquet(self.true_estimates_file)

    def _get_exact_recall(self) -> pd.DataFrame:
        return pd.read_parquet(self.exact_recall_file)
    
    def copy(self) -> Dataset:
        # Create a new instance without triggering __init__
        new_obj = self.__class__.__new__(self.__class__)

        # Copy core fields except heavy embedding lists
        attrs_to_copy = [
            "collection_name",
            "setting_params",
            "true_estimates_file",
            "exact_recall_file",
            "_spark",
            "_sc"
        ]

        for attr in attrs_to_copy:
            if hasattr(self, attr):
                setattr(new_obj, attr, getattr(self, attr))

        # Lazily load light-weight data
        new_obj.true_estimates = self._get_true_estimates()
        new_obj.exact_recall = self._get_exact_recall()

        # Skip heavy data
        new_obj.query_embedding_objects = None
        new_obj.dataset_embedding_objects = None

        return new_obj    


class Dataset_Image_KDE(Dataset):
    def __init__(self, spark, sc):
        self.collection_name = settings.COLLECTION_NAME["open-images_resnet-50"]
        self.setting_params = [10 ** p for p in np.arange(-0.25, 1.75, 0.05)]
        self.true_estimates_file = "s3://coactive-ml-rnd/SumEstimationExperiments/apr21/kde_Z_vals/full.parquet"
        self.exact_recall_file = "s3://coactive-ml-rnd/SumEstimationExperiments/apr21/kde_recall/topk_top5000.parquet"
        super().__init__(spark, sc)

    def _get_query_embedding_objects(self, spark) -> list[EmbeddingObject]:
        query_rows = spark.read.format("delta").option("header", "true").load("s3://coactive-ml-rnd/SumEstimationExperiments/apr21/kde_query_rows/").toPandas()
        query_embedding_objects = [EmbeddingObject(image_id=id, embedding=emb) for id, emb in zip(query_rows['CoactiveImageID'], query_rows['embedding'])]
        return query_embedding_objects

    def _get_dataset_embedding_objects(self, spark, sc) -> list[EmbeddingObject]:
        encoder = 'resnet-50'
        segment = 'train'

        dataset_embedding_objects = []
        for idx in tqdm(range(200)):
            embeddings_path = f"s3://coactive-ml-rnd/SumEstimationExperiments/OI_train_{encoder}_abs/{idx}"
            spark_df = spark.read.format("delta").option("header", "true").load(embeddings_path)
            spark_df = spark_df.repartition(sc.defaultParallelism).cache()
            rows = spark_df.select("CoactiveImageID").collect()
            dataset_embedding_objects.extend([EmbeddingObject(image_id=row.CoactiveImageID) for row in rows])
        return dataset_embedding_objects


class Dataset_Image_Softmax(Dataset):
    def __init__(self, spark, sc):
        self.collection_name = settings.COLLECTION_NAME["open-images_clip_vit_l14_336"]
        self.setting_params = [10 ** p for p in np.arange(-3.0, 1.0, 0.1)]
        self.true_estimates_file = "s3://coactive-ml-rnd/SumEstimationExperiments/apr21/softmax_Z_vals/full.parquet"
        self.exact_recall_file = "s3://coactive-ml-rnd/SumEstimationExperiments/apr21/softmax_recall/topk_top5000.parquet"
        super().__init__(spark, sc)

    def _get_query_embedding_objects(self, spark) -> list[EmbeddingObject]:
        query_rows = spark.read.format("delta").option("header", "true").load("s3://coactive-ml-rnd/SumEstimationExperiments/apr21/softmax_query_rows/")
        query_rows = query_rows.select('query_id', 'embedding').toPandas()
        query_embedding_objects = [EmbeddingObject(image_id=id, embedding=embedding) for id, embedding in zip(query_rows['query_id'], query_rows['embedding'])]
        return query_embedding_objects

    def _get_dataset_embedding_objects(self, spark, sc) -> list[EmbeddingObject]:
        encoder = 'clip_vit_normalised'
        segment = 'train'

        dataset_embedding_objects = []
        for idx in tqdm(range(200)):
            embeddings_path = f"s3://coactive-ml-rnd/SumEstimationExperiments/OI_train_{encoder}/{idx}"
            spark_df = spark.read.format("delta").option("header", "true").load(embeddings_path)
            spark_df = spark_df.repartition(sc.defaultParallelism).cache()
            rows = spark_df.select("CoactiveImageID").collect()
            dataset_embedding_objects.extend([EmbeddingObject(image_id=row.CoactiveImageID) for row in rows])
        return dataset_embedding_objects       


class Dataset_Image_BallCounting(Dataset):
    def __init__(self, spark, sc):
        self.collection_name = settings.COLLECTION_NAME["open-images_resnet-50"]
        self.setting_params = sorted(set(
            [10 ** p for p in np.arange(-3.0, 2.0, 0.1)] +
            [10 ** p for p in np.arange(0.5, 1.8, 0.05)]
        ))
        self.true_estimates_file = "s3://coactive-ml-rnd/SumEstimationExperiments/apr21/synthetic_Z_vals/full.parquet"
        self.exact_recall_file = "s3://coactive-ml-rnd/SumEstimationExperiments/apr21/kde_recall/topk_top5000.parquet"
        super().__init__(spark, sc)

    def _get_query_embedding_objects(self, spark) -> list[EmbeddingObject]:
        query_rows = spark.read.format("delta").option("header", "true").load("s3://coactive-ml-rnd/SumEstimationExperiments/apr21/kde_query_rows/").toPandas()
        query_embedding_objects = [EmbeddingObject(image_id=id, embedding=emb) for id, emb in zip(query_rows['CoactiveImageID'], query_rows['embedding'])]
        return query_embedding_objects

    def _get_dataset_embedding_objects(self, spark, sc) -> list[EmbeddingObject]:
        encoder = 'resnet-50'
        segment = 'train'

        dataset_embedding_objects = []
        for idx in tqdm(range(200)):
            embeddings_path = f"s3://coactive-ml-rnd/SumEstimationExperiments/OI_train_{encoder}_abs/{idx}"
            spark_df = spark.read.format("delta").option("header", "true").load(embeddings_path)
            spark_df = spark_df.repartition(sc.defaultParallelism).cache()
            rows = spark_df.select("CoactiveImageID").collect()
            dataset_embedding_objects.extend([EmbeddingObject(image_id=row.CoactiveImageID) for row in rows])
        return dataset_embedding_objects   


class Dataset_Text_KDE(Dataset):
    def __init__(self, spark, sc):
        self.collection_name = settings.COLLECTION_NAME["amazon-reviews_distilbert"]
        self.setting_params = [10 ** p for p in np.arange(-0.70,1.50,0.05)]
        self.true_estimates_file = "s3://coactive-ml-rnd/SumEstimationExperiments/may8_text/kde_Z_vals/full.parquet"
        self.exact_recall_file = "s3://coactive-ml-rnd/SumEstimationExperiments/may8_text/kde_recall/topk_top5000.parquet"
        super().__init__(spark, sc)

    def _get_query_embedding_objects(self, spark) -> list[EmbeddingObject]:
        query_rows = pd.read_parquet("s3://coactive-ml-rnd/SumEstimationExperiments/may8_text/kde_query_rows.parquet")
        query_embedding_objects = [EmbeddingObject(image_id=id, embedding=embedding.astype(float)) for id, embedding in zip(query_rows['coactive_image_id'], query_rows['embedding'])]
        return query_embedding_objects

    def _get_dataset_embedding_objects(self, spark, sc) -> list[EmbeddingObject]:

        def load_and_create_objects(file_lo) -> list[EmbeddingObject]:
            try:
                emb_df = pd.read_parquet(f"s3://coactive-ml-rnd/SumEstimationExperiments/may8_text/bert_embeddings/{file_lo}.parquet", columns=["coactive_image_id"])
                return [EmbeddingObject(image_id=id) for id in emb_df['coactive_image_id']]
            except Exception as e:
                print(f"Failed at {file_lo}: {e}")
                return []

        file_offsets = list(range(0, 10000000, 5000))
        results = Parallel(n_jobs=-1)(
            delayed(load_and_create_objects)(file_lo) for file_lo in tqdm(file_offsets)
        )
        dataset_embedding_objects = [obj for sublist in results for obj in sublist]
        return dataset_embedding_objects
    

class Dataset_Text_BallCounting(Dataset):
    def __init__(self, spark, sc):
        self.collection_name = settings.COLLECTION_NAME["amazon-reviews_distilbert"]
        self.setting_params = sorted(set(
            [10 ** p for p in np.arange(-5.0, 1.0, 0.1)] +
            [10 ** p for p in np.arange(0, 1.8, 0.05)]
        ))
        self.true_estimates_file = "s3://coactive-ml-rnd/SumEstimationExperiments/may8_text/synthetic_Z_vals/full.parquet"
        self.exact_recall_file = "s3://coactive-ml-rnd/SumEstimationExperiments/may8_text/kde_recall/topk_top5000.parquet"
        super().__init__(spark, sc)

    def _get_query_embedding_objects(self, spark) -> list[EmbeddingObject]:
        query_rows = pd.read_parquet("s3://coactive-ml-rnd/SumEstimationExperiments/may8_text/kde_query_rows.parquet")
        query_embedding_objects = [EmbeddingObject(image_id=id, embedding=embedding.astype(float)) for id, embedding in zip(query_rows['coactive_image_id'], query_rows['embedding'])]
        return query_embedding_objects

    def _get_dataset_embedding_objects(self, spark, sc) -> list[EmbeddingObject]:

        def load_and_create_objects(file_lo) -> list[EmbeddingObject]:
            try:
                emb_df = pd.read_parquet(f"s3://coactive-ml-rnd/SumEstimationExperiments/may8_text/bert_embeddings/{file_lo}.parquet", columns=["coactive_image_id"])
                return [EmbeddingObject(image_id=id) for id in emb_df['coactive_image_id']]
            except Exception as e:
                print(f"Failed at {file_lo}: {e}")
                return []

        file_offsets = list(range(0, 10000000, 5000))
        results = Parallel(n_jobs=-1)(
            delayed(load_and_create_objects)(file_lo) for file_lo in tqdm(file_offsets)
        )
        dataset_embedding_objects = [obj for sublist in results for obj in sublist]
        return dataset_embedding_objects
