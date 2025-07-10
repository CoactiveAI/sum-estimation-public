import pandas as pd
from pyspark.sql.types import FloatType, ArrayType
from sklearn.preprocessing import normalize
import numpy as np
import pandas as pd
from tqdm import tqdm

from time import time
import torch
from joblib import Parallel, delayed

from qdrant_data_classes import EmbeddingObject

def get_query_dataset_kde_image(spark, sc):
    query_rows = spark.read.format("delta").option("header", "true").load("s3://coactive-ml-rnd/SumEstimationExperiments/apr21/kde_query_rows/").toPandas()
    QUERY_EMBEDDING_OBJECTS = [EmbeddingObject(image_id=id, embedding=emb) for id, emb in zip(query_rows['CoactiveImageID'], query_rows['embedding'])]
    print(len(QUERY_EMBEDDING_OBJECTS))

    encoder = 'resnet-50'
    segment = 'train'

    DATASET_EMBEDDING_OBJECTS = []
    for idx in tqdm(range(2)):
        embeddings_path = f"s3://coactive-ml-rnd/SumEstimationExperiments/OI_train_{encoder}_abs/{idx}"
        spark_df = spark.read.format("delta").option("header", "true").load(embeddings_path)
        spark_df = spark_df.repartition(sc.defaultParallelism).cache()
        rows = spark_df.select("CoactiveImageID").collect()
        DATASET_EMBEDDING_OBJECTS.extend([EmbeddingObject(image_id=row.CoactiveImageID) for row in rows])
    print(len(DATASET_EMBEDDING_OBJECTS))

    return QUERY_EMBEDDING_OBJECTS, DATASET_EMBEDDING_OBJECTS


def get_query_dataset_softmax_image(spark, sc):
    query_rows = spark.read.format("delta").option("header", "true").load("s3://coactive-ml-rnd/SumEstimationExperiments/apr21/softmax_query_rows/")
    query_rows = query_rows.select('query_id', 'embedding').toPandas()
    QUERY_EMBEDDING_OBJECTS = [EmbeddingObject(image_id=id, embedding=embedding) for id, embedding in zip(query_rows['query_id'], query_rows['embedding'])]
    print(len(QUERY_EMBEDDING_OBJECTS))

    encoder = 'clip_vit_normalised'
    segment = 'train'

    DATASET_EMBEDDING_OBJECTS = []
    for idx in tqdm(range(2)):
        embeddings_path = f"s3://coactive-ml-rnd/SumEstimationExperiments/OI_train_{encoder}/{idx}"
        spark_df = spark.read.format("delta").option("header", "true").load(embeddings_path)
        spark_df = spark_df.repartition(sc.defaultParallelism).cache()
        rows = spark_df.select("CoactiveImageID").collect()
        DATASET_EMBEDDING_OBJECTS.extend([EmbeddingObject(image_id=row.CoactiveImageID) for row in rows])
    print(len(DATASET_EMBEDDING_OBJECTS))

    return QUERY_EMBEDDING_OBJECTS, DATASET_EMBEDDING_OBJECTS


def load_and_create_objects(file_lo):
    try:
        emb_df = pd.read_parquet(f"s3://coactive-ml-rnd/SumEstimationExperiments/may8_text/bert_embeddings/{file_lo}.parquet", columns=["coactive_image_id"])
        return [EmbeddingObject(image_id=id) for id in emb_df['coactive_image_id']]
    except Exception as e:
        print(f"Failed at {file_lo}: {e}")
        return []

def get_query_dataset_kde_text(spark, sc):
    query_rows = pd.read_parquet("s3://coactive-ml-rnd/SumEstimationExperiments/may8_text/kde_query_rows.parquet")
    QUERY_EMBEDDING_OBJECTS = [EmbeddingObject(image_id=id, embedding=embedding.astype(float)) for id, embedding in zip(query_rows['coactive_image_id'], query_rows['embedding'])]
    print(len(QUERY_EMBEDDING_OBJECTS))

    # FILE_OFFSETS = list(range(0, 10000000, 5000))
    FILE_OFFSETS = list(range(0, 100000, 5000))
    results = Parallel(n_jobs=-1)(
        delayed(load_and_create_objects)(file_lo) for file_lo in tqdm(FILE_OFFSETS)
    )
    DATASET_EMBEDDING_OBJECTS = [obj for sublist in results for obj in sublist]
    print(len(DATASET_EMBEDDING_OBJECTS))    

    return QUERY_EMBEDDING_OBJECTS, DATASET_EMBEDDING_OBJECTS

