# 1. Setup Qdrant
%pip install qdrant_client

import json
import requests
import numpy as np
from qdrant_client import QdrantClient

qdrant_host = '<REDACTED>'
qdrant_api_key = '<REDACTED>'

qdrant_client_params = {
    "url": qdrant_host, 
    "port": <REDACTED>,
    "api_key": qdrant_api_key,
}
qdrant = QdrantClient(**qdrant_client_params)


# 2. Helper Functions
MAX_RARITY_IN_VECTOR_DB = 40
def get_next_embedding_id_rarity_value() -> int:
    rarity_value = min(
        np.random.default_rng().geometric(0.5), MAX_RARITY_IN_VECTOR_DB
    )
    return rarity_value

def create_collection_http(config):
    qdrant_client = qdrant
    if qdrant_client.collection_exists(config["collection_name"]):
        qdrant_client.delete_collection(config["collection_name"])
    r = requests.put(
        qdrant_host + f"/collections/{config['collection_name']}",
        headers={"api-key": qdrant_api_key},
        data=json.dumps(config),
    )
    return r.text, r.status_code == 200

def create_index(collection_name, name, schema):
    qdrant.create_payload_index(
        collection_name=collection_name,
        field_name=name,
        field_schema=schema,
    )

def create_collection_config(collection_name: str, vector_description: dict) -> dict:
    return {
        "collection_name": collection_name,
        "vectors": vector_description,
        "shard_number": 3,
        "replication_factor": 2,
        "write_consistency_factor": 1,
        "on_disk_payload": True,
        "hnsw_config": {
            "m": 32,
            "ef_construct": 32,
            "full_scan_threshold": 10000,
            "max_indexing_threads": 0,
            "on_disk": True,
            "payload_m": 32,
        },
        "optimizers_config": {
            "deleted_threshold": 0.2,
            "vacuum_min_vector_number": 1000,
            "default_segment_number": 2,
            "memmap_threshold": 5000,
            "indexing_threshold": 10000,
            "flush_interval_sec": 5,
            "max_optimization_threads": 1,
        },
        "wal_config": {
            "wal_capacity_mb": 32,
            "wal_segments_ahead": 0,
        },
        "quantization_config": {
            "scalar": {
                "type": "int8",
                "quantile": 0.99,
                "always_ram": False,
            }
        },
    }

def create_qdrant_collection_with_indexes(collection_name: str, vector_description: dict):
    """
    Creates a Qdrant collection and adds payload indexes for level_0 to level_9.

    Parameters:
        collection_name (str): The name of the Qdrant collection.
        vector_description (dict): Dictionary describing vector configurations.
    """
    # Step 1: Create collection
    config = create_collection_config(collection_name, vector_description)
    response, success = create_collection_http(config)
    print(f"[Collection Creation] Success: {success}, Response: {response}")

    # Step 2: Create payload indexes for rarity levels
    for i in range(10):
        create_index(collection_name, f"level_{i}", models.PayloadSchemaType.INTEGER)


# 3. Upload Points Function
def qdrant_upload_points_http(rows, collection_name):
    # Mapping from collection to vector field name and row attribute
    collection_vector_mapping = {
        "open-images_resnet-50": ("abs", "embedding"),        
        "open-images_clip_vit_l14_336": ("unit", "normalised_embedding"),       
        "amazon-reviews_distilbert": ("abs", "embedding"),        
    }

    if collection_name not in collection_vector_mapping:
        raise ValueError(f"Unknown collection_name: {collection_name}")

    vector_field, row_attr = collection_vector_mapping[collection_name]

    points = []
    for row in rows:
        point = {
            "id": row.ImageID,
            "vector": {
                vector_field: getattr(row, row_attr),
            },
            "payload": {
                **{f"level_{i}": get_next_embedding_id_rarity_value() for i in range(10)}
            },
        }
        points.append(point)

    addrequest = {"points": points}

    r = requests.put(
        qdrant_host + f"/collections/{collection_name}/points",
        headers={"api-key": qdrant_api_key},
        data=json.dumps(addrequest),
    )

    return r.text, r.status_code == 200


# 4. Initialise collection
from joblib import Parallel, delayed
from pyspark.sql import SparkSession
from tqdm import tqdm
from time import sleep

def ingest_embeddings_to_qdrant_from_s3(
    collection_name: str,
    vector_description: dict,
    encoder: str,
    vector_field: str,
    embedding_field: str,
    s3_prefix: str,
    num_shards: int,
    batch_size: int = 100,
    max_retries: int = 3,
    retry_sleep_sec: int = 10
):
    """
    End-to-end pipeline to:
      - Create a Qdrant collection with proper indexes
      - Load and upload embeddings from Spark-based Delta tables on S3 in parallel

    Parameters:
        collection_name (str): Name of Qdrant collection to use/create
        vector_description (dict): e.g., {"abs1": {"size": 2048, "distance": "Euclid", "on_disk": True}}
        encoder (str): Model encoder name for S3 path (e.g., 'resnet-50', 'clip_vit_normalised')
        vector_field (str): Vector field name in Qdrant (e.g., 'abs', 'unit')
        embedding_field (str): Field name in Spark row with the embedding (e.g., 'embedding', 'normalised_embedding')
        s3_prefix (str): S3 prefix base path
        num_shards (int): Number of S3 shards (folders) to process
        batch_size (int): Number of records per upload batch
        max_retries (int): Max retry attempts per shard
        retry_sleep_sec (int): Sleep time between retries in seconds
    """
    # Step 1 & 2: Create Qdrant collection and indexes
    create_qdrant_collection_with_indexes(collection_name, vector_description)

    # Step 3: Loop through shard indices
    for idx in tqdm(range(num_shards), desc=f"Uploading to {collection_name}"):
        embeddings_path = f"{s3_prefix}_{encoder}/{idx}"
        try:
            spark_df = spark.read.format("delta").option("header", "true").load(embeddings_path)
            rows = spark_df.collect()
        except Exception as e:
            print(f"[Shard {idx}] Failed to read from S3: {e}")
            continue

        # Step 5: Retry loop for uploading
        for attempt in range(max_retries):
            try:
                results = Parallel(n_jobs=-1)(
                    delayed(qdrant_upload_points_http)(
                        rows[row_idx:row_idx + batch_size],
                        collection_name,
                        vector_field=vector_field,
                        embedding_field=embedding_field,
                    )
                    for row_idx in range(0, len(rows), batch_size)
                )
                success_count = sum(1 for _, ok in results if ok)
                print(f"Shard {idx}: {success_count}/{len(results)} batches succeeded")
                break
            except Exception as e:
                print(f"[Shard {idx} | Attempt {attempt+1}] Upload error: {e}")
                if attempt < max_retries - 1:
                    print(f"Retrying in {retry_sleep_sec}s...")
                    sleep(retry_sleep_sec)
                else:
                    print("Max retries reached. Moving to next shard.")

                
#######################################################################################
# 1 & 3. Images KDE and Ball Counting (both are done against the same collection)
ingest_embeddings_to_qdrant_from_s3(
    collection_name="open-images_resnet-50",
    vector_description={"abs": {"size": 2048, "distance": "Euclid", "on_disk": True}},
    encoder="resnet-50",
    vector_field="abs",
    embedding_field="embedding"
)

# 2. Images Softmax
ingest_embeddings_to_qdrant_from_s3(
    collection_name="open-images_clip_vit_l14_336",
    vector_description={"unit": {"size": 768, "distance": "Dot", "on_disk": True}},
    encoder="clip_vit_normalised",
    vector_field="unit",
    embedding_field="normalised_embedding"
)

# 4 & 5. Text KDE and Ball Counting (both are done against the same collection)
ingest_embeddings_to_qdrant_from_s3(
    collection_name="amazon-reviews_distilbert",
    vector_description={"abs": {"size": 768, "distance": "Euclid", "on_disk": True}},
    encoder="distilbert",
    vector_field="abs",
    embedding_field="embedding"
)