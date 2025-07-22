# 1. Setup Qdrant
%pip install qdrant_client

import json
import requests
import numpy as np
from tqdm import tqdm
from qdrant_client import QdrantClient

qdrant_host = '<REDACTED>'
qdrant_api_key = '<REDACTED>'

qdrant_client_params = {
    "url": qdrant_host, 
    "port": 443,
    "api_key": qdrant_api_key,
}
qdrant = QdrantClient(**qdrant_client_params)


# Helper Function
def store_qdrant_levels(
    collection_name: str,
    s3_input_prefix: str,
    s3_output_prefix: str,
    num_shards: int,
    join_with_original: bool = True,
    input_suffix: str = "",
    output_suffix: str = "",
):
    """
    For each shard, retrieves Qdrant payloads for ImageIDs and merges/saves them to S3.

    Parameters:
        collection_name (str): Qdrant collection name to retrieve from
        s3_input_prefix (str): Input S3 prefix path (excluding index)
        s3_output_prefix (str): Output S3 prefix path (excluding index)
        num_shards (int): Number of partitions/shards to process (e.g., 200)
        join_with_original (bool): Whether to join payloads with input Delta rows
        input_suffix (str): Optional suffix for input paths (e.g., '_with_f')
        output_suffix (str): Optional suffix for output paths (e.g., '_with_levels')
    """
    for idx in tqdm(range(num_shards), desc=f"Processing {collection_name}"):
        input_path = f"{s3_input_prefix}{input_suffix}/{idx}"
        output_path = f"{s3_output_prefix}{output_suffix}/{idx}"

        # Load shard
        spark_df = spark.read.format("delta").option("header", "true").load(input_path)
        spark_df = spark_df.repartition(sc.defaultParallelism).cache()

        # Extract IDs
        ids = spark_df.select("ImageID").rdd.flatMap(lambda x: x).collect()

        # Retrieve payloads from Qdrant
        try:
            res = qdrant.retrieve(
                collection_name=collection_name,
                ids=ids,
                with_payload=True,
                with_vectors=False
            )
        except Exception as e:
            print(f"[Shard {idx}] Qdrant retrieval failed: {e}")
            continue

        # Convert Qdrant results to rows
        rows = [
            {
                "ImageID": r.id,
                **{f"level_{i}": r.payload.get(f"level_{i}", None) for i in range(10)}
            }
            for r in res
        ]

        # Convert to DataFrame
        df_levels = spark.createDataFrame(rows)

        # Join with original data if required
        if join_with_original:
            enriched_df = spark_df.join(df_levels, on="ImageID", how="inner")
        else:
            enriched_df = df_levels

        # Write result
        enriched_df.write.format("delta").mode("overwrite").save(output_path)


#######################################################################################
# 1. Images KDE
store_qdrant_levels(
    collection_name="SumEstimation_image_KDE",
    s3_input_prefix="<REDACTED>/resnet-50",
    s3_output_prefix="<REDACTED>/resnet-50_with_levels",
    num_shards=200,
    join_with_original=True,
)

# 2. Images Softmax
store_qdrant_levels(
    collection_name="SumEstimation_image_softmax",
    s3_input_prefix="<REDACTED>/clip_vit_normalised",
    s3_output_prefix="<REDACTED>/clip_vit_normalised_with_levels",
    num_shards=200,
    join_with_original=True,
)

# 3. Text KDE
store_qdrant_levels(
    collection_name="SumEstimation_text_KDE",
    s3_input_prefix="<REDACTED>/distilbert",
    s3_output_prefix="<REDACTED>/distilbert_with_levels",
    num_shards=2000,
    join_with_original=True,
)