from FlagEmbedding import BGEM3FlagModel
from qdrant_client import QdrantClient
import os
from dotenv import load_dotenv
load_dotenv()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

SENTEMB = BGEM3FlagModel('BAAI/bge-m3',
                       use_fp16=True) 

client = QdrantClient(
   url = "https://e69451a1-5421-44b2-804e-7cfd44b35d4f.us-east4-0.gcp.cloud.qdrant.io:6333",
   api_key=os.environ['qdrant_api_key'],
)

def qdrant_search(collection_name, reformulated_query, limit):
    query_results = client.search(
        collection_name=collection_name,
        query_vector=SENTEMB.encode(reformulated_query)['dense_vecs'].tolist(),
        limit=limit,
    )
    return query_results


