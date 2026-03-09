import os
import argparse
from google.cloud import aiplatform
from vertexai.preview import rag
from google.cloud import storage
from google.cloud import bigquery
from urllib.parse import urlparse

def create_bucket(bucket_path: str):
    parsed_uri = urlparse(bucket_path)
    bucket_name = parsed_uri.netloc
    storage_client = storage.Client()
    try:
        bucket = storage_client.get_bucket(bucket_name)
        print(f"Bucket {bucket_name} already exists.")
    except Exception:
         print(f"Creating bucket {bucket_name}...")
         bucket = storage_client.create_bucket(bucket_name)
         print(f"Bucket {bucket_name} created.")

def upload_files(folder_path: str, bucket_path: str):
    parsed_uri = urlparse(bucket_path)
    bucket_name = parsed_uri.netloc
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    
    for root, _, files in os.walk(folder_path):
        for file in files:
            local_file = os.path.join(root, file)
            blob = bucket.blob(file)
            print(f"Uploading {local_file} to gs://{bucket_name}/{file}...")
            blob.upload_from_filename(local_file)

def create_rag_corpus(display_name: str, source_bucket: str):
    print(f"Creating RAG Corpus with display name '{display_name}'...")
    embedding_model_config = rag.EmbeddingModelConfig(
        publisher_model="publishers/google/models/text-embedding-004"
    )
    rag_corpus = rag.create_corpus(
        display_name=display_name, embedding_model_config=embedding_model_config
    )
    print(f"RAG Corpus created: {rag_corpus.name}")
    
    print(f"Importing files from {source_bucket}...")
    rag.import_files(
        corpus_name=rag_corpus.name,
        paths=[source_bucket],
        chunk_size=1024,
        chunk_overlap=100,
    )
    print("Files imported to RAG Corpus.")
    return rag_corpus

def create_bq_dataset_and_table(project_id: str, dataset_id: str, table_id: str, source_avro: str):
    client = bigquery.Client(project=project_id)
    
    # Create Dataset
    dataset_ref = client.dataset(dataset_id)
    try:
         client.get_dataset(dataset_ref)
         print(f"Dataset {dataset_id} already exists.")
    except Exception:
         print(f"Creating Dataset {dataset_id}...")
         dataset = bigquery.Dataset(dataset_ref)
         dataset.location = "US"
         client.create_dataset(dataset)
         print(f"Dataset {dataset_id} created.")

    # Load Table from AVRO
    table_ref = dataset_ref.table(table_id)
    job_config = bigquery.LoadJobConfig(source_format=bigquery.SourceFormat.AVRO)
    
    print(f"Loading data from {source_avro} into {dataset_id}.{table_id}...")
    load_job = client.load_table_from_uri(source_avro, table_ref, job_config=job_config)
    load_job.result()
    print(f"Table loaded. Total rows: {client.get_table(table_ref).num_rows}")

def main():
    parser = argparse.ArgumentParser(description="Setup environment for Google ADK Example")
    parser.add_argument("--project_id", required=True)
    parser.add_argument("--doc_bucket", required=True)
    parser.add_argument("--bq_bucket", required=True)
    parser.add_argument("--rag_corpus_name", default="nest_support_docs")
    parser.add_argument("--bq_dataset", default="adk_example_dataset")
    parser.add_argument("--bq_table", default="product_data")
    
    args = parser.parse_args()
    
    create_bucket(args.doc_bucket)
    create_bucket(args.bq_bucket)
    
    upload_files("nest_docs", args.doc_bucket)
    upload_files("bq_data", args.bq_bucket)
    
    import vertexai
    vertexai.init(project=args.project_id)
    
    # RAG Setup
    create_rag_corpus(args.rag_corpus_name, args.doc_bucket)
    
    # BQ Setup
    source_avro = f"{args.bq_bucket}/product_data.avro"
    create_bq_dataset_and_table(args.project_id, args.bq_dataset, args.bq_table, source_avro)

if __name__ == "__main__":
    main()
