import os
import argparse
from google.cloud import aiplatform
from vertexai.preview import rag
from google.cloud import storage
from google.cloud import bigquery
from urllib.parse import urlparse

def delete_rag_corpus(display_name: str):
    print(f"Checking for RAG Corpus with display name '{display_name}'...")
    try:
        existing_corpora = rag.list_corpora()
        corpora_list = getattr(existing_corpora, 'rag_corpora', [])
        for corpus in corpora_list:
            if getattr(corpus, 'display_name', None) == display_name:
                print(f"Found corpus: {corpus.name}. Deleting...")
                rag.delete_corpus(name=corpus.name, force=True)
                print(f"Deleted RAG Corpus: {corpus.name}")
                return True
        print(f"No RAG Corpus found with display name '{display_name}'.")
    except Exception as e:
         print(f"Error deleting RAG Corpus: {e}")
    return False

def delete_bucket(bucket_path: str):
    if not bucket_path:
        return
    parsed_uri = urlparse(bucket_path)
    bucket_name = parsed_uri.netloc
    storage_client = storage.Client()
    try:
         bucket = storage_client.get_bucket(bucket_name)
         print(f"Deleting all blobs in bucket {bucket_name}...")
         blobs = bucket.list_blobs()
         for blob in blobs:
              blob.delete()
         print(f"Deleting bucket {bucket_name}...")
         bucket.delete()
         print(f"Bucket {bucket_name} deleted.")
    except Exception as e:
         print(f"Bucket {bucket_name} not found or error deleting: {e}")

def delete_bq_dataset(project_id: str, dataset_id: str):
    client = bigquery.Client(project=project_id)
    dataset_ref = client.dataset(dataset_id)
    try:
        print(f"Deleting Dataset {dataset_id} (and tables)...")
        client.delete_dataset(dataset_ref, delete_contents=True, not_found_ok=True)
        print(f"Dataset {dataset_id} deleted.")
    except Exception as e:
        print(f"Error deleting Dataset {dataset_id}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Teardown environment for Google ADK Example")
    parser.add_argument("--project_id", required=True)
    parser.add_argument("--doc_bucket", required=True)
    parser.add_argument("--bq_bucket", required=True)
    parser.add_argument("--rag_corpus_name", default="nest_support_docs")
    parser.add_argument("--bq_dataset", default="adk_example_dataset")
    
    args = parser.parse_args()
    
    import vertexai
    vertexai.init(project=args.project_id)
    
    # Teardown RAG
    delete_rag_corpus(args.rag_corpus_name)
    
    # Teardown BQ
    delete_bq_dataset(args.project_id, args.bq_dataset)
    
    # Teardown GCS
    delete_bucket(args.doc_bucket)
    delete_bucket(args.bq_bucket)

if __name__ == "__main__":
    main()
