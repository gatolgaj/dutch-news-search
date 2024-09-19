import pandas as pd
from elasticsearch import Elasticsearch, helpers
import spacy
from sentence_transformers import SentenceTransformer
from datetime import datetime as dt
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Initialize spaCy for Dutch language
nlp = spacy.load('nl_core_news_sm')

# Initialize embedding model for multilingual support
model = SentenceTransformer('embaas/sentence-transformers-e5-large-v2')

# Connect to Elasticsearch 8.x
es = Elasticsearch(
    "http://localhost:9200",
    basic_auth=("elastic", "elastic"),  # Replace with your credentials
    verify_certs=False
)

# Define index name
index_name = 'news_articles_2'

# Delete the index if it already exists (optional)
if es.indices.exists(index=index_name):
    es.indices.delete(index=index_name)

# Create index with appropriate mappings, including datetime format correction
mapping = {
    "mappings": {
        "properties": {
            "datetime": {
                "type": "date",
                "format": "yyyy-MM-dd HH:mm:ss"
            },
            "title": {"type": "text"},
            "content": {"type": "text"},
            "category": {"type": "keyword"},
            "url": {"type": "keyword"},
            "keywords": {"type": "keyword"},
            "embedding": {
                "type": "dense_vector",
                "dims": 1024,
                "similarity": "l2_norm"
            }
        }
    }
}

# Create the index
es.indices.create(index=index_name, body=mapping)

# Function to extract keywords from text
def extract_keywords(text):
    doc = nlp(text)
    keywords = set()
    for chunk in doc.noun_chunks:
        keywords.add(chunk.text.lower())
    for ent in doc.ents:
        keywords.add(ent.text.lower())
    return list(keywords)

# Function to generate actions for bulk indexing
def generate_actions(df_chunk):
    actions = []
    for index, row in df_chunk.iterrows():
        datetime_str = row['datetime']
        title = row['title']
        content = row['content']
        category = row['category']
        url = row['url']

        # Validate and parse datetime
        try:
            datetime_value = dt.strptime(datetime_str.strip(), '%Y-%m-%d %H:%M:%S')
            # Convert back to string in the correct format
            datetime_value = datetime_value.strftime('%Y-%m-%d %H:%M:%S')
        except ValueError:
            print(f"Invalid datetime format for document ID {index}. Skipping.")
            continue

        # Combine title and content
        text = f"{title} {content}"

        # Extract keywords
        keywords = extract_keywords(text)

        # Generate embedding
        embedding = model.encode(text)

        # Check for NaN or infinite values in embedding
        if np.isnan(embedding).any() or np.isinf(embedding).any():
            print(f"Invalid embedding for document ID {index}. Skipping.")
            continue

        embedding = embedding.tolist()  # Convert to list for JSON serialization

        # Prepare document
        doc = {
            "_index": index_name,
            "_id": index,  # Use the DataFrame index as the document ID
            "_source": {
                "datetime": datetime_value,
                "title": title,
                "content": content,
                "category": category,
                "url": url,
                "keywords": keywords,
                "embedding": embedding
            }
        }
        actions.append(doc)
    return actions

# Read CSV and index data in chunks
chunk_size = 200  # Adjust based on memory constraints
csv_file = 'news.csv'  # Path to your CSV file

from elasticsearch.helpers import BulkIndexError

# Get total number of lines for progress bar
total_lines = sum(1 for _ in open(csv_file)) - 1  # Subtract header
total_chunks = (total_lines // chunk_size) + 1

# Use ThreadPoolExecutor for parallel chunk processing
max_workers = 8  # Adjust based on your system's capabilities

# Function to process and index a single chunk
def process_and_index_chunk(chunk):
    actions = generate_actions(chunk)
    if not actions:
        return 0  # No documents indexed
    try:
        helpers.bulk(es, actions)
        return len(actions)  # Number of documents successfully indexed
    except BulkIndexError as e:
        failed_docs = len(e.errors)
        print(f"{failed_docs} document(s) failed to index.")
        for error in e.errors:
            index_error = error.get('index', {})
            status = index_error.get('status')
            error_type = index_error.get('error', {}).get('type', 'Unknown type')
            error_reason = index_error.get('error', {}).get('reason', 'Unknown reason')
            doc_id = index_error.get('_id', 'N/A')
            print(f"Document ID {doc_id} failed to index. Status: {status}, Type: {error_type}, Reason: {error_reason}")
        successful_docs = len(actions) - failed_docs
        return successful_docs
    except Exception as e:
        print(f"Error indexing chunk: {str(e)}")
        return 0  # No documents indexed due to error

indexed_documents = 0

with ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = []
    with pd.read_csv(csv_file, chunksize=chunk_size) as reader:
        for chunk in reader:
            future = executor.submit(process_and_index_chunk, chunk)
            futures.append(future)

    # Use tqdm to display progress bar
    for future in tqdm(as_completed(futures), total=len(futures), desc="Indexing progress"):
        indexed_documents += future.result()

print(f"Data indexing complete. Total documents indexed: {indexed_documents}")