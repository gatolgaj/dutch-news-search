# News Articles Elasticsearch Indexing Script

This script reads news articles from a CSV file, processes each article to extract keywords and generate embeddings, and indexes the data into an Elasticsearch cluster. It leverages spaCy for keyword extraction and SentenceTransformer for generating text embeddings. The script is designed to handle large datasets efficiently by processing and indexing data in chunks using multithreading.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Code Explanation](#code-explanation)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## Prerequisites

Before running the script, ensure you have the following installed and configured:

- **Python 3.7 or higher**
- **Elasticsearch 8.x** running locally or accessible remotely
- **Required Python Packages:**
  - `pandas`
  - `elasticsearch`
  - `elasticsearch[async]`
  - `spacy`
  - `sentence-transformers`
  - `numpy`
  - `tqdm`
- **spaCy Dutch Language Model:**
  - `nl_core_news_sm`
- **SentenceTransformer Model:**
  - `embaas/sentence-transformers-e5-large-v2`
- **CSV File:**
  - A CSV file named `news.csv` containing the news articles data with the following columns:
    - `datetime` (format: `YYYY-MM-DD HH:MM:SS`)
    - `title`
    - `content`
    - `category`
    - `url`

## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/news-elasticsearch-indexing.git
   cd news-elasticsearch-indexing
   ```

2. **Create a Virtual Environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install Required Packages**

   ```bash
   pip install -r requirements.txt
   ```

   *If you don't have a `requirements.txt` file, you can install packages manually:*

   ```bash
   pip install pandas elasticsearch spacy sentence-transformers numpy tqdm
   ```

4. **Download spaCy Dutch Language Model**

   ```bash
   python -m spacy download nl_core_news_sm
   ```

## Configuration

1. **Elasticsearch Connection**

   Update the Elasticsearch connection details in the script if necessary:

   ```python
   es = Elasticsearch(
       "http://localhost:9200",
       basic_auth=("elastic", "elastic"),  # Replace with your credentials
       verify_certs=False
   )
   ```

   - Replace `"http://localhost:9200"` with your Elasticsearch host if it's running remotely.
   - Update `basic_auth` with your Elasticsearch username and password.

2. **Index Name**

   Set the desired Elasticsearch index name:

   ```python
   index_name = 'news_articles'
   ```

3. **CSV File Path**

   Ensure the CSV file path is correct:

   ```python
   csv_file = 'news.csv'  # Path to your CSV file
   ```

4. **Adjust Chunk Size and Workers**

   - **Chunk Size:** Adjust based on your memory constraints.

     ```python
     chunk_size = 200  # Number of records per chunk
     ```

   - **Max Workers:** Adjust based on your CPU capabilities.

     ```python
     max_workers = 8  # Number of threads to use
     ```

## Usage

1. **Start Elasticsearch**

   Ensure your Elasticsearch instance is running and accessible.

2. **Run the Script**

   ```bash
   python index_news_articles.py
   ```

3. **Monitor Progress**

   The script uses `tqdm` to display a progress bar indicating the indexing progress.

4. **Verify Data in Elasticsearch**

   After completion, you can verify the indexed data using Kibana or Elasticsearch APIs:

   ```bash
   curl -X GET "localhost:9200/news_articles/_search?pretty"
   ```

4. **Run the search Application**

 
    Search app is a simple Flask application which will point to the ElasticSearch and run various queries
    ```bash
    flask run
    ```
     apllication will be running at `localhost:5001`

## Code Explanation

### 1. Imports and Initializations

```python
import pandas as pd
from elasticsearch import Elasticsearch, helpers
import spacy
from sentence_transformers import SentenceTransformer
from datetime import datetime as dt
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
```

- **Libraries Used:**
  - `pandas`: Reading CSV files.
  - `elasticsearch`: Connecting to Elasticsearch.
  - `spacy`: Keyword extraction.
  - `sentence-transformers`: Generating text embeddings.
  - `numpy`: Numerical operations.
  - `concurrent.futures`: Multithreading.
  - `tqdm`: Progress bar.

### 2. Load Models

```python
# Initialize spaCy for Dutch language
nlp = spacy.load('nl_core_news_sm')

# Initialize embedding model
model = SentenceTransformer('embaas/sentence-transformers-e5-large-v2')
```

- Loads the spaCy model for keyword extraction.
- Loads the SentenceTransformer model for generating embeddings.

### 3. Elasticsearch Setup

```python
# Connect to Elasticsearch
es = Elasticsearch(
    "http://localhost:9200",
    basic_auth=("elastic", "elastic"),  # Replace with your credentials
    verify_certs=False
)

# Define index name
index_name = 'news_articles'

# Delete the index if it already exists
if es.indices.exists(index=index_name):
    es.indices.delete(index=index_name)

# Create index with mappings
mapping = {
    "mappings": {
        "properties": {
            "datetime": {"type": "date", "format": "yyyy-MM-dd HH:mm:ss"},
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
```

- Establishes a connection to Elasticsearch.
- Defines and creates the index with the appropriate mappings.

### 4. Keyword Extraction Function

```python
def extract_keywords(text):
    doc = nlp(text)
    keywords = set()
    for chunk in doc.noun_chunks:
        keywords.add(chunk.text.lower())
    for ent in doc.ents:
        keywords.add(ent.text.lower())
    return list(keywords)
```

- Uses spaCy to extract noun chunks and named entities as keywords.

### 5. Generate Actions for Bulk Indexing

```python
def generate_actions(df_chunk):
    actions = []
    for index, row in df_chunk.iterrows():
        # Extract and process fields
        # Validate datetime
        # Generate embedding
        # Prepare document
        doc = {
            "_index": index_name,
            "_id": index,
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
```

- Processes each row in the DataFrame chunk.
- Validates and formats the datetime field.
- Combines title and content for embedding generation.
- Extracts keywords and generates embeddings.
- Prepares actions for bulk indexing.

### 6. Multithreaded Chunk Processing

```python
# Function to process and index a single chunk
def process_and_index_chunk(chunk):
    actions = generate_actions(chunk)
    if not actions:
        return 0
    try:
        helpers.bulk(es, actions)
        return len(actions)
    except BulkIndexError as e:
        # Handle indexing errors
        return successful_docs
    except Exception as e:
        # Handle other exceptions
        return 0
```

- Uses `ThreadPoolExecutor` to process chunks in parallel.
- Handles exceptions during indexing.

### 7. Reading CSV and Indexing Data

```python
indexed_documents = 0

with ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = []
    with pd.read_csv(csv_file, chunksize=chunk_size) as reader:
        for chunk in reader:
            future = executor.submit(process_and_index_chunk, chunk)
            futures.append(future)

    # Progress bar
    for future in tqdm(as_completed(futures), total=len(futures), desc="Indexing progress"):
        indexed_documents += future.result()

print(f"Data indexing complete. Total documents indexed: {indexed_documents}")
```

- Reads the CSV file in chunks.
- Submits each chunk to the thread pool for processing.
- Displays a progress bar using `tqdm`.
- Outputs the total number of documents indexed.

## Troubleshooting

- **Elasticsearch Connection Errors:**
  - Ensure Elasticsearch is running and the connection details are correct.
  - Verify that the `basic_auth` credentials are correct.

- **Model Loading Errors:**
  - Make sure the spaCy and SentenceTransformer models are downloaded.
  - If you encounter issues, try updating the models or reinstalling them.

- **Memory Constraints:**
  - Adjust the `chunk_size` if you run into memory issues.
  - Reduce the `max_workers` if your CPU is overloaded.

- **CSV File Issues:**
  - Ensure the CSV file exists and the path is correct.
  - Verify that the CSV has the required columns with correct data types.

- **Datetime Parsing Errors:**
  - The script expects the `datetime` column in `YYYY-MM-DD HH:MM:SS` format.
  - If your data uses a different format, modify the `strptime` format string accordingly.

- **Embedding Generation Errors:**
  - If embeddings contain `NaN` or infinite values, check the input text for anomalies.
  - Ensure the SentenceTransformer model supports the language of your text.

- **Elasticsearch Indexing Errors:**
  - Check Elasticsearch logs for detailed error messages.
  - Verify that the index mappings match the data types being indexed.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

- **spaCy:** Industrial-strength Natural Language Processing in Python.
- **SentenceTransformers:** Multilingual Sentence Embeddings using BERT.
- **Elasticsearch:** Distributed, RESTful search and analytics engine.