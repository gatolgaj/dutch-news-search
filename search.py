import json
from pprint import pprint
import os

from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer

load_dotenv()


class Search:
    def __init__(self):
        # Use the same embedding model you used during indexing
        self.model = SentenceTransformer('embaas/sentence-transformers-e5-large-v2')

        # Connect to Elasticsearch
        self.es = Elasticsearch(
            "http://localhost:9200",
            basic_auth=("elastic", "elastic"), 
            verify_certs=False
        )

        # Verify the connection
        client_info = self.es.info()
        print("Connected to Elasticsearch!")
        pprint(client_info)

    def get_embedding(self, text):
        # Generate embedding and convert to list
        return self.model.encode(text).tolist()

    def search(self, body):
        # Use your index name 'news_articles_2'
        return self.es.search(index="news_articles_2", body=body)

    def retrieve_document(self, id):
        return self.es.get(index="news_articles_2", id=id)