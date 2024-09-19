import os
from flask import Flask, render_template, request, redirect, url_for, abort
from elasticsearch import Elasticsearch, exceptions as es_exceptions
from sentence_transformers import SentenceTransformer
from datetime import datetime
from werkzeug.exceptions import HTTPException
from werkzeug.middleware.proxy_fix import ProxyFix
import re

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'your_secret_key')

# Adjust this if your app is behind a proxy (e.g., when deploying with Gunicorn and Nginx)
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_host=1)

class Search:
    def __init__(self):
        # Initialize the embedding model
        self.model = SentenceTransformer('embaas/sentence-transformers-e5-large-v2')

        # Connect to Elasticsearch
        try:
            self.es = Elasticsearch(
                "http://localhost:9200",
                basic_auth=("elastic", "elastic"),  # Replace with your password
                verify_certs=False,
                timeout=30,
                max_retries=10,
                retry_on_timeout=True
            )
            if not self.es.ping():
                raise ConnectionError("Cannot connect to Elasticsearch")
            print("Connected to Elasticsearch!, Please check the Password")
        except es_exceptions.ConnectionError as e:
            print(f"Elasticsearch connection error: {e}")
            raise

    def get_embedding(self, text):
        # Generate embedding and convert to list
        return self.model.encode(text).tolist()

    def search(self, index, body):
        try:
            return self.es.search(index=index, body=body)
        except es_exceptions.ConnectionError as e:
            print(f"Elasticsearch connection error: {e}")
            abort(503, description="Service Unavailable")
        except es_exceptions.TransportError as e:
            print(f"Elasticsearch transport error: {e}")
            abort(500, description="Internal Server Error")

    def retrieve_document(self, index, id):
        try:
            return self.es.get(index=index, id=id)
        except es_exceptions.NotFoundError:
            abort(404, description="Document not found")
        except es_exceptions.ConnectionError as e:
            print(f"Elasticsearch connection error: {e}")
            abort(503, description="Service Unavailable")
        except es_exceptions.TransportError as e:
            print(f"Elasticsearch transport error: {e}")
            abort(500, description="Internal Server Error")

# Initialize the Search class
es_client = Search()

# Define index name
INDEX_NAME = 'news_articles_2'

@app.errorhandler(HTTPException)
def handle_exception(e):
    """Return custom error page for HTTP errors."""
    return render_template("error.html", error=e), e.code

@app.route('/', methods=['GET', 'POST'])
def handle_search():
    if request.method == 'POST':
        return perform_search()
    else:
        # Initial page load: render the search page with minimal form
        return render_template(
            "index.html",
            results=None,
            categories=[],
            selected_category='',
            selected_keywords='',
            query='',
            from_=0,
            total=0,
            aggs={}
        )

def perform_search():
    # Get form data with default values
    query = request.form.get("query", "").strip()
    from_ = request.form.get("from_", type=int, default=0)
    selected_category = request.form.get("category", "").strip()
    selected_keywords = request.form.get("keywords", "").strip()

    # Input validation
    if from_ < 0:
        from_ = 0

    # Generate embedding for the query if provided
    embedding_vector = es_client.get_embedding(query) if query else None

    # Build must and filter clauses
    must_clauses = []
    filter_clauses = []

    if query:
        must_clauses.append({
            "multi_match": {
                "query": query,
                "fields": ["title", "content", "keywords"],
                "type": "best_fields",
                "operator": "and"
            }
        })
    else:
        must_clauses.append({"match_all": {}})

    filters, parsed_query = extract_filters(query)

    # # Add category filter if selected
    # if selected_category:
    #     filter_clauses.append({
    #         "term": {"category": selected_category}
    #     })

    # # Add keywords filter if entered
    # if selected_keywords:
    #     keywords_list = [kw.strip() for kw in selected_keywords.split(",") if kw.strip()]
    #     if keywords_list:
    #         # Use terms query for multiple keywords
    #         filter_clauses.append({
    #             "terms": {"keywords": keywords_list}
    #         })

    # Build the Elasticsearch query
    es_query = {
        "size": 5,
        "from": from_,
        "query": {
            "bool": {
                "must": must_clauses,
                "filter": filters
            }
        },
        "knn":{
            "field": "embedding",
            "query_vector": es_client.get_embedding(parsed_query),
            "k": 10,
            "num_candidates": 1500,
        },
        "rank":{"rrf": {}},
        "aggs": {
            "category-agg": {
                "terms": {
                    "field": "category",
                    "size": 10  # Adjust as needed
                }
            },
            "keywords-agg": {
                "terms": {
                    "field": "keywords",
                    "size": 10  # Adjust as needed
                }
            },
            "year-agg": {
                "date_histogram": {
                    "field": "datetime",
                    "calendar_interval": "year",
                    "format": "yyyy",
                },
            },
        },
        # "highlight": {
        #     "fields": {
        #         "title": {},
        #         "content": {},
        #         "keywords": {}
        #     }
        # }
    }

    # If embedding is available, use script_score for semantic search
    if embedding_vector:
        es_query["query"] = {
            "script_score": {
                "query": es_query["query"],
                "script": {
                    "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                    "params": {"query_vector": embedding_vector}
                }
            }
        }

    # Execute the search
    results = es_client.search(index=INDEX_NAME, body=es_query)

    # Process aggregations
    aggs = {
        "Category": {
            bucket["key"]: bucket["doc_count"]
            for bucket in results.get("aggregations", {}).get("category-agg", {}).get("buckets", [])
        },
        "Keywords": {
            bucket["key"]: bucket["doc_count"]
            for bucket in results.get("aggregations", {}).get("keywords-agg", {}).get("buckets", [])
        },
        "Year": {
            bucket["key_as_string"]: bucket["doc_count"]
            for bucket in results.get("aggregations", {}).get("year-agg", {}).get("buckets", [])
            if bucket["doc_count"] > 0
        },
    }

    # Get list of categories for the category dropdown
    categories = sorted(aggs["Category"].keys())
    # print(categories)
    # print(results)
    return render_template(
        "index.html",
        results=results["hits"]["hits"],
        query=parsed_query,
        from_=from_,
        total=results["hits"]["total"]["value"],
        aggs=aggs,
        categories=categories,
        selected_category=selected_category,
        selected_keywords=selected_keywords,
    )

@app.route('/document/<id>', methods=['GET'])
def get_document(id):
    # Retrieve the document by ID
    document = es_client.retrieve_document(index=INDEX_NAME, id=id)
    if not document:
        abort(404, description="Document not found")

    # Extract fields
    title = document["_source"].get("title", "Untitled")
    content = document["_source"].get("content", "")
    paragraphs = content.split("\n") if content else []

    return render_template("document.html", title=title, paragraphs=paragraphs)

def extract_filters(query):
    category_regex = r"category:([^\s]+)\s*"
    keywords_regex = r"keywords:([^\s]+)\s*"

    # Extract category
    category_match = re.search(category_regex, query)
    if category_match:
        category = category_match.group(1)
        query = re.sub(category_regex, "", query).strip()
    else:
        category = None

    # Extract all keywords
    keywords = re.findall(keywords_regex, query)
    if keywords:
        query = re.sub(keywords_regex, "", query).strip()

    # Build filters
    filters = {"filter": []}
    if category:
        filters["filter"].append({"term": {"category": {"value": category}}})
    if keywords:
        filters["filter"].append({"terms": {"keywords": keywords}})

    # Return filters and the cleaned query
    if not filters["filter"]:
        return [], query  # No filters found
    else:
        print(filters["filter"])
        return filters["filter"], query

if __name__ == '__main__':
    app.run(debug=False)