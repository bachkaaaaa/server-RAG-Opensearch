from flask import Flask, request, jsonify
import pandas as pd
from opensearchpy import OpenSearch
from sentence_transformers import SentenceTransformer
import requests
import json

app = Flask(__name__)

# Initialize OpenSearch client
client = OpenSearch(
    hosts=[{'host': 'localhost', 'port': 9200}],
    http_auth=('admin', 'admin'),
    use_ssl=False,
    verify_certs=False,
    ssl_assert_hostname=False,
    ssl_show_warn=False
)

# Initialize sentence transformer model
model = SentenceTransformer('all-mpnet-base-v2')

# Load and prepare data (same as notebook)
df = pd.read_csv("myntra_products_catalog.csv").loc[:499]
df.fillna("None", inplace=True)
df["DescriptionVector"] = df["Description"].apply(lambda x: model.encode(x))

# Ollama server configuration
OLLAMA_URL = "http://localhost:11434/api/generate"  # Adjust if your Ollama server is different

@app.route('/search', methods=['POST'])
def search():
    try:
        # Get query from request
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({"error": "Query parameter is required"}), 400
        
        input_keyword = data['query']
        
        # Perform OpenSearch query
        vector_of_input_keyword = model.encode(input_keyword).tolist()
        
        query = {
            "size": 2,
            "query": {
                "knn": {
                    "DescriptionVector": {
                        "vector": vector_of_input_keyword,
                        "k": 2
                    }
                }
            },
            "_source": ["ProductName", "Description"]
        }
        
        search_results = client.search(body=query, index="all_products")
        hits = search_results["hits"]["hits"]
        
        # Format search results
        formatted_results = [
            {
                "ProductName": hit["_source"]["ProductName"],
                "Description": hit["_source"]["Description"],
                "Score": hit["_score"]
            } for hit in hits
        ]
        
        # Prepare prompt for Ollama
        prompt = f"""
Query: {input_keyword}

Search Results:
{json.dumps(formatted_results, indent=2)}

Based on the query and search results above, provide a concise and relevant response to the user's query.
"""
        
        # Send to Ollama
        ollama_payload = {
            "model": "deepseek-r1:1.5B",
            "prompt": prompt,
            "stream": False
        }
        
        ollama_response = requests.post(OLLAMA_URL, json=ollama_payload)
        
        if ollama_response.status_code != 200:
            return jsonify({"error": "Failed to get response from Ollama server"}), 500
        
        ollama_result = ollama_response.json()
        response_text = ollama_result.get("response", "No response from Ollama")
        
        # Return response to original caller
        return jsonify({
            "query": input_keyword,
            "search_results": formatted_results,
            "ollama_response": response_text
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Create index if it doesn't exist
    opensearch_mapping = {
        "mappings": {
            "properties": {
                "ProductID": {"type": "keyword"},
                "ProductName": {"type": "text"},
                "Description": {"type": "text"},
                "DescriptionVector": {
                    "type": "knn_vector",
                    "dimension": 768,
                    "method": {
                        "name": "hnsw",
                        "space_type": "cosinesimil",
                        "engine": "nmslib",
                        "parameters": {
                            "ef_construction": 512,
                            "m": 16
                        }
                    }
                }
            }
        }
    }
    
    try:
        client.indices.create(
            index="all_products",
            body={
                "settings": {"index.knn": True},
                "mappings": opensearch_mapping["mappings"]
            }
        )
        
        # Ingest data
        record_list = df.to_dict("records")
        for record in record_list:
            if hasattr(record["DescriptionVector"], "tolist"):
                record["DescriptionVector"] = record["DescriptionVector"].tolist()
            client.index(
                index="all_products",
                body=record,
                id=record["ProductID"],
                refresh=True
            )
    except Exception as e:
        print(f"Index creation or data ingestion failed: {e}")
    
    app.run(debug=True, host='0.0.0.0', port=5000)