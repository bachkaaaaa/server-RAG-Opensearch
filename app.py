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
# TASK DEFINITION
You are an expert log consultant, analyzer, and debugging specialist. Your task is to analyze the query and log-related search results provided, then deliver a clear diagnostic assessment and actionable solutions.

## QUERY
{input_keyword}

## SEARCH RESULTS
{json.dumps(formatted_results, indent=2)}

## INSTRUCTIONS
1. ANALYZE: First, understand the log patterns, error messages, and system behaviors described in the search results
2. IDENTIFY: Detect anomalies, error patterns, root causes, or system bottlenecks in the logs
3. DIAGNOSE: Determine the most likely underlying issues based on the log evidence
4. SOLVE: Provide specific solutions to address the identified problems
5. PREVENT: Suggest monitoring, alerting, or code improvements to prevent similar issues
6. EXPLAIN: Clarify complex log patterns or technical concepts when necessary

## RESPONSE FORMAT
- Start with a clear summary of the log analysis findings
- Include specific log entries that indicate problems, with explanation
- Provide concrete debugging steps or solutions
- When applicable, include code snippets for fixes or improved logging
- Prioritize issues by severity/impact when multiple problems exist
- End with preventative recommendations

## IMPORTANT NOTES
- Recognize common log patterns across different technologies (web servers, databases, containers, etc.)
- Interpret timestamps, sequence of events, and correlation between different log entries
- Be aware that logs may be incomplete or misleading; account for gaps in information
- If log information is insufficient, acknowledge limitations and suggest what additional logs would help
- Consider environmental factors (load, memory, network, etc.) that might contribute to issues

Respond as an experienced SRE/DevOps engineer helping diagnose and resolve production issues.
"""
        
        # Send to Ollama
        ollama_payload = {
            "model": "deepseek-coder-v2:latest",
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
    
            "response": response_text
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