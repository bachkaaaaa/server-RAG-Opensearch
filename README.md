# Flask Search API with OpenSearch and Ollama

This repository contains a Flask-based search API that processes queries using OpenSearch for vector-based search and Ollama for generating responses. It leverages a product catalog dataset (`myntra_products_catalog.csv`) to search for products based on their descriptions.

## Features

- **Vector Search**: Uses OpenSearch for efficient product search
- **Natural Language Responses**: Generates human-like responses using Ollama's `deepseek-r1:1.5B` model
- **RESTful API**: Exposes a single `/search` endpoint for querying products
- **Automatic Indexing**: Ingests the dataset into OpenSearch on startup

## Prerequisites

- **Python**: Version 3.12.3 recommended
- **OpenSearch**: Running on `localhost:9200` with `admin:admin` credentials
- **Ollama**: Running on `localhost:11434` with the `deepseek-r1:1.5B` model
- **Dataset**: `myntra_products_catalog.csv` (not included; obtain separately and place in the project root)

## Setup Instructions

### 1. Clone the Repository
```bash
git clone <repository-url>
cd <repository-name>
```


### 2. Create and Activate a Virtual Environment
```bash
python -m venv .venv
```
###  3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Ensure OpenSearch and Ollama are Running
- **Start OpenSearch**: Verify accessibility at `http://localhost:9200`
- **Start Ollama**: Confirm the `deepseek-r1:1.5B` model is available at `http://localhost:11434`

### 6. Run the Flask Server
```bash
python app.py
```