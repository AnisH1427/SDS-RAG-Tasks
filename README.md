# SDS-RAG-Tasks

## Project Introduction
This repository was created as part of a technical assessment for Smart Data Solution. The goal is to build a Retrieval-Augmented Generation (RAG) pipeline that processes, chunks, and retrieves information from financial PDF documents, enabling efficient question answering and data extraction.

## Data Description
- **Type of Data**: The primary data source consists of financial reports in PDF format (e.g., `2022 Q3 AAPL.pdf`).
- **Data Flow**: Raw PDFs are processed, text is extracted, chunked, and stored for downstream retrieval and embedding tasks. The extraction and chunking results are saved as JSON files in `data/processed/`.

## Project Hierarchy 
```
SDS-RAG-Tasks/
│
├── config/                
│   └── prompt_config.yaml # System prompt
│
├── data/                  
│   ├── raw/               knowledge source
│   │   └── 2022 Q3 AAPL.pdf
│   └── processed/         # Processed and chunked data (JSON)
│       ├── extracted_pdf_data.json   
│       └── chunked_pdf_data.json     
│
├── docs/                  
│
├── noteboks/              
│   └── testing_of_text_extraction_method.ipynb # Extracts data and exports JSON format
│
├── src/                   
│   ├── chunking.py        
│   ├── embeddings_store.py
│   ├── main.py            
│   └── retrieval.py       
│
├── requirements.txt       # Python dependencies (for pip users)
├── pyproject.toml         # Project metadata and dependencies
└── README.md              # Project documentation
```

## Project Workflow

1. **Extraction & Chunking**  
   Use the notebook in `noteboks/` (e.g., `testing_of_text_extraction_method.ipynb`) to extract and chunk data from PDFs. This will export `extracted_pdf_data.json` and `chunked_pdf_data.json` in `data/processed/`.

2. **Embedding Generation & Storage**  
   Run the embedding script (e.g., `src/embeddings_store.py`) to generate embeddings using Google Embedding and store them in Qdrant. This step is separate from the Streamlit app.

3. **Retrieval & QA (Streamlit App)**  
   Use the Streamlit app to query, retrieve, and interact with the processed and embedded data.

## Setup Instructions

### Option 1: Quick Start with pip
1. **Create a virtual environment:**
   - On Windows:
     ```sh
     python -m venv .venv
     .venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```sh
     python3 -m venv .venv
     source .venv/bin/activate
     ```
2. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

### Option 2: Recommended (using uv and pyproject.toml)
1. **Install [uv](https://github.com/astral-sh/uv):**
   ```sh
   pip install uv
   ```
2. **Create and activate a virtual environment:**
   ```sh
   uv venv .venv
   # Activate as above
   ```
3. **Install dependencies:**
   ```sh
   uv sync
   ```

### Set Environment Variables
Create a `.env` file in the project root with the following variables:
```
GROQ_API_KEY=your_groq_api_key
QDRANT_API_KEY=your_qdrant_api_key
QDRANT_URL=your_qdrant_url
QDRANT_COLLECTION_NAME=your_collection_name
GOOGLE_API_KEY=your_google_api_key
EMBEDDING_MODEL=models/text-embedding-004
MODEL=llama-3.1-8b-instant
```

## Usage
To use the app, make sure the processed data exists in `data/processed/` (run the notebook in `noteboks/` if not), and embeddings are generated and stored in Qdrant (run `src/embeddings_store.py` if not). Then start the Streamlit application:
```sh
streamlit run src/main.py
```
This will launch the web interface for querying and interacting with your knowledge base.

## Re-evaluation & Testing
- For custom configurations, edit `config/prompt_config.yaml`.
- Use the processed data in `data/processed/` for downstream tasks or QA.

## Deployment
The application is also hosted on Streamlit Cloud. You can access it directly here:

[https://sds-rag-tasks-y64gs2urakzskcdrmojtjy.streamlit.app/](https://sds-rag-tasks-y64gs2urakzskcdrmojtjy.streamlit.app/)

## License
See [LICENSE](LICENSE) for details.


