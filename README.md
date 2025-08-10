# SDS-RAG-Tasks

## Project Introduction
This repository was created as part of a technical assessment for Smart Data Solution. The goal is to build a Retrieval-Augmented Generation (RAG) pipeline that processes, chunks, and retrieves information from financial PDF documents, enabling efficient question answering and data extraction.

## Data Description
- **Type of Data**: The primary data source consists of financial reports in PDF format (e.g., `2022 Q3 AAPL.pdf`).
- **Data Flow**: Raw PDFs are processed, text is extracted, chunked, and stored for downstream retrieval and embedding tasks.

## Project Hierarchy & Chain of Thought
```
SDS-RAG-Tasks/
│
├── config/                
│   └── prompt_config.yaml # System prompt instructions
│
├── data/                  
│   ├── raw/               # Raw PDF files
│   └── processed/         # Processed and chunked data (JSON)
│
├── docs/                  
│
├── noteboks/              # 
│   └── testing_of_text_extraction_method.ipynb # Experiments with extraction methods
│
├── src/                   # Source code
│   ├── chunking.py        
│   ├── embeddings_store.py
│   ├── main.py            
│   └── retrieval.py       
│
├── requirements.txt       # Python dependencies (for pip users)
├── pyproject.toml         # Project metadata and dependencies
├── uv.lock                
└── README.md              
```

## Project Workflow

1. **Experimentation & Validation**  
   Use notebooks in `noteboks/` (e.g., `testing_of_text_extraction_method.ipynb`) to test and compare different text extraction methods on your PDFs.

2. **Raw Data Ingestion**  
   Place your PDF files in `data/raw/`.

3. **Text Processing**  
   Use the Streamlit app (`src/main.py`) to extract and chunk text from PDFs. Processed data is saved in `data/processed/`.

4. **Embedding & Retrieval**  
   Leverage the app’s features to generate vector embeddings and perform semantic search over your documents.


## Setup Instructions

### Option 1: Quick Start with pip (No uv)
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
1. **Prepare Data:**
   - Place your PDF files in `data/raw/`.
2. **Run the Streamlit App:**
   - Start the application:
     ```sh
     streamlit run src/main.py
     ```
   - This will launch the web interface for uploading, processing, and querying PDFs.
3. **Experiment:**
   - Open and run notebooks in `noteboks/` (e.g., `testing_of_text_extraction_method.ipynb`) to test and compare extraction methods.

## Re-evaluation & Testing
- To re-run the pipeline, update or add new PDFs in `data/raw/` and use the Streamlit app.
- For custom configurations, edit `config/prompt_config.yaml`.
- Use the processed data in `data/processed/` for downstream tasks or QA.

## License
See [LICENSE](LICENSE) for details.

