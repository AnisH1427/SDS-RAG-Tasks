import os
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter

EXTRACTED_DATA_PATH = os.path.join(
    os.path.dirname(__file__), '../data/processed/extracted_pdf_data.json'
)
CHUNKED_DATA_PATH = os.path.join(
    os.path.dirname(__file__), '../data/processed/chunked_pdf_data.json'
)

def load_extracted_data():
    with open(EXTRACTED_DATA_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)

def chunk_all_pages(chunk_size=500, chunk_overlap=50):
    data = load_extracted_data()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunked = []
    for page, content in data.items():
        text = content.get('text', '')
        if text:
            chunks = splitter.split_text(text)
            for i, chunk in enumerate(chunks):
                chunked.append({
                    'page': page,
                    'chunk_id': f'{page}_chunk_{i}',
                    'text': chunk
                })
    with open(CHUNKED_DATA_PATH, 'w', encoding='utf-8') as f:
        json.dump(chunked, f, ensure_ascii=False, indent=2)
    print(f'Chunked data written to {CHUNKED_DATA_PATH} with {len(chunked)} chunks.')

if __name__ == '__main__':
    chunk_all_pages()
