import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Document paths that work in both local and Vercel
TAX_DOCUMENTS = {
    'tax_data': os.path.join(BASE_DIR, 'data', 'tax_rates.csv'),
    'tax_code': os.path.join(BASE_DIR, 'data', 'tax_code.pdf'),
    'tax_instructions': os.path.join(BASE_DIR, 'data', 'tax_instructions.pdf'),
    'tax_presentation': os.path.join(BASE_DIR, 'data', 'tax_presentation.pptx')
}

# Model settings
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
MAX_CHUNK_SIZE = 1000
TOP_K_RESULTS = 3
SIMILARITY_THRESHOLD = 0.7

# Chunking settings
CHUNK_OVERLAP = 200    # characters 