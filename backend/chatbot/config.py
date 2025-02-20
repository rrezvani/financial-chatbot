import os

# Base path for data files
DATA_PATH = "/Users/romtinrezvani/Downloads"

# Document paths
TAX_DOCUMENTS = {
    'tax_code': os.path.join(DATA_PATH, "usc26@118-78.pdf"),
    'tax_instructions': os.path.join(DATA_PATH, "i1040gi.pdf"),
    'tax_presentation': os.path.join(DATA_PATH, "MIC_3e_Ch11.pptx"),
    'tax_data': os.path.join(DATA_PATH, "tax_data.csv")
}

# Vector search settings
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
SIMILARITY_THRESHOLD = 0.5
TOP_K_RESULTS = 3

# Chunking settings
MAX_CHUNK_SIZE = 1000  # characters
CHUNK_OVERLAP = 200    # characters 