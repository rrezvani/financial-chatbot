import pandas as pd
import fitz  # PyMuPDF for PDF processing
from pptx import Presentation
from sentence_transformers import SentenceTransformer
import faiss
import networkx as nx
import json
from pathlib import Path
import numpy as np

class DataProcessor:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.index = None
        self.graph = nx.Graph()
        self.documents = []
        self.data_dir = Path('data')
        self.data_dir.mkdir(exist_ok=True)
        
    def process_csv(self, file_path):
        """Process tabular financial data"""
        df = pd.read_csv(file_path)
        for _, row in df.iterrows():
            # Convert row to string representation
            text = " ".join([f"{col}: {val}" for col, val in row.items()])
            self.documents.append({
                'content': text,
                'source': 'csv',
                'metadata': dict(row)
            })
            
    def process_pdf(self, file_path):
        """Extract text from PDF reports"""
        doc = fitz.open(file_path)
        for page_num in range(len(doc)):
            text = doc[page_num].get_text()
            self.documents.append({
                'content': text,
                'source': 'pdf',
                'metadata': {'page': page_num}
            })
            
    def process_ppt(self, file_path):
        """Extract text from PowerPoint presentations"""
        prs = Presentation(file_path)
        for slide_num, slide in enumerate(prs.slides):
            text = ""
            for shape in slide.shapes:
                if hasattr(shape, "text"):
                    text += shape.text + " "
            if text.strip():
                self.documents.append({
                    'content': text,
                    'source': 'ppt',
                    'metadata': {'slide': slide_num}
                })
                
    def build_search_index(self):
        """Build FAISS index for vector search"""
        texts = [doc['content'] for doc in self.documents]
        embeddings = self.model.encode(texts)
        
        # Initialize FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings.astype('float32'))
        
        # Build knowledge graph
        for i, doc in enumerate(self.documents):
            self.graph.add_node(i, **doc)
            # Add edges between similar documents
            if i > 0:
                similarity = self.calculate_similarity(embeddings[i], embeddings[i-1])
                if similarity > 0.7:  # Threshold for similarity
                    self.graph.add_edge(i, i-1, weight=similarity)
    
    def calculate_similarity(self, vec1, vec2):
        """Calculate cosine similarity between vectors"""
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    
    def search(self, query, k=5):
        """Perform semantic search across all data sources"""
        query_vector = self.model.encode([query])
        D, I = self.index.search(query_vector.astype('float32'), k)
        
        results = []
        for idx in I[0]:
            doc = self.documents[idx]
            # Get related documents from graph
            related = list(self.graph.neighbors(idx))
            related_docs = [self.documents[r] for r in related]
            
            results.append({
                'content': doc['content'],
                'source': doc['source'],
                'metadata': doc['metadata'],
                'related_documents': related_docs
            })
            
        return results 