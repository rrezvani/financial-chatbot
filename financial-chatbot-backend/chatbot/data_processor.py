import pandas as pd
import fitz  # PyMuPDF for PDF processing
from pptx import Presentation
from sentence_transformers import SentenceTransformer
import faiss
import networkx as nx
import json
from pathlib import Path
import numpy as np
import PyPDF2
import pptx
import os
from . import config

class DataProcessor:
    def __init__(self):
        # Initialize sentence transformer model
        self.model = SentenceTransformer(config.EMBEDDING_MODEL)
        
        # Store processed text chunks
        self.documents = []
        
        # Vector store for semantic search
        self.vector_store = None
        
        # Knowledge graph for relationships
        self.graph = nx.Graph()
        
        # Initialize dictionaries for tax data
        self.tax_rates = {}
        self.tax_rules = {}
        
        # Process our tax documents
        self.load_tax_documents()

    def load_tax_documents(self):
        """Load and process all tax documents once"""
        try:
            # Load CSV data
            tax_rates = pd.read_csv(config.TAX_DOCUMENTS['tax_data'])
            if tax_rates is not None:
                self.process_tax_rates(tax_rates)
            
            # Load PDF data
            try:
                with open(config.TAX_DOCUMENTS['tax_code'], 'rb') as pdf_file:
                    pdf_reader = PyPDF2.PdfReader(pdf_file)
                    self.process_tax_guidelines(pdf_reader)
            except FileNotFoundError:
                print(f"Warning: Tax code PDF not found at {config.TAX_DOCUMENTS['tax_code']}")
            
            # Load PPT data
            try:
                prs = pptx.Presentation(config.TAX_DOCUMENTS['tax_presentation'])
                self.process_tax_examples(prs)
            except FileNotFoundError:
                print(f"Warning: Tax presentation not found at {config.TAX_DOCUMENTS['tax_presentation']}")
            
            # Build search indices
            self.build_search_index()
        except Exception as e:
            print(f"Error loading tax documents: {str(e)}")

    def process_tax_rates(self, df):
        """Process tax rates from CSV"""
        try:
            # Print the columns to debug
            print("Available columns:", df.columns.tolist())
            
            for _, row in df.iterrows():
                # Use the actual column names from your CSV
                bracket_id = f"bracket_{row['Income']}"
                
                self.tax_rates[bracket_id] = {
                    'range': str(row['Income']),
                    'rate': str(row['Tax Rate']),
                    'deductions': str(row['Deductions']),
                    'conditions': f"Type: {row['Taxpayer Type']}, Year: {row['Tax Year']}, State: {row['State']}"
                }
                
                # Add to graph
                self.graph.add_node(bracket_id, 
                                  type='tax_rate',
                                  data=self.tax_rates[bracket_id])
        except Exception as e:
            print(f"Error processing tax rates: {str(e)}")

    def process_tax_guidelines(self, pdf):
        """Process tax guidelines from PDF"""
        chunks = self.extract_text_from_pdf(pdf)
        for i, chunk in enumerate(chunks):
            rule_id = f"rule_{i}"
            self.tax_rules[rule_id] = {
                'text': chunk,
                'source': 'guidelines'
            }
            
            # Add to graph and connect to related rates
            self.graph.add_node(rule_id, 
                              type='tax_rule',
                              data=self.tax_rules[rule_id])
            
            # Connect rules to relevant tax brackets
            for bracket_id in self.tax_rates:
                if self.is_related(chunk, self.tax_rates[bracket_id]):
                    self.graph.add_edge(rule_id, bracket_id)

    def process_tax_examples(self, ppt):
        """Process tax examples from PPT"""
        chunks = self.extract_text_from_ppt(ppt)
        for i, chunk in enumerate(chunks):
            example_id = f"example_{i}"
            self.tax_examples[example_id] = {
                'text': chunk,
                'source': 'examples'
            }
            
            # Add to graph
            self.graph.add_node(example_id,
                              type='example',
                              data=self.tax_examples[example_id])
            
            # Connect examples to relevant rules and rates
            for rule_id in self.tax_rules:
                if self.is_related(chunk, self.tax_rules[rule_id]):
                    self.graph.add_edge(example_id, rule_id)

    def build_search_index(self):
        """Build vector store and knowledge graph"""
        # Create embeddings for all text chunks
        embeddings = self.model.encode(self.documents)
        
        # Build FAISS index
        dimension = embeddings.shape[1]
        self.vector_store = faiss.IndexFlatL2(dimension)
        self.vector_store.add(embeddings.astype('float32'))
        
    def search(self, query):
        """Search for relevant tax information"""
        # Get query embedding
        query_vector = self.model.encode([query])
        
        # Semantic search
        D, I = self.vector_store.search(query_vector.astype('float32'), k=config.TOP_K_RESULTS)
        
        # Get relevant documents
        results = [self.documents[i] for i in I[0]]
        
        # Enhance with graph relationships
        enhanced_results = []
        for doc_id in I[0]:
            if doc_id in self.graph:
                neighbors = list(self.graph.neighbors(doc_id))
                related_info = []
                for neighbor in neighbors:
                    node_data = self.graph.nodes[neighbor].get('data', {})
                    if node_data:
                        related_info.append(node_data)
                
                enhanced_results.append({
                    'main_content': self.documents[doc_id],
                    'related_info': related_info
                })
        
        return {
            'direct_matches': results,
            'enhanced_results': enhanced_results
        }

    def extract_text_from_pdf(self, pdf_path):
        """Extract text from PDF and split into chunks"""
        chunks = []
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            current_chunk = ""
            
            for page in pdf_reader.pages:
                text = page.extract_text()
                
                # Split into sentences (rough approximation)
                sentences = text.replace('\n', ' ').split('. ')
                
                for sentence in sentences:
                    if len(current_chunk) + len(sentence) < config.MAX_CHUNK_SIZE:
                        current_chunk += sentence + '. '
                    else:
                        chunks.append(current_chunk)
                        current_chunk = sentence + '. '
                        
            if current_chunk:
                chunks.append(current_chunk)
                
        return chunks

    def extract_text_from_ppt(self, ppt_path):
        """Extract text from PowerPoint and split into chunks"""
        chunks = []
        prs = Presentation(ppt_path)
        
        for slide in prs.slides:
            slide_text = ""
            
            # Get text from shapes
            for shape in slide.shapes:
                if hasattr(shape, "text"):
                    slide_text += shape.text + " "
                    
            # Get text from notes
            if slide.has_notes_slide:
                notes = slide.notes_slide.notes_text_frame.text
                slide_text += " [Notes: " + notes + "]"
                
            if slide_text.strip():
                chunks.append(slide_text.strip())
                
        return chunks

    def is_related(self, text: str, tax_info: dict) -> bool:
        """Determine if a text chunk is related to tax information"""
        # Convert tax_info to text for comparison
        tax_text = f"Tax bracket {tax_info['range']} with rate {tax_info['rate']}"
        if tax_info['deductions']:
            tax_text += f" and deductions {tax_info['deductions']}"
        if tax_info['conditions']:
            tax_text += f" under conditions {tax_info['conditions']}"
        
        # Calculate similarity
        similarity = self.calculate_similarity(text, tax_text)
        return similarity > config.SIMILARITY_THRESHOLD

    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate cosine similarity between two texts"""
        embeddings = self.model.encode([text1, text2])
        similarity = np.dot(embeddings[0], embeddings[1]) / (
            np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
        )
        return similarity 