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
        self.tax_examples = {}
        
        # Process our tax documents
        self.load_tax_documents()

    def load_tax_documents(self):
        """Load and process all tax documents once"""
        try:
            # Load CSV data
            print(f"Attempting to load CSV from: {config.TAX_DOCUMENTS['tax_data']}")
            tax_rates = pd.read_csv(config.TAX_DOCUMENTS['tax_data'])
            if tax_rates is not None:
                print("CSV loaded successfully, processing tax rates...")
                self.process_tax_rates(tax_rates)
            
            # Load main tax code PDF
            try:
                print(f"Attempting to load main tax code from: {config.TAX_DOCUMENTS['tax_code']}")
                with open(config.TAX_DOCUMENTS['tax_code'], 'rb') as pdf_file:
                    pdf_reader = PyPDF2.PdfReader(pdf_file)
                    self.process_tax_guidelines(pdf_reader, "tax_code")
            except FileNotFoundError as e:
                print(f"Warning: Tax code PDF not found at {config.TAX_DOCUMENTS['tax_code']}")
                print(f"Error details: {str(e)}")
            
            # Load tax instructions PDF
            try:
                print(f"Attempting to load tax instructions from: {config.TAX_DOCUMENTS['tax_instructions']}")
                with open(config.TAX_DOCUMENTS['tax_instructions'], 'rb') as pdf_file:
                    pdf_reader = PyPDF2.PdfReader(pdf_file)
                    self.process_tax_guidelines(pdf_reader, "instructions")
            except FileNotFoundError as e:
                print(f"Warning: Tax instructions PDF not found at {config.TAX_DOCUMENTS['tax_instructions']}")
                print(f"Error details: {str(e)}")
            
            # Load PPT data
            ppt_path = config.TAX_DOCUMENTS['tax_presentation']
            print(f"Checking if PPT exists at: {ppt_path}")
            if os.path.exists(ppt_path):
                print("PPT file found, attempting to load...")
                try:
                    prs = pptx.Presentation(ppt_path)
                    self.process_tax_examples(prs)
                except Exception as e:
                    print(f"Error loading PPT: {str(e)}")
                    print(f"Error type: {type(e)}")
            else:
                print(f"PPT file not found at: {ppt_path}")
            
            print("Building search index...")
            self.build_search_index()
        except Exception as e:
            print(f"Error loading tax documents: {str(e)}")
            print(f"Error type: {type(e)}")
            import traceback
            print(f"Full traceback: {traceback.format_exc()}")

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

    def process_tax_guidelines(self, pdf_reader, source):
        """Process tax guidelines from PDF"""
        # Extract text directly from the PdfReader object
        chunks = []
        
        # Get total number of pages
        total_pages = len(pdf_reader.pages)
        print(f"PDF ({source}) has {total_pages} pages. Processing first 20 pages...")
        
        # Only process first 20 pages - most tax guidelines have important info up front
        for page_num in range(min(20, total_pages)):
            if page_num % 5 == 0:  # Progress update more frequently
                print(f"Processing page {page_num}...")
                
            try:
                text = pdf_reader.pages[page_num].extract_text()
                
                # More aggressive filtering
                if len(text.strip()) < 100:  # Skip very short pages
                    continue
                    
                # Only keep paragraphs that mention relevant tax terms
                relevant_terms = ['tax', 'income', 'deduction', 'rate', 'bracket']
                paragraphs = text.split('\n\n')
                relevant_paragraphs = [
                    p for p in paragraphs 
                    if any(term in p.lower() for term in relevant_terms)
                ]
                
                # Join relevant paragraphs and add to chunks
                if relevant_paragraphs:
                    chunks.append(' '.join(relevant_paragraphs))
                    
            except Exception as e:
                print(f"Error processing page {page_num}: {str(e)}")
                continue

        print(f"Processed {len(chunks)} relevant chunks from the PDF")
        print("Building relationships for relevant chunks only...")

        # Only process chunks that are highly relevant
        for i, chunk in enumerate(chunks):
            rule_id = f"rule_{source}_{i}"  # Add source to ID
            self.tax_rules[rule_id] = {
                'text': chunk,
                'source': source  # Use the provided source
            }
            self.documents.append(chunk)
            
            # Add to graph without checking relationships
            # We'll rely on vector search instead
            self.graph.add_node(rule_id, 
                              type='tax_rule',
                              data=self.tax_rules[rule_id])

        print("Finished processing PDF chunks")

    def process_tax_examples(self, ppt):
        """Process tax examples from PPT"""
        print("Starting to process PPT...")
        try:
            chunks = []
            print(f"Processing {len(ppt.slides)} slides...")
            
            for slide in ppt.slides:
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
            
            print(f"Extracted {len(chunks)} chunks from PPT")
            
            for i, chunk in enumerate(chunks):
                if i % 10 == 0:  # Progress update
                    print(f"Processing chunk {i}...")
                    
                example_id = f"example_{i}"
                self.tax_examples[example_id] = {
                    'text': chunk,
                    'source': 'examples'
                }
                self.documents.append(chunk)  # Add to documents for search
                
                # Add to graph
                self.graph.add_node(example_id,
                                  type='example',
                                  data=self.tax_examples[example_id])
                    
            print("Finished processing PPT")
            
        except Exception as e:
            print(f"Error processing PPT: {str(e)}")
            print(f"Error type: {type(e)}")
            import traceback
            print(f"Full traceback: {traceback.format_exc()}")

    def build_search_index(self):
        """Build vector store and knowledge graph"""
        try:
            print(f"Building search index for {len(self.documents)} documents...")
            
            if not self.documents:
                print("Warning: No documents to index")
                return
            
            # Process in batches to avoid memory issues
            batch_size = 32
            all_embeddings = []
            
            for i in range(0, len(self.documents), batch_size):
                batch = self.documents[i:i + batch_size]
                print(f"Processing batch {i//batch_size + 1}/{(len(self.documents) + batch_size - 1)//batch_size}")
                
                # Create embeddings for current batch
                batch_embeddings = self.model.encode(batch)
                all_embeddings.append(batch_embeddings)
                
            # Concatenate all embeddings
            print("Concatenating embeddings...")
            embeddings = np.vstack(all_embeddings)
            
            # Build FAISS index
            print("Building FAISS index...")
            dimension = embeddings.shape[1]
            self.vector_store = faiss.IndexFlatL2(dimension)
            self.vector_store.add(embeddings.astype('float32'))
            
            print("Search index built successfully!")
            
        except Exception as e:
            print(f"Error building search index: {str(e)}")
            print(f"Error type: {type(e)}")
            import traceback
            print(f"Full traceback: {traceback.format_exc()}")

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