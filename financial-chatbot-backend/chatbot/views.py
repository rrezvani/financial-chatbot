from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser
from .data_processor import DataProcessor
from .models import TaxDocument, SearchQuery
from . import config  # Import config directly
import os

# Initialize the data processor
processor = DataProcessor()

@api_view(['GET'])
def test_endpoint(request):
    return Response({"message": "Financial Chatbot Backend is Running!"})

@api_view(['POST'])
def upload_dataset(request):
    """Endpoint to upload and process datasets"""
    if 'file' not in request.FILES:
        return Response({'error': 'No file provided'}, status=400)
        
    file = request.FILES['file']
    file_type = request.data.get('type', '').lower()
    
    # Save file temporarily
    file_path = os.path.join('data', file.name)
    with open(file_path, 'wb+') as destination:
        for chunk in file.chunks():
            destination.write(chunk)
            
    # Process based on file type
    try:
        if file_type == 'csv':
            processor.process_csv(file_path)
        elif file_type == 'pdf':
            processor.process_pdf(file_path)
        elif file_type == 'ppt':
            processor.process_ppt(file_path)
        else:
            return Response({'error': 'Unsupported file type'}, status=400)
            
        # Build search index after processing
        processor.build_search_index()
        
        return Response({
            'message': f'Successfully processed {file_type} file',
            'document_count': len(processor.documents)
        })
        
    except Exception as e:
        return Response({'error': str(e)}, status=500)
    finally:
        # Clean up temporary file
        if os.path.exists(file_path):
            os.remove(file_path)

@api_view(['POST'])
def chat_endpoint(request):
    """Handle chat messages and return relevant tax information"""
    print(f"Received chat request: {request.data}")  # Debug log
    
    message = request.data.get('message')
    if not message:
        return Response({'error': 'No message provided'}, status=400)

    try:
        # Use the search method we implemented in DataProcessor
        results = processor.search(message)
        
        # Format the response in a more readable way
        response_text = ""
        
        # Add direct matches if any
        if results['direct_matches']:
            response_text += results['direct_matches'][0] + "\n\n"
            
        # Add tax rate information if any
        if results['enhanced_results']:
            response_text += "Applicable tax rates:\n"
            for rate_info in results['enhanced_results'][:3]:  # Limit to top 3 matches
                response_text += f"• Income Range: ${rate_info['income_range']}\n"
                response_text += f"  Tax Rate: {float(rate_info['rate'])*100:.2f}%\n"
                response_text += f"  {rate_info['conditions']}\n\n"
                
        if not response_text:
            response_text = "I couldn't find specific information for that query. Please try rephrasing your question."
            
        # Store the query and results
        query = SearchQuery.objects.create(
            query=message,
            response=response_text,
            direct_matches=len(results.get('direct_matches', [])),
            enhanced_matches=len(results.get('enhanced_results', []))
        )
        
        response = {
            'message': response_text,
            'status': 'success'
        }
        print(f"Sending response: {response}")  # Debug log
        return Response(response)
        
    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")  # Debug log
        return Response({
            'message': 'Sorry, I encountered an error. Please try again.',
            'error': str(e),
            'status': 'error'
        }, status=500)

# Remove the document initialization code from the bottom of views.py
# We'll add it to a management command later 