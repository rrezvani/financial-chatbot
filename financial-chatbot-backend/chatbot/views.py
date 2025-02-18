from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser
from .data_processor import DataProcessor
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
    """Enhanced chat endpoint with semantic search"""
    user_message = request.data.get('message', '')
    
    if not processor.documents:
        return Response({
            'response': 'Please upload some datasets first.',
            'status': 'error'
        })
        
    # Perform semantic search
    search_results = processor.search(user_message)
    
    # Format response with relevant information
    response = {
        'response': f"Based on the available data, here are the relevant results:",
        'results': search_results,
        'status': 'success'
    }
    
    return Response(response) 