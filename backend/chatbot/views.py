from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser
from .data_processor import DataProcessor
from .models import TaxDocument, SearchQuery
from . import config  # Import config directly
import os
from .serializers import SearchQuerySerializer, ChatResponseSerializer

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
    print(f"Received chat request: {request.data}")
    
    message = request.data.get('message')
    if not message:
        return Response({
            'message': 'No message provided',
            'enhanced_results': [],
            'status': 'error'
        }, status=400)

    try:
        # Use the search method we implemented in DataProcessor
        results = processor.search(message)
        
        if not results:
            response_data = {
                'message': 'I could not find specific information to answer your question.',
                'enhanced_results': [],
                'status': 'success'
            }
            serializer = ChatResponseSerializer(data=response_data)
            serializer.is_valid(raise_exception=True)
            return Response(serializer.data)

        # Store the query and results
        query_data = {
            'query': message,
            'response': str(results)
        }
        query_serializer = SearchQuerySerializer(data=query_data)
        query_serializer.is_valid(raise_exception=True)
        query_serializer.save()
        
        # Format the response
        response_data = {
            'message': '\n'.join(results.get('direct_matches', [])) if results.get('direct_matches') else 'No relevant information found',
            'enhanced_results': results.get('enhanced_results', []),
            'status': 'success'
        }
        
        response_serializer = ChatResponseSerializer(data=response_data)
        response_serializer.is_valid(raise_exception=True)
        
        print(f"Sending response: {response_serializer.data}")
        return Response(response_serializer.data)
        
    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")
        error_response = {
            'message': 'Sorry, I encountered an error. Please try again.',
            'enhanced_results': [],
            'status': 'error'
        }
        serializer = ChatResponseSerializer(data=error_response)
        serializer.is_valid(raise_exception=True)
        return Response(serializer.data, status=500)

# Remove the document initialization code from the bottom of views.py
# We'll add it to a management command later 