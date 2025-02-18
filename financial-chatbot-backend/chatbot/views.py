from rest_framework.decorators import api_view
from rest_framework.response import Response

@api_view(['GET'])
def test_endpoint(request):
    return Response({"message": "Financial Chatbot Backend is Running!"})

@api_view(['POST'])
def chat_endpoint(request):
    user_message = request.data.get('message', '')
    # For now, just echo the message back
    return Response({
        "response": f"You said: {user_message}",
        "status": "success"
    }) 