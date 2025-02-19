from django.urls import path
from . import views

urlpatterns = [
    path('chat/', views.chat_endpoint, name='chat'),
    path('test/', views.test_endpoint, name='test'),
    path('upload/', views.upload_dataset, name='upload'),
] 