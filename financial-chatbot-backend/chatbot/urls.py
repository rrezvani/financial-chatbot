from django.urls import path
from . import views

urlpatterns = [
    path('test/', views.test_endpoint),
    path('chat/', views.chat_endpoint),
] 