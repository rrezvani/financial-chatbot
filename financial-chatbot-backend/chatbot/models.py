from django.db import models
from django.utils import timezone

class TaxDocument(models.Model):
    DOCUMENT_TYPES = [
        ('CSV', 'Tax Data CSV'),
        ('PDF', 'Tax Code PDF'),
        ('PPT', 'Tax Presentation'),
    ]
    
    title = models.CharField(max_length=200)
    file_path = models.CharField(max_length=500)
    doc_type = models.CharField(max_length=3, choices=DOCUMENT_TYPES)
    uploaded_at = models.DateTimeField(default=timezone.now)
    processed = models.BooleanField(default=False)
    chunk_count = models.IntegerField(default=0)
    
    def __str__(self):
        return f"{self.title} ({self.doc_type})"

class SearchQuery(models.Model):
    query = models.TextField()
    timestamp = models.DateTimeField(default=timezone.now)
    response = models.TextField()
    direct_matches = models.IntegerField(default=0)
    enhanced_matches = models.IntegerField(default=0)
    
    def __str__(self):
        return f"Query: {self.query[:50]}..." 