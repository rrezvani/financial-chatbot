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
    response = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return self.query 

class GraphData(models.Model):
    GRAPH_TYPES = [
        ('SD', 'Supply/Demand'),
        ('TR', 'Tax Revenue'),
        ('TS', 'Time Series'),
        ('CP', 'Comparison'),
        ('TB', 'Tax Bracket'),
    ]
    
    figure_number = models.CharField(max_length=10)  # e.g., "11.1"
    title = models.TextField()
    graph_type = models.CharField(max_length=2, choices=GRAPH_TYPES)
    data_points = models.JSONField()  # Store structured data from graph
    interpretation = models.TextField()  # Store key insights
    created_at = models.DateTimeField(auto_now_add=True)

class GraphRelationship(models.Model):
    source = models.ForeignKey(GraphData, on_delete=models.CASCADE, related_name='source_relationships')
    target = models.ForeignKey(GraphData, on_delete=models.CASCADE, related_name='target_relationships')
    relationship_type = models.CharField(max_length=50)  # e.g., "supports", "contradicts", "extends" 