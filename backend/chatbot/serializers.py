from rest_framework import serializers
from .models import SearchQuery, GraphData

class SearchQuerySerializer(serializers.ModelSerializer):
    class Meta:
        model = SearchQuery
        fields = ['query', 'response', 'created_at']

class TaxRateSerializer(serializers.Serializer):
    type = serializers.CharField()
    income_range = serializers.CharField()
    rate = serializers.CharField()
    conditions = serializers.CharField()
    details = serializers.DictField(required=False)

class ChatResponseSerializer(serializers.Serializer):
    message = serializers.CharField()
    enhanced_results = TaxRateSerializer(many=True, required=False)
    status = serializers.CharField()

class GraphDataSerializer(serializers.ModelSerializer):
    class Meta:
        model = GraphData
        fields = ['figure_number', 'title', 'graph_type', 'data_points', 'interpretation']

class GraphResponseSerializer(serializers.Serializer):
    graph_reference = serializers.CharField()
    explanation = serializers.CharField()
    data_points = serializers.DictField()
    related_insights = serializers.ListField(child=serializers.CharField()) 