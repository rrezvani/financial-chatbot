from rest_framework import serializers
from .models import SearchQuery

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