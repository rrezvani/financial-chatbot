from django.contrib import admin
from .models import TaxDocument, SearchQuery

@admin.register(TaxDocument)
class TaxDocumentAdmin(admin.ModelAdmin):
    list_display = ('title', 'doc_type', 'uploaded_at', 'processed')
    list_filter = ('doc_type', 'processed')
    search_fields = ('title',)
    ordering = ('-uploaded_at',)

@admin.register(SearchQuery)
class SearchQueryAdmin(admin.ModelAdmin):
    list_display = ('query', 'created_at')
    list_filter = ('created_at',)
    search_fields = ('query', 'response')
    ordering = ('-created_at',) 