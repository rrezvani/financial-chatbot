from django.contrib import admin
from .models import TaxDocument, SearchQuery

@admin.register(TaxDocument)
class TaxDocumentAdmin(admin.ModelAdmin):
    list_display = ('title', 'doc_type', 'uploaded_at', 'processed', 'chunk_count')
    list_filter = ('doc_type', 'processed')
    search_fields = ('title',)
    ordering = ('-uploaded_at',)

@admin.register(SearchQuery)
class SearchQueryAdmin(admin.ModelAdmin):
    list_display = ('query', 'timestamp', 'direct_matches', 'enhanced_matches')
    list_filter = ('timestamp',)
    search_fields = ('query', 'response')
    ordering = ('-timestamp',) 