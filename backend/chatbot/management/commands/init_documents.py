from django.core.management.base import BaseCommand
from chatbot.models import TaxDocument
from chatbot import config
import os

class Command(BaseCommand):
    help = 'Initialize tax documents in the database'

    def handle(self, *args, **kwargs):
        for name, path in config.TAX_DOCUMENTS.items():
            if os.path.exists(path):
                doc_type = 'CSV' if path.endswith('.csv') else 'PDF' if path.endswith('.pdf') else 'PPT'
                doc, created = TaxDocument.objects.get_or_create(
                    title=name,
                    file_path=path,
                    doc_type=doc_type,
                    processed=True
                )
                if created:
                    self.stdout.write(f'Created document record for {name}')
                else:
                    self.stdout.write(f'Document {name} already exists') 