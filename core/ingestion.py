from pathlib import Path
from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
from docling_core.types.doc import DoclingDocument


class IngestionPipeline:
    """
    Multi-format document ingestion using Docling.
    Handles PDF, DOCX, PPTX, and TXT files.
    Produces chunks compatible with WAYNE's vector store.
    """

    def __init__(self):
        self.converter = DocumentConverter()
        self.chunker = HybridChunker(tokenizer="sentence-transformers/all-MiniLM-L6-v2")

    def process_file(self, file_path: str) -> list:
        """
        Parse a file and return a list of chunk dicts.
        Each chunk has: text, source, page, chunk_id, type="document"
        """
        path = Path(file_path)
        print(f"[INGESTION] Processing: {path.name}...")

        try:
            if path.suffix.lower() == ".txt":
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
                doc = DoclingDocument(name=path.stem)
                doc.add_text(label="text", text=content)
            else:
                result = self.converter.convert(str(file_path))
                doc = result.document

            raw_chunks = list(self.chunker.chunk(doc))
            processed = []

            for i, chunk in enumerate(raw_chunks):
                heading = chunk.meta.headings[0] if chunk.meta.headings else "General"
                processed.append({
                    "content": chunk.text,
                    "file_path": str(file_path),
                    "language": "document",
                    "source": str(file_path),
                    "page": heading,
                    "chunk_id": i,
                    "type": "document"
                })

            print(f"[INGESTION] ✅ {len(processed)} chunks from {path.name}")
            return processed

        except Exception as e:
            print(f"[INGESTION] ❌ Error processing {path.name}: {e}")
            return []

    def is_supported(self, file_path: str) -> bool:
        """Check if this file type can be ingested."""
        return Path(file_path).suffix.lower() in {".pdf", ".docx", ".pptx", ".txt"}
