import os
import json
import hashlib
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

import ollama as ollama_lib
from qdrant_client import QdrantClient, models
from qdrant_client.models import PointStruct, VectorParams, Distance, SparseVectorParams, Modifier
import uuid

from config import (
    QDRANT_URL, QDRANT_COLLECTION, SPARSE_VECTOR_NAME,
    OLLAMA_EMBED_MODEL, TOP_K_RETRIEVAL, TOP_K_RERANK
)
from tools.repo_scanner import scan_repo
from tools.file_io import read_file
from core.reranking import Reranker


@dataclass
class CodeChunk:
    file_path: str
    language: str
    start_line: int
    end_line: int
    content: str
    type: str = "code"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_path": self.file_path,
            "language": self.language,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "content": self.content,
            "type": self.type,
        }


class CodeIndexer:
    SKIP_EXTENSIONS = {
        '.pyc', '.pyo', '.so', '.o', '.dll', '.exe', '.bin',
        '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.ico',
        '.pdf', '.doc', '.docx', '.xls', '.xlsx',
        '.zip', '.tar', '.gz', '.rar', '.7z',
        '.min.js', '.bundle.js', '.map'
    }
    SKIP_DIRS = {
        '__pycache__', '.git', 'node_modules', 'venv', 'env',
        '.venv', '.env', 'dist', 'build', '.egg-info', 'qdrant_storage'
    }

    def __init__(self, repo_path: str):
        self.repo_path = repo_path
        self.ollama = ollama_lib.Client()
        self.qdrant = QdrantClient(url=QDRANT_URL, timeout=60)
        self.reranker = Reranker()
        self.collection = QDRANT_COLLECTION
        self._ensure_collection()
        self._files_cache: List[Dict] = []
        self.load_or_build_index()

    def _embed(self, text: str) -> List[float]:
        response = self.ollama.embeddings(model=OLLAMA_EMBED_MODEL, prompt=text)
        return response["embedding"]

    def _ensure_collection(self):
        try:
            self.qdrant.get_collection(self.collection)
        except Exception:
            print(f"[INDEXER] Creating Qdrant collection: {self.collection}")
            # nomic-embed-text produces 768-dim vectors
            self.qdrant.create_collection(
                collection_name=self.collection,
                vectors_config={"dense": VectorParams(size=768, distance=Distance.COSINE)},
                sparse_vectors_config={
                    SPARSE_VECTOR_NAME: SparseVectorParams(
                        index=models.SparseIndexParams(on_disk=False),
                        modifier=Modifier.IDF
                    )
                }
            )

    def _file_exists_in_db(self, file_path: str) -> bool:
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        result = self.qdrant.scroll(
            collection_name=self.collection,
            scroll_filter=Filter(must=[
                FieldCondition(key="file_path", match=MatchValue(value=file_path))
            ]),
            limit=1
        )
        return len(result[0]) > 0

    def load_or_build_index(self):
        count = self.qdrant.count(self.collection).count
        if count > 0:
            print(f"[INDEXER] ✅ Loaded existing index: {count} vectors in Qdrant")
            self._build_files_cache()
        else:
            print("[INDEXER] Building new index...")
            self.build_index()

    def build_index(self):
        chunks = self._scan_and_chunk()
        if not chunks:
            print("[INDEXER] ⚠️ No files to index.")
            return

        print(f"[INDEXER] Indexing {len(chunks)} chunks...")
        points = []
        for chunk in chunks:
            if self._file_exists_in_db(chunk.file_path):
                continue
            text = chunk.content
            dense = self._embed(text)
            points.append(PointStruct(
                id=str(uuid.uuid4()),
                vector={
                    "dense": dense,
                    SPARSE_VECTOR_NAME: models.Document(text=text, model="Qdrant/bm25")
                },
                payload=chunk.to_dict()
            ))

        if points:
            self.qdrant.upsert(collection_name=self.collection, points=points)
            print(f"[INDEXER] ✅ Indexed {len(points)} new chunks")

        self._build_files_cache()

    def _build_files_cache(self):
        """Build a local cache of indexed files for metadata queries."""
        seen = set()
        result, _ = self.qdrant.scroll(
            collection_name=self.collection,
            limit=10000,
            with_payload=True
        )
        for point in result:
            fp = point.payload.get("file_path", "")
            lang = point.payload.get("language", "unknown")
            if fp and fp not in seen:
                seen.add(fp)
                self._files_cache.append({"path": fp, "language": lang})

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Hybrid search (dense + sparse/BM25) with reranking."""
        query_dense = self._embed(query)

        response = self.qdrant.query_points(
            collection_name=self.collection,
            prefetch=[
                models.Prefetch(query=query_dense, using="dense", limit=TOP_K_RETRIEVAL),
                models.Prefetch(
                    query=models.Document(text=query, model="Qdrant/bm25"),
                    using=SPARSE_VECTOR_NAME,
                    limit=TOP_K_RETRIEVAL
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=TOP_K_RETRIEVAL,
            with_payload=True
        )

        candidates = [
            {**p.payload, "score": p.score if hasattr(p, "score") else 0.0}
            for p in response.points
        ]

        reranked = self.reranker.rerank(query, candidates, top_k=k)
        return reranked

    def _scan_and_chunk(self) -> List[CodeChunk]:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)

        files = scan_repo(self.repo_path)
        chunks = []

        def walk_tree(tree, base=""):
            for name, value in tree.items():
                if isinstance(value, dict):
                    if value.get("type") == "file":
                        rel_path = value.get("path", os.path.join(base, name))
                        lang = value.get("language", "unknown")
                        if lang in ("unknown", "Text"):
                            continue
                        ext = os.path.splitext(rel_path)[1]
                        if ext in self.SKIP_EXTENSIONS:
                            continue
                        skip = False
                        for sd in self.SKIP_DIRS:
                            if sd in rel_path:
                                skip = True
                                break
                        if skip:
                            continue
                        full_path = os.path.join(self.repo_path, rel_path)
                        try:
                            if os.path.getsize(full_path) > 200_000:
                                continue
                        except:
                            continue
                        content = read_file(full_path)
                        if not content or content.startswith("Error"):
                            continue
                        text_chunks = splitter.split_text(content)
                        lines = content.split("\n")
                        lines_per_chunk = max(1, len(lines) // max(len(text_chunks), 1))
                        for i, chunk_text in enumerate(text_chunks):
                            start = i * lines_per_chunk
                            end = start + lines_per_chunk
                            chunks.append(CodeChunk(
                                file_path=rel_path,
                                language=lang,
                                start_line=start,
                                end_line=end,
                                content=chunk_text
                            ))
                    else:
                        walk_tree(value, os.path.join(base, name))

        walk_tree(files)
        return chunks

    # ─── Public API (unchanged signatures) ────────────────────────────────

    def get_file_list(self) -> List[Dict]:
        return self._files_cache

    def get_file_count(self) -> int:
        return len(self._files_cache)

    def get_architecture_summary(self) -> str:
        by_lang: Dict[str, List[str]] = {}
        for f in self._files_cache:
            lang = f.get("language", "unknown")
            by_lang.setdefault(lang, []).append(f["path"])
        lines = [f"Repository has {len(self._files_cache)} indexed files:\n"]
        for lang, paths in sorted(by_lang.items()):
            lines.append(f"  {lang} ({len(paths)} files):")
            for p in paths[:5]:
                lines.append(f"    - {p}")
            if len(paths) > 5:
                lines.append(f"    ... and {len(paths) - 5} more")
        return "\n".join(lines)
