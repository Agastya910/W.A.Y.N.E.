import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import hashlib
import time

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.repo_scanner import scan_repo
from tools.file_io import read_file


@dataclass
class CodeChunk:
    """A single code chunk with metadata."""
    file_path: str
    language: str
    start_line: int
    end_line: int
    content: str
    embedding: Optional[List[float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_path": self.file_path,
            "language": self.language,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "content": self.content,
        }


class CodeIndexer:
    """
    Builds and manages a semantic index of the codebase.
    Uses local embeddings (nomic-embed-text) and FAISS for retrieval.
    """
    
    def __init__(self, repo_path: str, index_dir: str = ".repopilot_index"):
        self.repo_path = repo_path
        self.index_dir = os.path.join(repo_path, index_dir)
        os.makedirs(self.index_dir, exist_ok=True)
        
        # Initialize embeddings model (runs locally via Ollama)
        try:
            self.embeddings = OllamaEmbeddings(
                model="nomic-embed-text",
                base_url="http://localhost:11434"
            )
            print("[INDEXER] ✅ Connected to Ollama embeddings service.")
        except Exception as e:
            print(f"[INDEXER] ⚠️  Could not connect to Ollama: {e}")
            self.embeddings = None
        
        self.vector_store = None
        self.chunk_metadata = {}
        self.files_index = {}
        self.load_or_build_index()
    
    def load_or_build_index(self):
        """Load existing index or build new one."""
        index_path = os.path.join(self.index_dir, "faiss_index")
        metadata_path = os.path.join(self.index_dir, "metadata.json")
        files_path = os.path.join(self.index_dir, "files.json")
        
        if os.path.exists(index_path) and os.path.exists(metadata_path):
            print("[INDEXER] Loading existing index...")
            try:
                self.vector_store = FAISS.load_local(
                    index_path, self.embeddings, allow_dangerous_deserialization=True
                )
                with open(metadata_path, "r") as f:
                    self.chunk_metadata = json.load(f)
                with open(files_path, "r") as f:
                    self.files_index = json.load(f)
                print(f"[INDEXER] ✅ Loaded {len(self.chunk_metadata)} chunks from {len(self.files_index)} files")
            except Exception as e:
                print(f"[INDEXER] Failed to load index: {e}. Rebuilding...")
                self.build_index()
        else:
            print("[INDEXER] Building new index...")
            self.build_index()
    
    def build_index(self):
        """Scan repo, chunk files, embed, and build FAISS index."""
        chunks = self._scan_and_chunk()
        
        if not chunks:
            print("[INDEXER] ⚠️  No files to index!")
            return
        
        # Prepare documents for FAISS
        texts = [chunk.content for chunk in chunks]
        metadatas = [chunk.to_dict() for chunk in chunks]
        
        print(f"[INDEXER] 🔄 Embedding {len(texts)} chunks with Ollama (this may take 1-5 minutes for large repos)...")
        if self.embeddings:
            try:
                # Show progress
                start_time = time.time()
                self.vector_store = FAISS.from_texts(
                    texts=texts,
                    embedding=self.embeddings,
                    metadatas=metadatas
                )
                elapsed = time.time() - start_time
                print(f"[INDEXER] ✅ Created vector store with {len(texts)} chunks ({elapsed:.1f}s)")
            except Exception as e:
                print(f"[INDEXER] ❌ Error creating embeddings: {e}")
                return
        else:
            print("[INDEXER] ❌ Embeddings unavailable, skipping FAISS index")
            return
        
        # Store metadata mapping
        self.chunk_metadata = {str(i): m for i, m in enumerate(metadatas)}
        
        # Build files index
        files_set = set()
        for metadata in metadatas:
            files_set.add((metadata["file_path"], metadata["language"]))
        self.files_index = {f[0]: f[1] for f in files_set}
        
        # Save to disk
        index_path = os.path.join(self.index_dir, "faiss_index")
        try:
            self.vector_store.save_local(index_path)
            
            with open(os.path.join(self.index_dir, "metadata.json"), "w") as f:
                json.dump(self.chunk_metadata, f, indent=2)
            
            with open(os.path.join(self.index_dir, "files.json"), "w") as f:
                json.dump(self.files_index, f, indent=2)
            
            print(f"[INDEXER] ✅ Index saved ({len(self.files_index)} files, {len(chunks)} chunks)")
        except Exception as e:
            print(f"[INDEXER] ❌ Error saving index: {e}")
    
    def _scan_and_chunk(self) -> List[CodeChunk]:
        """Scan repo and split files into chunks."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\nclass ", "\n\ndef ", "\n\n", "\n", " "]
        )
        
        chunks = []
        file_tree = scan_repo(self.repo_path)
        
        skipped_count = 0
        processed_count = 0
        
        def process_files(tree: Dict, current_path: str = ""):
            nonlocal skipped_count, processed_count
            
            for name, item in tree.items():
                if isinstance(item, dict) and item.get("type") == "file":
                    file_path = item.get("path", "")
                    if not file_path:
                        continue
                    
                    full_path = os.path.join(self.repo_path, file_path)
                    content = read_file(full_path)
                    
                    if content.startswith("Error") or not content.strip():
                        skipped_count += 1
                        continue
                    
                    language = item.get("language", "unknown")
                    
                    # For large repos (like Linux kernel), be more permissive
                    # Only skip files > 500KB (not 100KB)
                    if len(content) > 500000:
                        skipped_count += 1
                        continue
                    
                    # Skip binary files and non-code
                    if language in ["unknown", "Text"]:
                        skipped_count += 1
                        continue
                    
                    # Split into chunks
                    try:
                        split_chunks = splitter.split_text(content)
                        line_count = len(content.split('\n'))
                        
                        for i, chunk_content in enumerate(split_chunks):
                            start_line = max(0, int(i * line_count / len(split_chunks)))
                            end_line = min(line_count, int((i + 1) * line_count / len(split_chunks)))
                            
                            chunk = CodeChunk(
                                file_path=file_path,
                                language=language,
                                start_line=start_line,
                                end_line=end_line,
                                content=chunk_content
                            )
                            chunks.append(chunk)
                        
                        processed_count += 1
                    except Exception as e:
                        print(f"[INDEXER] Error chunking {file_path}: {e}")
                        skipped_count += 1
                
                elif isinstance(item, dict):
                    process_files(item, os.path.join(current_path, name))
        
        process_files(file_tree)
        print(f"[INDEXER] 📊 Scanned: {processed_count} files indexed, {skipped_count} skipped")
        return chunks
    
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve top-K relevant chunks using semantic search.
        Returns: List of {file_path, language, content, score}
        """
        if self.vector_store is None:
            print("[RETRIEVAL] ⚠️  Vector store not available")
            return []
        
        print(f"[RETRIEVAL] 🔍 Searching for: '{query}' (top {k})")
        try:
            results = self.vector_store.similarity_search_with_score(query, k=k)
        except Exception as e:
            print(f"[RETRIEVAL] ❌ Search error: {e}")
            return []
        
        retrieved = []
        for doc, score in results:
            retrieved.append({
                "file_path": doc.metadata.get("file_path"),
                "language": doc.metadata.get("language"),
                "start_line": doc.metadata.get("start_line"),
                "end_line": doc.metadata.get("end_line"),
                "content": doc.page_content,
                "score": float(score)
            })
        
        print(f"[RETRIEVAL] ✅ Found {len(retrieved)} relevant chunks")
        return retrieved
    
    def get_file_list(self) -> List[Dict[str, str]]:
        """Get all indexed files (local, no LLM)."""
        return [
            {"path": path, "language": lang}
            for path, lang in sorted(self.files_index.items())
        ]
    
    def get_file_count(self) -> int:
        """Get total number of indexed files."""
        return len(self.files_index)
    
    def get_architecture_summary(self) -> str:
        """Return a basic architecture summary from indexed files."""
        file_list = self.get_file_list()
        if not file_list:
            return "Repository is empty or not indexed."
        
        # Group by language
        by_lang = {}
        for f in file_list:
            lang = f['language']
            by_lang[lang] = by_lang.get(lang, 0) + 1
        
        summary = f"Repository Structure:\n"
        summary += f"Total Files: {len(file_list)}\n\n"
        summary += "Files by Language:\n"
        for lang, count in sorted(by_lang.items(), key=lambda x: -x[1])[:10]:  # Top 10 langs
            summary += f"  - {lang}: {count} files\n"
        
        return summary


if __name__ == "__main__":
    # Test indexing
    indexer = CodeIndexer(".")
    print(f"\nFiles indexed: {indexer.get_file_count()}")
    print(indexer.get_architecture_summary())
    
    results = indexer.search("error handling", k=3)
    print(f"\nSearch results for 'error handling':")
    for r in results:
        print(f"  {r['file_path']}: {r['content'][:100]}...")
