from typing import List, Dict, Any
import json
import os
import subprocess

from tools import repo_scanner, code_search, file_io, diff_writer
from tools.github_helper import clone_github_repo
from core.indexer_ import CodeIndexer
from agent.edit_engine import EditEngine
from llm.local_llm_client import LocalLLMClient
from core.ingestion import IngestionPipeline


class Executor:
    """
    Executor: runs tool calls from the planner.
    """
    
    def __init__(self, repo_path: str = "."):
        self.repo_path = repo_path
        self.edit_engine = EditEngine(repo_path)
        self._pending_edit = None  # Store pending edit for confirmation
        self._edit_history = []  # stack of (file_path, original_content, instruction)
        
        self.tools = {
            "scan_repo": repo_scanner.scan_repo,
            "search_code": code_search.search_code,
            "read_file": file_io.read_file,
            "write_diff": diff_writer.write_diff,
            
            "github_clone": self._github_clone_tool,
            "github_analyze": self._github_analyze_tool,
            "report": self._report_tool,
            "llm_analysis": self._llm_analysis_tool,
            
            # Edit tools
            "edit_file": self._edit_file_tool,
            "apply_edit": self._apply_edit_tool,
            "undo": self.undo_last_edit,
            "fix_file": self._fix_file_tool,
            "index_documents": self._index_documents_tool,
        }
    
    def _github_clone_tool(self, repo_url: str, dest_path: str, timeout: int = 120) -> Dict[str, Any]:
        """Clone a GitHub repo."""
        return clone_github_repo(repo_url, dest_path, timeout)
    
    def _github_analyze_tool(self, repo_path: str, query: str) -> str:
        """
        Analyze a cloned GitHub repo by re-indexing and answering the query.
        """
        if not os.path.exists(repo_path):
            return f"❌ Repository path does not exist: {repo_path}"
        
        try:
            print(f"[ANALYZER] Indexing cloned repository at {repo_path}...")
            indexer = CodeIndexer(repo_path)
            
            file_count = indexer.get_file_count()
            print(f"[ANALYZER] Indexed {file_count} files")
            
            # Search in the cloned repo
            print(f"[ANALYZER] Searching for: '{query}'")
            results = indexer.search(query, k=5)
            
            if not results:
                summary = indexer.get_architecture_summary()
                return f"Repository indexed ({file_count} files).\n\nArchitecture:\n{summary}\n\nNo specific matches found for '{query}', but repository is ready for analysis."
            
            # Format results
            analysis = f"Repository indexed ({file_count} files).\n\nRelevant findings for '{query}':\n\n"
            for i, r in enumerate(results, 1):
                analysis += f"{i}. **{r['file_path']}** ({r['language']})\n"
                analysis += f"   Lines {r['start_line']}-{r['end_line']}\n"
                analysis += f"   Relevance: {r['score']:.2f}\n"
                analysis += f"   Content:\n   ```\n{r['content'][:300]}\n   ```\n\n"
            
            return analysis
        
        except Exception as e:
            return f"❌ Error analyzing repository: {str(e)}"
    
    def _report_tool(self, message: str) -> str:
        """Simple message reporting tool."""
        return message
    
    def _llm_analysis_tool(self, query: str, analysis: str) -> str:
        """Store LLM analysis result."""
        return analysis
    
    def _edit_file_tool(self, file_path: str, instruction: str, target: str = None) -> Dict[str, Any]:
        """
        Generate an edit preview for a file.
        Returns diff for user review.
        """
        result = self.edit_engine.preview_edit(file_path, instruction, target)
        
        if result["success"]:
            # Store for potential apply
            self._pending_edit = {
                "file_path": file_path,
                "modified": result["modified"],
                "instruction": instruction,
                "summary": result["summary"]
            }
        
        return result
    
    def _apply_edit_tool(self, confirm: bool = True) -> Dict[str, Any]:
        """
        Apply the pending edit after user confirmation.
        """
        if not self._pending_edit:
            return {"success": False, "message": "No pending edit to apply"}
        
        if not confirm:
            self._pending_edit = None
            return {"success": False, "message": "Edit cancelled by user"}
            
        # BEFORE applying, save to undo history
        try:
            abs_path = os.path.join(self.repo_path, self._pending_edit["file_path"])
            original_content = ""
            if os.path.exists(abs_path):
                with open(abs_path, "r", encoding="utf-8") as f:
                    original_content = f.read()
            
            self._edit_history.append({
                "file_path": self._pending_edit["file_path"],
                "original": original_content,
                "instruction": self._pending_edit["instruction"]
            })
        except Exception as e:
            print(f"[EXECUTOR] Warning: Could not save to undo history: {e}")
        
        result = self.edit_engine.apply_edit(
            self._pending_edit["file_path"],
            self._pending_edit["modified"],
            self._pending_edit["instruction"],
            self._pending_edit["summary"]
        )
        
        self._pending_edit = None
        return result
        
    def undo_last_edit(self) -> Dict[str, Any]:
        """Undo the last file edit."""
        if not self._edit_history:
            return {"success": False, "message": "Nothing to undo"}
        
        entry = self._edit_history.pop()
        abs_path = os.path.join(self.repo_path, entry["file_path"])
        
        try:
            with open(abs_path, "w", encoding="utf-8") as f:
                f.write(entry["original"])
            return {"success": True, "message": f"↩️ Reverted {entry['file_path']}"}
        except Exception as e:
            return {"success": False, "message": f"Error during undo: {e}"}
    
    def _fix_file_tool(self, file_path: str, max_cycles: int = 5) -> Dict[str, Any]:
        """
        Self-healing loop: run a Python file, detect errors, auto-fix, repeat.
        Uses the LLM with JSON-mode output for structured patch generation.
        Integrates with the existing undo stack.
        """
        llm = LocalLLMClient()
        abs_path = os.path.join(self.repo_path, file_path)

        if not os.path.exists(abs_path):
            return {"success": False, "message": f"File not found: {file_path}"}

        print(f"\n[FIX] Starting self-healing loop for: {file_path}")

        for cycle in range(max_cycles):
            print(f"\n[FIX] Cycle {cycle + 1}/{max_cycles}")

            # Run the file
            try:
                result = subprocess.run(
                    ["python", abs_path],
                    capture_output=True, text=True, timeout=10
                )
            except subprocess.TimeoutExpired:
                return {"success": False, "message": "Execution timed out after 10s"}

            if result.returncode == 0:
                print(f"[FIX] ✅ File runs successfully after {cycle} fix(es).")
                return {
                    "success": True,
                    "message": f"✅ {file_path} runs successfully after {cycle} fix(es).",
                    "cycles": cycle
                }

            error_output = result.stderr.strip()
            print(f"[FIX] Error detected: {error_output[:120]}")

            # Read current file content
            try:
                with open(abs_path, "r", encoding="utf-8") as f:
                    current_code = f.read()
            except Exception as e:
                return {"success": False, "message": f"Cannot read file: {e}"}

            # Save to undo stack before applying any fix
            self._edit_history.append({
                "file_path": file_path,
                "original": current_code,
                "instruction": f"auto-fix cycle {cycle + 1}"
            })

            # Ask LLM for fix (JSON mode for reliable parsing)
            prompt_messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a Python debugging assistant. "
                        "You will receive a broken Python file and its error. "
                        "Return ONLY valid JSON with a single key 'fixed_code' "
                        "containing the complete corrected file as a string."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"FILE: {file_path}\n\n"
                        f"ERROR:\n{error_output}\n\n"
                        f"CODE:\n{current_code}"
                    )
                }
            ]

            raw = llm.chat(prompt_messages, temperature=0.2, json_mode=True)

            try:
                import json
                parsed = json.loads(raw)
                fixed_code = parsed.get("fixed_code", "").strip()
                if not fixed_code:
                    raise ValueError("Empty fixed_code in response")
            except Exception as e:
                print(f"[FIX] ⚠️ Could not parse LLM response: {e}")
                continue

            # Write the fix
            try:
                with open(abs_path, "w", encoding="utf-8") as f:
                    f.write(fixed_code)
                print(f"[FIX] Patch applied (cycle {cycle + 1})")
            except Exception as e:
                return {"success": False, "message": f"Could not write fix: {e}"}

        return {
            "success": False,
            "message": f"❌ Could not fix {file_path} after {max_cycles} cycles. Use 'undo' to revert."
        }
    
    def has_pending_edit(self) -> bool:
        """Check if there's a pending edit awaiting confirmation."""
        return self._pending_edit is not None
    
    def get_pending_edit_info(self) -> Dict:
        """Get info about pending edit."""
        return self._pending_edit
    
    def execute_plan(self, plan: List[Dict[str, Any]]) -> List[Any]:
        """Execute a plan of tool calls."""
        results = []
        
        for tool_call in plan:
            tool_name = tool_call.get("tool_name")
            args = tool_call.get("args", {})
            
            if tool_name not in self.tools:
                results.append({
                    "tool": tool_name,
                    "error": f"Unknown tool: {tool_name}"
                })
                continue
            
            print(f"[EXECUTOR] → {tool_name}")
            
            try:
                result = self.tools[tool_name](**args)
                results.append({"tool": tool_name, "result": result})
            except Exception as e:
                results.append({
                    "tool": tool_name,
                    "error": str(e)
                })
        
        return results

    def _index_documents_tool(self, folder_path: str) -> str:
        """
        Index all supported documents (PDF, DOCX, PPTX, TXT) in a folder
        into WAYNE's Qdrant vector store.
        """
        import os
        import uuid
        from qdrant_client.models import PointStruct
        from core.indexer_ import CodeIndexer
        from config import QDRANT_COLLECTION, SPARSE_VECTOR_NAME
        import ollama as ollama_lib

        abs_folder = os.path.abspath(folder_path)
        if not os.path.exists(abs_folder):
            return f"❌ Folder not found: {abs_folder}"

        ingestion = IngestionPipeline()
        indexer = CodeIndexer(self.repo_path)  # reuse existing qdrant client

        supported_exts = {".pdf", ".docx", ".pptx", ".txt"}
        files = [
            f for f in os.listdir(abs_folder)
            if os.path.splitext(f)[1].lower() in supported_exts
        ]

        if not files:
            return f"⚠️ No supported documents found in {abs_folder}"

        total_chunks = 0
        for file in files:
            file_path = os.path.join(abs_folder, file)

            # Skip already-indexed files
            if indexer._file_exists_in_db(file_path):
                print(f"[INDEX_DOCS] ⏩ Skipping {file} (already indexed)")
                continue

            chunks = ingestion.process_file(file_path)
            if not chunks:
                continue

            # Embed and upsert into the same Qdrant collection as code
            ollama_client = ollama_lib.Client()
            points = []
            for chunk in chunks:
                dense = ollama_client.embeddings(
                    model="nomic-embed-text", prompt=chunk["content"]
                )["embedding"]
                from qdrant_client.models import models
                points.append(PointStruct(
                    id=str(uuid.uuid4()),
                    vector={
                        "dense": dense,
                        SPARSE_VECTOR_NAME: models.Document(
                            text=chunk["content"], model="Qdrant/bm25"
                        )
                    },
                    payload=chunk
                ))

            if points:
                indexer.qdrant.upsert(collection_name=QDRANT_COLLECTION, points=points)
                total_chunks += len(points)

        return f"✅ Indexed {total_chunks} chunks from {len(files)} document(s) in {abs_folder}"
