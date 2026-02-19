from typing import List, Dict, Any
import json
import re
import os

from llm.local_llm_client import LocalLLMClient
from core.indexer_ import CodeIndexer
from core.query_router import QueryRouter, QueryType
from tools.github_helper import clone_github_repo, get_repo_url_from_query
from agent.chat_history import ChatHistory


class Planner:
    """
    Planner agent: decomposes user queries into tool calls.
    Now retrieval-aware: only sends relevant context to LLM.
    Supports GitHub repo analysis.
    """
    
    def __init__(self, repo_path: str):
        self.repo_path = repo_path
        self.llm_client = LocalLLMClient()
        self.indexer = CodeIndexer(repo_path)
        self.router = QueryRouter()
        self.history = ChatHistory(repo_path)
    
    def create_plan(self, user_query: str) -> List[Dict[str, Any]]:
        """
        Create a plan for the user query.
        
        Flow:
        1. Check if user wants to analyze a GitHub repo
        2. Classify query type
        3. If metadata: answer locally
        4. If edit: generate edit plan
        5. If search/reasoning: retrieve relevant chunks
        6. Ask LLM to plan given context
        """
        
        # Check for GitHub repo in query
        github_url = get_repo_url_from_query(user_query)
        if github_url:
            return self._plan_github_analysis(user_query, github_url)
        
        query_type = self.router.classify(user_query)
        print(f"[PLANNER] Query type: {query_type.value}")
        
        # METADATA: Handle locally
        if query_type == QueryType.METADATA:
            return self._handle_metadata_query(user_query)
        
        # EDIT: Generate edit plan
        if query_type == QueryType.EDIT:
            return self._handle_edit_query(user_query)
            
        # UNDO: Revert last edit
        if query_type == QueryType.UNDO:
            return self._handle_undo_query(user_query)
        
        # FIX: Self-healing loop
        if query_type == QueryType.FIX:
            return self._handle_fix_query(user_query)

        # INDEX_DOCS: Document ingestion
        if query_type == QueryType.INDEX_DOCS:
            return self._handle_index_docs_query(user_query)
        
        # TOOL: Direct execution
        if query_type == QueryType.TOOL_CALL:
            return self._extract_tool_calls(user_query)
        
        # SEARCH / REASONING: Retrieve + LLM
        retrieved_context = self._retrieve_context(user_query)
        return self._plan_with_llm(user_query, retrieved_context, query_type)
    
    def _plan_github_analysis(self, query: str, github_url: str) -> List[Dict[str, Any]]:
        """Plan for analyzing a GitHub repository."""
        print(f"[PLANNER] GitHub URL detected: {github_url}")
        print("[PLANNER] Planning: Clone → Index → Analyze")
        
        return [
            {
                "tool_name": "github_clone",
                "args": {
                    "repo_url": github_url,
                    "dest_path": "./analyzed_repo",
                    "timeout": 120
                }
            },
            {
                "tool_name": "github_analyze",
                "args": {
                    "repo_path": "./analyzed_repo",
                    "query": query
                }
            }
        ]
    
    def _handle_metadata_query(self, query: str) -> List[Dict[str, Any]]:
        """Answer metadata queries without LLM."""
        print("[PLANNER] Answering metadata query locally...")
        
        files = self.indexer.get_file_list()
        
        if "how many" in query.lower() and "files" in query.lower():
            return [{
                "tool_name": "report",
                "args": {"message": f"This repo has {len(files)} indexed files."}
            }]
        
        if "list" in query.lower():
            file_list = "\n".join([f"  - {f['path']} ({f['language']})" for f in files[:20]])
            if len(files) > 20:
                file_list += f"\n  ... and {len(files) - 20} more files"
            return [{
                "tool_name": "report",
                "args": {"message": f"Files in repo ({len(files)} total):\n{file_list}"}
            }]
        
        if "architecture" in query.lower() or "structure" in query.lower():
            return [{
                "tool_name": "report",
                "args": {"message": self.indexer.get_architecture_summary()}
            }]
        
        # Fallback
        return [{
            "tool_name": "report",
            "args": {"message": f"Found {len(files)} files in repository."}
        }]
    
    def _handle_edit_query(self, query: str) -> List[Dict[str, Any]]:
        """Handle edit queries by finding target file and generating edit plan."""
        import re
        
        # Try to find explicit file reference
        file_pattern = r'[\w\-_/]+\.(?:py|js|ts|go|rs|java|cpp|c|jsx|tsx)'
        file_matches = re.findall(file_pattern, query)
        
        file_path = None
        target = None
        
        if file_matches:
            file_path = file_matches[0]
        else:
            # Use retrieval to find relevant file
            results = self.indexer.search(query, k=1)
            if results:
                file_path = results[0]['file_path']
        
        # Extract function/class target
        func_match = re.search(r'(?:function|def|method)\s+(\w+)', query.lower())
        class_match = re.search(r'class\s+(\w+)', query.lower())
        if func_match:
            target = func_match.group(1)
        elif class_match:
            target = class_match.group(1)
        
        if not file_path:
            return [{
                "tool_name": "report",
                "args": {"message": "❌ Could not identify target file. Please specify the file name."}
            }]
        
        print(f"[PLANNER] Edit target: {file_path}")
        
        return [{
            "tool_name": "edit_file",
            "args": {
                "file_path": file_path,
                "instruction": query,
                "target": target
            }
        }]
    
    def _handle_undo_query(self, query: str) -> List[Dict[str, Any]]:
        """Handle undo requests."""
        print("[PLANNER] Planning undo operation...")
        return [{
            "tool_name": "undo",
            "args": {}
        }]
    
    def _handle_fix_query(self, query: str) -> List[Dict[str, Any]]:
        """Handle fix/self-heal queries."""
        import re
        file_pattern = r'[\w\-_/]+\.(?:py|js|ts|go|rs|java|cpp|c)'
        file_matches = re.findall(file_pattern, query)
        file_path = file_matches[0] if file_matches else None

        if not file_path:
            results = self.indexer.search(query, k=1)
            if results:
                file_path = results[0]['file_path']

        if not file_path:
            return [{
                "tool_name": "report",
                "args": {"message": "❌ Could not identify target file to fix. Please specify the filename."}
            }]

        print(f"[PLANNER] Fix target: {file_path}")
        return [{"tool_name": "fix_file", "args": {"file_path": file_path}}]

    def _handle_index_docs_query(self, query: str) -> List[Dict[str, Any]]:
        """Handle document folder indexing requests."""
        import re
        # Try to extract a path from the query
        path_match = re.search(r'["\']([^"\']+)["\']|(\S+/\S+|\S+\\\S+)', query)
        folder = None
        if path_match:
            folder = path_match.group(1) or path_match.group(2)

        return [{
            "tool_name": "index_documents",
            "args": {"folder_path": folder or "."}
        }]

    def _retrieve_context(self, query: str) -> List[Dict[str, Any]]:
        """Retrieve top-3 relevant code chunks (reduced from 5 to make room for history)."""
        return self.indexer.search(query, k=3)
    
    def _plan_with_llm(
        self,
        user_query: str,
        retrieved_context: List[Dict[str, Any]],
        query_type: QueryType
    ) -> List[Dict[str, Any]]:
        """Use LLM to create plan given retrieved context."""
        
        context_str = "\n".join([
            f"File: {c['file_path']}\n```{c['language']}\n{c['content'][:500]}\n```"
            for c in retrieved_context
        ])
        
        history_context = self.history.get_context_block()
        
        prompt = f"""
You are a senior software engineer analyzing a codebase.

{f"[Conversation history]{chr(10)}{history_context}" if history_context else ""}

User Query: "{user_query}"

Relevant Code Context (top 3 chunks):
{context_str if context_str else "[No relevant code found]"}

Based on the query and context, answer the user's query, if not clear what the query says, provide a technical analysis. Be specific and reference file names.
Keep your response clear, concise, and actionable.
"""
        
        print("[PLANNER] Generating analysis...")
        full_response = ""
        for chunk in self.llm_client.generate_text_stream(prompt):
            print(chunk, end="", flush=True)
            full_response += chunk
        print()  # New line after streaming
        
        return [{
            "tool_name": "llm_analysis",
            "args": {
                "query": user_query,
                "analysis": full_response
            }
        }]
    
    def _extract_tool_calls(self, query: str) -> List[Dict[str, Any]]:
        """Extract git_clone or direct tool calls."""
        github_url = get_repo_url_from_query(query)
        if github_url:
            return [{
                "tool_name": "github_clone",
                "args": {
                    "repo_url": github_url,
                    "dest_path": "./analyzed_repo",
                    "timeout": 120
                }
            }]
        return []
