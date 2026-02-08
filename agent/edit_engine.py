"""
Edit Engine - Grounded file editing with retrieval

No hallucination. No long explanations.
Uses existing retrieval to fetch real code, then generates minimal edits.
"""

import os
import re
from typing import Dict, Optional, Tuple

from tools.file_retriever import FileRetriever
from tools.file_io import read_file, write_file
from tools.diff_writer import create_diff
from llm.local_llm_client import LocalLLMClient
from core.indexer_ import CodeIndexer
from agent.task_logger import log_edit


class EditEngine:
    """
    Grounded editing engine.
    
    Flow:
    1. Parse edit request → extract target file + instruction
    2. Retrieve full file content (no chunking)
    3. Send focused prompt to LLM
    4. LLM returns ONLY modified code
    5. Generate diff preview
    6. On confirm, apply edit + log to memory
    """
    
    def __init__(self, repo_path: str):
        self.repo_path = repo_path
        self.retriever = FileRetriever(repo_path)
        self.indexer = CodeIndexer(repo_path)
        self.llm = LocalLLMClient()
    
    def parse_edit_request(self, query: str) -> Dict:
        """
        Extract target file and edit instruction from query.
        
        Returns:
            {
                "file_path": str or None,
                "instruction": str,
                "target": str (function/class name if mentioned)
            }
        """
        query_lower = query.lower()
        
        # Try to find explicit file reference
        file_pattern = r'[\w\-_/]+\.(?:py|js|ts|go|rs|java|cpp|c|jsx|tsx)'
        file_matches = re.findall(file_pattern, query)
        
        file_path = None
        if file_matches:
            # Use the first file mentioned
            file_path = file_matches[0]
            # Verify it exists
            if not self.retriever.get_file(file_path):
                # Try to find it
                found = self.retriever.find_file_by_name(os.path.basename(file_path))
                if found:
                    file_path = found
                else:
                    file_path = None
        
        # If no explicit file, use retrieval to find relevant file
        if not file_path:
            results = self.indexer.search(query, k=1)
            if results:
                file_path = results[0]['file_path']
        
        # Extract function/class target if mentioned
        target = None
        func_match = re.search(r'(?:function|def|method)\s+(\w+)', query_lower)
        class_match = re.search(r'class\s+(\w+)', query_lower)
        
        if func_match:
            target = func_match.group(1)
        elif class_match:
            target = class_match.group(1)
        
        return {
            "file_path": file_path,
            "instruction": query,
            "target": target
        }
    
    def generate_edit(self, file_path: str, instruction: str, target: str = None) -> Tuple[str, str, str]:
        """
        Generate an edit for a file.
        
        Returns:
            (original_content, modified_content, change_summary)
        """
        # Get full file content
        original = self.retriever.get_file(file_path)
        if not original:
            return None, None, f"Could not read file: {file_path}"
        
        # If target specified, extract just that portion for context
        context = original
        if target:
            func_info = self.retriever.get_function(file_path, target)
            if func_info:
                context = f"Target function:\n{func_info['content']}\n\nFull file for context:\n{original}"
        
        # Focused prompt - no essays, just code
        prompt = f"""You are editing code. Return ONLY the complete modified file content.
No explanations. No markdown. No code fences. Just the code.

FILE: {file_path}
---
{original}
---

INSTRUCTION: {instruction}

Return the complete modified file:"""

        # Generate with LLM
        modified = self.llm.generate_text(prompt, max_tokens=4096, temperature=0.3)
        
        # Clean up response
        modified = self._clean_llm_output(modified, original)
        
        # Generate 1-line summary
        summary_prompt = f"Summarize this code change in ONE short sentence (max 10 words):\nInstruction: {instruction}"
        change_summary = self.llm.generate_text(summary_prompt, max_tokens=50, temperature=0.3)
        change_summary = change_summary.strip().split('\n')[0][:100]
        
        return original, modified, change_summary
    
    def _clean_llm_output(self, output: str, original: str) -> str:
        """Clean LLM output to get just the code."""
        # Remove markdown code fences if present
        output = re.sub(r'^```\w*\n?', '', output)
        output = re.sub(r'\n?```$', '', output)
        
        # Remove common prefixes
        prefixes = ['Here is the modified file:', 'Modified file:', 'Updated code:']
        for prefix in prefixes:
            if output.lower().startswith(prefix.lower()):
                output = output[len(prefix):].lstrip()
        
        # If output is suspiciously short, return original
        if len(output) < len(original) * 0.3:
            return original
        
        return output.strip()
    
    def preview_edit(self, file_path: str, instruction: str, target: str = None) -> Dict:
        """
        Generate edit and return preview (diff).
        
        Returns:
            {
                "success": bool,
                "file_path": str,
                "diff": str,
                "original": str,
                "modified": str,
                "summary": str,
                "error": str (if failed)
            }
        """
        if not file_path:
            return {"success": False, "error": "No file specified or found"}
        
        original, modified, summary = self.generate_edit(file_path, instruction, target)
        
        if original is None:
            return {"success": False, "error": summary}
        
        diff = create_diff(original, modified, file_path)
        
        if not diff.strip():
            return {
                "success": True,
                "file_path": file_path,
                "diff": "[No changes detected]",
                "original": original,
                "modified": modified,
                "summary": "No changes needed"
            }
        
        return {
            "success": True,
            "file_path": file_path,
            "diff": diff,
            "original": original,
            "modified": modified,
            "summary": summary
        }
    
    def apply_edit(self, file_path: str, modified_content: str, instruction: str, summary: str) -> Dict:
        """
        Apply an edit to a file and log it.
        
        Returns:
            {"success": bool, "message": str}
        """
        abs_path = os.path.join(self.repo_path, file_path)
        
        result = write_file(abs_path, modified_content)
        
        if "Successfully" in result:
            # Log to memory
            log_edit(self.repo_path, instruction, file_path, summary)
            return {"success": True, "message": f"✅ Applied edit to {file_path}"}
        else:
            return {"success": False, "message": result}
