import re
from typing import Literal
from enum import Enum

class QueryType(Enum):
    """Types of queries the agent can handle."""
    METADATA = "metadata"          # File counts, structure → local tools
    SEARCH = "search"              # Find where X is → retrieval + local
    REASONING = "reasoning"        # Explain, refactor → LLM + retrieval
    TOOL_CALL = "tool_call"        # git_clone with URL → executor
    EDIT = "edit"                  # Edit/modify/fix code → edit engine
    UNDO = "undo"                  # Revert last edit
    FIX = "fix"
    INDEX_DOCS = "index_docs"


class QueryRouter:
    """
    Routes queries to the appropriate handler.
    Prevents trivial queries from hitting the LLM.
    """
    
    @staticmethod
    def classify(query: str) -> QueryType:
        """Classify query into one of the above types."""
        query_lower = query.lower()
        
        # UNDO queries
        undo_keywords = ["undo", "revert", "cancel last", "go back"]
        if any(kw in query_lower for kw in undo_keywords):
            return QueryType.UNDO
            
        # EDIT queries - check first (highest priority for editing)
        edit_keywords = ["edit", "change", "modify", "update", "fix", "refactor", "add to", "remove from", "delete from", "replace", "rename"]
        if any(kw in query_lower for kw in edit_keywords):
            return QueryType.EDIT
        
        # FIX queries
        fix_keywords = ["fix", "heal", "auto-fix", "self-heal", "debug and fix", "fix errors in"]
        if any(kw in query_lower for kw in fix_keywords):
            return QueryType.FIX

        # INDEX_DOCS queries
        index_keywords = ["index documents", "index folder", "index files", "add documents", "scan folder"]
        if any(kw in query_lower for kw in index_keywords):
            return QueryType.INDEX_DOCS

        # METADATA queries
        metadata_keywords = ["how many", "list", "count", "what files", "structure", "hierarchy", "number of files"]
        if any(kw in query_lower for kw in metadata_keywords):
            return QueryType.METADATA
        
        # SEARCH queries
        search_keywords = ["where", "find", "locate", "look for", "search for"]
        if any(kw in query_lower for kw in search_keywords):
            return QueryType.SEARCH
        
        # TOOL queries - only if there's an actual URL/actionable link
        tool_keywords = ["clone", "download"]
        has_url = bool(re.search(r'https?://|github\.com/[\w\-/]+', query_lower))
        if any(kw in query_lower for kw in tool_keywords) and has_url:
            return QueryType.TOOL_CALL
        
        # CAPABILITY questions → reasoning (not tool_call)
        capability_keywords = ["can you", "would it", "is it able", "do you support", "can it"]
        if any(kw in query_lower for kw in capability_keywords):
            return QueryType.REASONING
        
        # Default: REASONING
        return QueryType.REASONING
    
    @staticmethod
    def needs_llm(query_type: QueryType) -> bool:
        """Does this query need LLM reasoning?"""
        return query_type in [QueryType.REASONING, QueryType.SEARCH]


if __name__ == "__main__":
    router = QueryRouter()
    
    test_queries = [
        "How many Python files are in this repo?",
        "Where is authentication handled?",
        "Refactor this module to use dependency injection",
        "Clone https://github.com/user/repo.git",
        "would it take github repos as input?",
        "can you correct errors in this codebase?",
        "is it able to analyze multiple repos?",
    ]
    
    for q in test_queries:
        query_type = router.classify(q)
        print(f"'{q}' → {query_type.value}")
