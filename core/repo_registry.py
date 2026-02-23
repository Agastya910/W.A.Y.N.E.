"""
Repo Registry — deterministic per-repo identification for WAYNE.

Each repository gets a unique, stable ID derived from its canonical
absolute path. This ID is used to create isolated Qdrant collections
so that multiple repos never pollute each other's vector stores.
"""

import hashlib
import os
import re


def _canonical_path(repo_path: str) -> str:
    """Resolve a repo path to its canonical form for stable hashing."""
    return os.path.realpath(os.path.abspath(repo_path))


def get_repo_id(repo_path: str) -> str:
    """
    Generate a deterministic 8-char hex ID from the repo's canonical path.
    
    This matches the naming scheme already used in .repopilot_indexes/
    (e.g. claim-verifier_5eaca942).
    """
    canonical = _canonical_path(repo_path)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:8]


def _sanitize_name(name: str) -> str:
    """
    Sanitize a repo name for use in Qdrant collection names.
    Qdrant allows alphanumeric, hyphens, and underscores.
    """
    sanitized = re.sub(r"[^a-zA-Z0-9_-]", "_", name)
    # Collapse multiple underscores
    sanitized = re.sub(r"_+", "_", sanitized).strip("_")
    return sanitized[:40]  # keep it short


def get_collection_name(repo_path: str) -> str:
    """
    Generate a unique Qdrant collection name for a repository.
    
    Format: wayne_{sanitized_repo_name}_{hash8}
    Example: wayne_claim-verifier_5eaca942
    """
    repo_name = os.path.basename(os.path.normpath(repo_path))
    sanitized = _sanitize_name(repo_name)
    repo_id = get_repo_id(repo_path)
    return f"wayne_{sanitized}_{repo_id}"


def get_repo_metadata(repo_path: str) -> dict:
    """
    Return full metadata for a repo, useful for logging and debugging.
    """
    canonical = _canonical_path(repo_path)
    repo_name = os.path.basename(os.path.normpath(repo_path))
    repo_id = get_repo_id(repo_path)
    collection = get_collection_name(repo_path)
    return {
        "repo_id": repo_id,
        "repo_name": repo_name,
        "collection_name": collection,
        "canonical_path": canonical,
    }
