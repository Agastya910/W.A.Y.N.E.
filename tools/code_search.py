import re
import os
import requests
from typing import List, Dict, Any
from .repo_scanner import scan_repo
from config import GITHUB_TOKEN

def search_github(repo_url: str, query: str) -> List[Dict[str, Any]]:
    """Searches a remote repo using GitHub Search API with authentication."""
    parts = repo_url.replace("https://github.com/", "").split("/")
    if len(parts) < 2: return []
    owner, repo = parts[0], parts[1]
    
    # Auth headers are critical for the Search API
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json"
    } if GITHUB_TOKEN else {}
    
    # GitHub Search API 
    api_url = f"https://api.github.com/search/code?q={query}+repo:{owner}/{repo}"
    
    try:
        resp = requests.get(api_url, headers=headers)
        if resp.status_code == 200:
            items = resp.json().get("items", [])
            results = []
            for item in items:
                # Determine raw url for download/viewing
                raw_url = item["html_url"].replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
                results.append({
                    "file_path": item["path"],
                    "line_number": "N/A (Remote Search)", 
                    "line": f"Match found in {item['name']}",
                    "download_url": raw_url
                })
            return results
        return [{"error": f"GitHub API Error: {resp.status_code} - {resp.reason}"}]
    except Exception as e:
        return [{"error": str(e)}]

def search_code(repo_path: str, query: str, regex: bool = False) -> List[Dict[str, Any]]:
    """
    Searches for a query in the code of a repository.
    """
    if repo_path.startswith("http"):
        # Regex is not supported by GitHub API, fallback to simple query
        return search_github(repo_path, query)

    results = []
    file_tree = scan_repo(repo_path)

    def search_in_files(tree: Dict[str, Any]):
        for name, item in tree.items():
            if isinstance(item, dict) and "type" in item and item["type"] == "file":
                file_path = item["path"] # Local absolute path
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        for i, line in enumerate(f):
                            if regex:
                                if re.search(query, line):
                                    results.append({"file_path": item["path"], "line_number": i + 1, "line": line.strip()})
                            else:
                                if query in line:
                                    results.append({"file_path": item["path"], "line_number": i + 1, "line": line.strip()})
                except Exception:
                    pass
            elif isinstance(item, dict):
                search_in_files(item)

    search_in_files(file_tree)
    return results