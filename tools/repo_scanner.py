import os
from typing import Dict, List, Any

# Common junk directories and files to ignore
IGNORE_DIRS = {
    "node_modules",
    ".git",
    "venv",
    "build",
    "dist",
    "__pycache__",
    ".idea",
    ".vscode",
    ".repopilot_index",
    ".pytest_cache",
    ".venv",
    "env",
    ".env",
    "htmlcov",
    ".coverage",
    "site-packages",
    ".tox",
    ".eggs",
    "*.egg-info",
    "target",
    ".gradle",
    "node_modules",
    "vendor",
    "deps",
    "tmp",
}

IGNORE_FILES = {
    ".DS_Store",
    ".gitignore",
    ".gitattributes",
    "thumbs.db",
}


def get_language(file_path: str) -> str:
    """
    Determines the programming language of a file based on its extension.
    
    Args:
        file_path (str): The path to the file.
    
    Returns:
        str: The programming language, or "unknown" if not recognized.
    """
    extension_map = {
        ".py": "Python",
        ".js": "JavaScript",
        ".ts": "TypeScript",
        ".tsx": "TypeScript",
        ".jsx": "JavaScript",
        ".java": "Java",
        ".c": "C",
        ".cpp": "C++",
        ".cc": "C++",
        ".cxx": "C++",
        ".h": "C/C++ Header",
        ".hpp": "C++ Header",
        ".cs": "C#",
        ".go": "Go",
        ".rs": "Rust",
        ".php": "PHP",
        ".rb": "Ruby",
        ".swift": "Swift",
        ".kt": "Kotlin",
        ".scala": "Scala",
        ".m": "Objective-C",
        ".mm": "Objective-C++",
        ".sh": "Shell",
        ".bash": "Shell",
        ".zsh": "Shell",
        ".fish": "Shell",
        ".html": "HTML",
        ".htm": "HTML",
        ".css": "CSS",
        ".scss": "SCSS",
        ".sass": "SASS",
        ".less": "Less",
        ".json": "JSON",
        ".xml": "XML",
        ".yaml": "YAML",
        ".yml": "YAML",
        ".md": "Markdown",
        ".txt": "Text",
        ".sql": "SQL",
        ".r": "R",
        ".R": "R",
        ".lua": "Lua",
        ".vim": "VimL",
        ".pl": "Perl",
        ".java": "Java",
    }
    
    _, ext = os.path.splitext(file_path)
    return extension_map.get(ext.lower(), "unknown")


def scan_repo(repo_path: str, max_depth: int = 20) -> Dict[str, Any]:
    """
    Scans a repository and builds a file tree with metadata.
    More permissive than before for large repos like Linux kernel.
    
    Args:
        repo_path (str): The path to the repository.
        max_depth (int): Maximum directory depth to scan (to avoid infinite recursion).
    
    Returns:
        Dict[str, Any]: A dictionary representing the file tree.
    """
    file_tree = {}
    
    try:
        for root, dirs, files in os.walk(repo_path):
            # Calculate current depth
            current_depth = root[len(repo_path):].count(os.sep)
            if current_depth > max_depth:
                dirs[:] = []  # Don't recurse deeper
                continue
            
            # Remove ignored directories IN-PLACE
            dirs_to_remove = []
            for d in dirs:
                if d in IGNORE_DIRS or d.startswith('.'):
                    dirs_to_remove.append(d)
            for d in dirs_to_remove:
                dirs.remove(d)
            
            # Build current level in tree
            current_level = file_tree
            
            # Navigate/create nested dict for this path
            rel_path = os.path.relpath(root, repo_path)
            if rel_path != ".":
                path_parts = rel_path.split(os.path.sep)
                for part in path_parts:
                    if part and part not in current_level:
                        current_level[part] = {}
                    if part:
                        current_level = current_level[part]
            
            # Add files to current level
            for file in files:
                if file in IGNORE_FILES or file.startswith('.'):
                    continue
                
                file_path = os.path.join(root, file)
                try:
                    file_size = os.path.getsize(file_path)
                    language = get_language(file)
                    rel_file_path = os.path.relpath(file_path, repo_path)
                    
                    current_level[file] = {
                        "type": "file",
                        "size": file_size,
                        "language": language,
                        "path": rel_file_path,
                    }
                except OSError:
                    # File might be inaccessible, skip it
                    pass
    
    except Exception as e:
        print(f"[SCANNER] Error scanning repository: {e}")
    
    return file_tree


if __name__ == '__main__':
    # For testing purposes
    import json
    
    # Create a dummy repo structure for testing
    os.makedirs("test_repo/src", exist_ok=True)
    with open("test_repo/src/main.py", "w") as f:
        f.write("print('hello world')")
    
    os.makedirs("test_repo/node_modules", exist_ok=True)  # This should be ignored
    
    repo_tree = scan_repo("test_repo")
    print(json.dumps(repo_tree, indent=2))
    
    # Cleanup
    import shutil
    shutil.rmtree("test_repo")
