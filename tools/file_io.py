import os
import requests

def read_file(file_path: str) -> str:
    """
    Reads the content of a file (Local path or HTTP URL).

    Args:
        file_path (str): The path or URL to the file.

    Returns:
        str: The content or error message.
    """
    try:
        # Remote File Handling
        if file_path.startswith("http"):
            response = requests.get(file_path)
            if response.status_code == 200:
                return response.text
            else:
                return f"Error reading remote file: {response.status_code} {response.reason}"
        
        # Local File Handling
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
            
    except Exception as e:
        return f"Error reading file {file_path}: {e}"

def write_file(file_path: str, content: str) -> str:
    """
    Writes content to a file. 
    NOTE: Remote writing is not supported in this version without authentication/API implementation.
    """
    if file_path.startswith("http"):
        return "Error: Cannot write directly to remote URLs. Please clone the repo to make edits."
        
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Successfully wrote to {file_path}"
    except Exception as e:
        return f"Error writing to file {file_path}: {e}"