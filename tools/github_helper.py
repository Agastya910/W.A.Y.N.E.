"""
Helper for cloning and analyzing GitHub repositories.
"""
import subprocess
import os
from typing import Dict, Any, Optional


def clone_github_repo(repo_url: str, dest_path: str = "./cloned_repo", timeout: int = 120) -> Dict[str, Any]:
    """
    Clone a GitHub repository using shallow clone (faster for large repos).
    
    Args:
        repo_url: Full GitHub URL (e.g., https://github.com/user/repo.git or https://github.com/user/repo)
        dest_path: Where to clone it
        timeout: How long to wait (default 120s for large repos)
    
    Returns:
        {"success": bool, "path": str, "message": str}
    """
    try:
        # Normalize URL
        if not repo_url.endswith('.git'):
            repo_url = repo_url + '.git'
        
        # Clean up old clone if exists
        if os.path.exists(dest_path):
            try:
                subprocess.run(["rm", "-rf", dest_path], check=True, timeout=10)
            except:
                pass
        
        print(f"[GITHUB] Cloning {repo_url} (this may take a minute for large repos)...")
        
        # Use shallow clone to speed up large repos
        result = subprocess.run(
            ["git", "clone", "--depth", "1", repo_url, dest_path],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        if result.returncode == 0:
            # Count files
            try:
                file_count = sum([len(files) for _, _, files in os.walk(dest_path)])
                return {
                    "success": True,
                    "path": dest_path,
                    "message": f"✅ Successfully cloned {repo_url}\nLocal path: {dest_path}\nFiles: ~{file_count}",
                    "file_count": file_count
                }
            except:
                return {
                    "success": True,
                    "path": dest_path,
                    "message": f"✅ Successfully cloned {repo_url} to {dest_path}",
                }
        else:
            error_msg = result.stderr.strip()
            
            # Provide helpful error messages
            if "not found" in error_msg.lower() or "404" in error_msg:
                return {
                    "success": False,
                    "path": None,
                    "message": f"❌ Repository not found: {repo_url}\n\nMake sure:\n  - The URL is correct\n  - The repo is public (private repos need SSH keys)\n  - GitHub is accessible from your network"
                }
            elif "permission denied" in error_msg.lower():
                return {
                    "success": False,
                    "path": None,
                    "message": f"❌ Permission denied. This is likely a private repository.\n\nTo use private repos, set up SSH:\n  ssh-keygen -t ed25519\n  cat ~/.ssh/id_ed25519.pub  # Add this to GitHub settings"
                }
            else:
                return {
                    "success": False,
                    "path": None,
                    "message": f"❌ Git clone failed:\n{error_msg}"
                }
    
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "path": None,
            "message": f"❌ Clone timed out after {timeout}s. The repository may be too large.\n\nTry:\n  - Using a smaller repo to test\n  - Increasing timeout with --timeout flag\n  - Using SSH key auth for faster access"
        }
    except Exception as e:
        return {
            "success": False,
            "path": None,
            "message": f"❌ Error cloning repo:\n{str(e)}"
        }


def get_repo_url_from_query(query: str) -> Optional[str]:
    """
    Extract GitHub URL from a user query.
    
    Examples:
        "Analyze https://github.com/user/repo"
        "Clone and understand facebook/react"
        "Find auth in vuejs/vue"
    """
    import re
    
    # Look for full URL first
    url_match = re.search(r'https?://github\.com/[\w\-./]+(?:\.git)?', query)
    if url_match:
        url = url_match.group(0)
        if not url.endswith('.git'):
            url += '.git'
        return url
    
    # Look for shorthand (user/repo)
    # This regex is more careful to avoid matching too much
    shorthand_match = re.search(r'\b([a-zA-Z0-9][a-zA-Z0-9\-]{0,38})/([a-zA-Z0-9][a-zA-Z0-9\-_.]{0,38})\b', query)
    if shorthand_match:
        user, repo = shorthand_match.groups()
        # Avoid matching common words
        if user.lower() not in ['the', 'and', 'or', 'for', 'with', 'from']:
            return f"https://github.com/{user}/{repo}.git"
    
    return None


if __name__ == "__main__":
    # Test
    result = clone_github_repo("https://github.com/torvalds/linux.git", "./test_clone", timeout=30)
    print(result)
