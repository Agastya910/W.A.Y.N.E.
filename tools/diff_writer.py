import difflib
from typing import List

def create_diff(original_content: str, modified_content: str, file_path: str) -> str:
    """
    Creates a git-style diff between two strings.
    """
    diff = difflib.unified_diff(
        original_content.splitlines(keepends=True),
        modified_content.splitlines(keepends=True),
        fromfile=f"a/{file_path}",
        tofile=f"b/{file_path}",
    )
    return "".join(diff)

def write_diff(original_content: str, modified_content: str, file_path: str) -> str:
    """
    Generates a diff and returns it.
    """
    diff = create_diff(original_content, modified_content, file_path)
    # We return the diff so the executor can capture it
    return diff

if __name__ == '__main__':
    original = "hello world\nthis is a test\n"
    modified = "hello world\nthis is a modified test\n"
    file_path = "test.txt"
    print(write_diff(original, modified, file_path))
