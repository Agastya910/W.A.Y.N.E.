import os
import json
from typing import Dict, Any

class CodebaseMemory:
    """
    Manages the memory of the codebase, including summaries of files,
    modules, and the global architecture.
    """
    def __init__(self, repo_path: str):
        """
        Initializes the codebase memory.

        Args:
            repo_path (str): The path to the repository.
        """
        self.repo_path = repo_path
        self.memory_path = os.path.join(repo_path, ".repopilot", "memory.json")
        self.memory = self._load_memory()

    def _load_memory(self) -> Dict[str, Any]:
        """
        Loads the memory from the memory file.

        Returns:
            Dict[str, Any]: The loaded memory.
        """
        if os.path.exists(self.memory_path):
            with open(self.memory_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"files": {}, "modules": {}, "architecture": ""}

    def _save_memory(self):
        """
        Saves the memory to the memory file.
        """
        os.makedirs(os.path.dirname(self.memory_path), exist_ok=True)
        with open(self.memory_path, "w", encoding="utf-8") as f:
            json.dump(self.memory, f, indent=2)

    def get_file_summary(self, file_path: str) -> str:
        """
        Gets the summary of a file.

        Args:
            file_path (str): The path to the file.

        Returns:
            str: The summary of the file, or an empty string if not found.
        """
        return self.memory["files"].get(file_path, "")

    def set_file_summary(self, file_path: str, summary: str):
        """
        Sets the summary of a file.

        Args:
            file_path (str): The path to the file.
            summary (str): The summary of the file.
        """
        self.memory["files"][file_path] = summary
        self._save_memory()

    def get_module_summary(self, module_path: str) -> str:
        """
        Gets the summary of a module.

        Args:
            module_path (str): The path to the module.

        Returns:
            str: The summary of the module, or an empty string if not found.
        """
        return self.memory["modules"].get(module_path, "")

    def set_module_summary(self, module_path: str, summary: str):
        """
        Sets the summary of a module.

        Args:
            module_path (str): The path to the module.
            summary (str): The summary of the module.
        """
        self.memory["modules"][module_path] = summary
        self._save_memory()

    def get_architecture_summary(self) -> str:
        """
        Gets the global architecture summary.

        Returns:
            str: The global architecture summary, or an empty string if not found.
        """
        return self.memory.get("architecture", "")

    def set_architecture_summary(self, summary: str):
        """
        Sets the global architecture summary.

        Args:
            summary (str): The global architecture summary.
        """
        self.memory["architecture"] = summary
        self._save_memory()

if __name__ == '__main__':
    # For testing purposes
    repo_path = "test_repo"
    os.makedirs(repo_path, exist_ok=True)
    memory = CodebaseMemory(repo_path)
    
    # Test file summary
    memory.set_file_summary("src/main.py", "This is the main file.")
    print(f"File summary: {memory.get_file_summary('src/main.py')}")
    
    # Test module summary
    memory.set_module_summary("src", "This is the source module.")
    print(f"Module summary: {memory.get_module_summary('src')}")

    # Test architecture summary
    memory.set_architecture_summary("This is a simple test project.")
    print(f"Architecture summary: {memory.get_architecture_summary()}")

    # cleanup
    import shutil
    shutil.rmtree(repo_path)
