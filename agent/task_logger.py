"""
Task Logger - Lightweight edit memory

Appends completed edits to agent_memory/task.md
"""

import os
from datetime import datetime


def log_edit(repo_path: str, user_instruction: str, file_path: str, change_summary: str):
    """
    Log a completed edit to agent_memory/task.md
    
    Format:
    ### <timestamp>
    User: <short instruction>
    File: <path>
    Change: <1 line summary>
    Status: done
    """
    memory_dir = os.path.join(repo_path, "agent_memory")
    os.makedirs(memory_dir, exist_ok=True)
    
    task_file = os.path.join(memory_dir, "task.md")
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    
    # Truncate instruction to keep it compact
    short_instruction = user_instruction[:80] + "..." if len(user_instruction) > 80 else user_instruction
    
    entry = f"""
### {timestamp}
User: {short_instruction}
File: {file_path}
Change: {change_summary}
Status: done
"""
    
    with open(task_file, "a", encoding="utf-8") as f:
        f.write(entry)
    
    return task_file
