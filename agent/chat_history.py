import json
import os
from collections import deque
from typing import List, Dict, Optional

class ChatHistory:
    """
    3-layer memory system for WAYNE:
    1. System context (Fixed)
    2. Summary (Compressed history of older turns)
    3. Sliding window (Last 2 raw turns)
    """
    def __init__(self, repo_path: str, max_recent: int = 2):
        self.repo_path = repo_path
        self.max_recent = max_recent  # keep last N turns raw
        self.recent = deque(maxlen=max_recent)
        self.summary = ""
        self.edits_log = []  # structured, not prose
        self._path = os.path.join(repo_path, ".repopilot", "chat_history.json")
        self._load()

    def add_turn(self, user_query: str, assistant_action: str, edit_info: dict = None):
        """Call after every completed turn."""
        # If we're at capacity, compress the oldest turn into summary
        if len(self.recent) == self.max_recent:
            oldest = self.recent[0] # The deque will push it out, but we want to summarize it before it's gone if we were to add more. 
            # Actually deque(maxlen=2) handles removal. In the design doc turn addition is simple.
            self._compress_into_summary(oldest)

        turn = {
            "user": user_query,
            "action": assistant_action[:150],  # truncate
        }
        if edit_info:
            turn["edit"] = {
                "file": edit_info.get("file_path"),
                "change": edit_info.get("summary", "")[:80]
            }
            self.edits_log.append(turn["edit"])

        self.recent.append(turn)
        self._save()

    def _compress_into_summary(self, turn: dict):
        """Rule-based compression — NO LLM call."""
        parts = []
        if self.summary:
            parts.append(self.summary)
        desc = f"User asked: '{turn['user'][:60]}' → {turn['action'][:60]}"
        if "edit" in turn and turn["edit"]:
            desc += f" [edited {turn['edit']['file']}]"
        parts.append(desc)
        # Keep summary under ~300 tokens (~1200 chars)
        self.summary = "\n".join(parts)[-1200:]

    def get_context_block(self) -> str:
        """Returns the string to inject into your LLM prompt."""
        parts = []
        if self.summary:
            parts.append(f"[Previous context]\n{self.summary}")
        if self.edits_log:
            recent_edits = self.edits_log[-5:]
            edits_str = ", ".join(f"{e['file']}({e['change'][:30]})" for e in recent_edits if e)
            parts.append(f"[Recent edits] {edits_str}")
        for turn in self.recent:
            parts.append(f"User: {turn['user'][:100]}\nAction: {turn['action'][:100]}")
        return "\n".join(parts) if parts else ""

    def _save(self):
        os.makedirs(os.path.dirname(self._path), exist_ok=True)
        data = {
            "summary": self.summary,
            "recent": list(self.recent),
            "edits_log": self.edits_log[-20:]
        }
        try:
            with open(self._path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"[CHAT HISTORY] Warning: Could not save history: {e}")

    def _load(self):
        if os.path.exists(self._path):
            try:
                with open(self._path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.summary = data.get("summary", "")
                self.recent = deque(data.get("recent", []), maxlen=self.max_recent)
                self.edits_log = data.get("edits_log", [])
            except Exception as e:
                print(f"[CHAT HISTORY] Warning: Could not load history: {e}")
