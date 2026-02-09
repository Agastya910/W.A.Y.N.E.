
# Adding Chat History to RepoPilot

You have two separate problems, and I'll give you a concrete plan for each. No new files beyond one.

---

## Problem 1: Undo Edits

Your `EditEngine` already stores the `original` content during `preview_edit`, but you throw it away after `apply_edit` [^1]. The fix is dead simple — keep a stack of `(file_path, original_content)` in the `Executor`.

**In `executor.py`, change `_apply_edit_tool`** to push onto a stack before applying:

```python
# Add to __init__
self._edit_history = []  # stack of (file_path, original_content, instruction)

# In _apply_edit_tool, BEFORE calling self.edit_engine.apply_edit():
self._edit_history.append({
    "file_path": self._pending_edit["file_path"],
    "original": open(os.path.join(self.repo_path, self._pending_edit["file_path"])).read(),
    "instruction": self._pending_edit["instruction"]
})
```

Then add an undo method:

```python
def undo_last_edit(self):
    if not self._edit_history:
        return {"success": False, "message": "Nothing to undo"}
    entry = self._edit_history.pop()
    abs_path = os.path.join(self.repo_path, entry["file_path"])
    with open(abs_path, "w") as f:
        f.write(entry["original"])
    return {"success": True, "message": f"↩️ Reverted {entry['file_path']}"}
```

Register `"undo"` in your `self.tools` dict, and detect "undo" in the `QueryRouter` [^8]. That's it — no LLM involved, purely mechanical [^6].

---

## Problem 2: Conversation History for Context

This is the harder problem. You're running Qwen 7B Q4 with roughly a 4K–8K usable context window [^18]. You can't dump the full chat history in there. The research-proven approach for small local models is a **3-layer memory** system [^19][^34]:

### The Architecture

| Layer | What it stores | Token cost | When it's used |
|---|---|---|---|
| **System context** | System prompt + current task | Fixed ~300 tokens | Every turn |
| **Summary** | Compressed history of all older turns | ~200–400 tokens | Every turn |
| **Sliding window** | Last 2 raw user+assistant turns | Variable, ~500–1000 tokens | Every turn |

This is a simplified version of what MemGPT calls "tiered memory" and what Claude Code calls "compaction" [^36][^37]. The key insight from Factory's research: **structured summaries** (not free-form prose) retain the information that matters for task continuation [^39].

### Implementation: One New File

Create `agent/chat_history.py`:

```python
import json
import os
from collections import deque

class ChatHistory:
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
            oldest = self.recent
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
        if "edit" in turn:
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
            edits_str = ", ".join(f"{e['file']}({e['change'][:30]})" for e in recent_edits)
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
        with open(self._path, "w") as f:
            json.dump(data, f, indent=2)

    def _load(self):
        if os.path.exists(self._path):
            with open(self._path) as f:
                data = json.load(f)
            self.summary = data.get("summary", "")
            self.recent = deque(data.get("recent", []), maxlen=self.max_recent)
            self.edits_log = data.get("edits_log", [])
```

### Wire It Up

**In `cli.py`**, add 3 lines to your main loop [^16]:

```python
# After initializing planner/executor/verifier:
from agent.chat_history import ChatHistory
history = ChatHistory(repo_path)

# After each completed turn (after verifier):
edit_info = executor.get_pending_edit_info() if executor.has_pending_edit() else None
history.add_turn(query, results.get("tool", ""), edit_info)
```

**In `planner.py`**, inject the context into your LLM prompts [^2]:

```python
# In __init__:
from agent.chat_history import ChatHistory
self.history = ChatHistory(repo_path)

# In _plan_with_llm, add to the prompt:
history_context = self.history.get_context_block()
prompt = f"""
You are a senior software engineer analyzing a codebase.

{f"[Conversation history]{chr(10)}{history_context}" if history_context else ""}

User Query: "{user_query}"
...
"""
```

### Why This Specific Design

- **No LLM call for compression.** Your Qwen 7B is slow enough already — using it to summarize would double latency. The rule-based `_compress_into_summary` is instant and predictable [^42][^39].
- **Structured edits log separate from prose summary.** When the user says "undo the edit," you don't need the LLM to figure out what happened — you have a structured list [^39]. The `QueryRouter` can detect "undo" keywords and route directly to the undo stack without ever touching the LLM.
- **Only 2 raw turns in the sliding window.** For a 7B model at Q4, you want to reserve at most ~1500 tokens for history, leaving the rest for system prompt + retrieved code + generation [^22][^19]. Two turns is the sweet spot — the user can refer to "what I just said" without blowing your budget.
- **Persists to disk.** If the user Ctrl+C's and restarts (which your logs show they do), the history survives [^16].

### What NOT to Do

- **Don't pass all messages as-is.** Small models hallucinate dramatically when the context is stuffed. Pinecone's LangChain guide confirms that even `ConversationBufferWindowMemory(k=1)` is used for constrained models [^31].
- **Don't use LLM-based summarization** unless you upgrade to a 70B+ model or use an API. For local 7B, rule-based compression of structured data outperforms LLM-generated summaries which can introduce hallucinated context [^39][^29].
- **Don't use vector-store memory** for this use case. It adds complexity and latency, and your conversations are short enough that a simple deque + summary covers it [^19].

### Token Budget Breakdown

For a ~4K context window with Qwen 7B:

| Component | Tokens |
|---|---|
| System prompt | ~300 |
| Chat history block | ~400–600 |
| Retrieved code chunks (top 3 instead of 5) | ~1500 |
| Generation headroom | ~1200 |
| **Total** | **~4000** |

You may want to reduce `k=5` to `k=3` in `_retrieve_context` in `planner.py` to make room for the history block [^2].


---

## References

1. [edit_engine.py](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/58908120/9f8e3814-3f59-4463-aa45-2d7422e98918/edit_engine.py?AWSAccessKeyId=ASIA2F3EMEYERORSWIQR&Signature=D4CMkHI439N4O7w7%2F%2F6D5ZthMno%3D&x-amz-security-token=IQoJb3JpZ2luX2VjEMD%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJGMEQCIHKdXphv%2BMmkknx1a4PCN%2FpEthWMxNgB19%2BpO9b%2F9tJtAiAPmeVjU59oX7mmNfrhwjSTtyIseWQn0gf8%2FHdBdHmfQir8BAiJ%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F8BEAEaDDY5OTc1MzMwOTcwNSIMGFYlSVRDOvm11tfVKtAEyeX%2FjZFIrYPMAMDbH3P5tnooBdQ%2Br4vdConwYPzzCKGxTNG7r9kFcQWkjvSXsWC1%2F6eEoYxE46%2BWEFkd2b%2Bauv2sPyRotX3R9Pz5Ky3gbHBCdZyUK7pdOugIc4PL18JUyFxCeJL8xOo6dqtUNrUcBZ65GbTZ1o%2BLxdrV4lUaxzFRLZeFF%2B%2BiwohdNeMxA2qXuuVVwrVxDOBqKXbhj%2F4jt1jLEL2bdZheHIqWTxerFldJKtynsMYMw7h0TQuk2uVXrgA4ob8ythXu%2BY3idbjN%2FUbQ3WHiBa4dfIBpsxE%2BgllO8uE2oDShXJ6ztcWOTHziKPzYy%2F6VscCGcTvqDsB1XiDPXrTMatoxLSOCudSawXJxeIG4ceY0sBBYQCNsZwwoF9WpTpxw7IHh2p1PnI1we7%2BFckdHT%2BfDIAFBFXFAch9yHFJR9SJPwISfxzNEsH2QfoUyz5As0vmpBE9eXn7JIMUBHwwIREnI%2BMzr%2FamGE9FKO8Dq%2F%2FcZO4LgmZzekshL3WMJSwtElh%2BKKAVl5rcaFZ2uDduUgbHG9ApdK83UT1R%2F3JyvVzxaSPojplvgUpkOYi6B8iQ0yyc6AsDZC%2FXgqnEHToCfRAffKbxvlb9M98sqN02rCuJ6AcUk8XzS21bCJTUvOWJCBhT%2B%2FXGGOdS1nffmP%2Fto7UF0fz4AN0V%2F8%2FkOK2LKjTHiSCR5assbLKccqcFYAttxYTFAFqYaZrcFkvCyBGP6ZbY90T3NuVpsvWaZUq2O2UU53VgGHVVnqFhJWmt9GyPwHFv1LIsLkYo8ZzDroabMBjqZAei%2FxlzRBA0dkjveZmnU13lsN2JZvWZP7%2BDcunDusOrMtbCI8t1gJ1jqJG64sP1u3dKp2Ez2qqHixZKGo2%2BNBCFeykq53l20Q%2BgVekA1mIuSnikcjIRVytzJD3G%2FiTkMpcDP%2F0tdAjGinDO7ivp9OXz86vTlPulgtu7h%2FJxo631gy%2BqAZojF6RFrrcnm0z%2F0yUyKNrbJKMjriw%3D%3D&Expires=1770628000) - """

Edit Engine - Grounded file editing with retrieval

No hallucination. No long explanations.

Us...

2. [planner.py](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/58908120/e9908341-8bcc-4614-a9ce-6d65bd78a6c7/planner.py?AWSAccessKeyId=ASIA2F3EMEYERORSWIQR&Signature=IjNOsJm15dmVKyTOMIsL50ZsDr0%3D&x-amz-security-token=IQoJb3JpZ2luX2VjEMD%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJGMEQCIHKdXphv%2BMmkknx1a4PCN%2FpEthWMxNgB19%2BpO9b%2F9tJtAiAPmeVjU59oX7mmNfrhwjSTtyIseWQn0gf8%2FHdBdHmfQir8BAiJ%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F8BEAEaDDY5OTc1MzMwOTcwNSIMGFYlSVRDOvm11tfVKtAEyeX%2FjZFIrYPMAMDbH3P5tnooBdQ%2Br4vdConwYPzzCKGxTNG7r9kFcQWkjvSXsWC1%2F6eEoYxE46%2BWEFkd2b%2Bauv2sPyRotX3R9Pz5Ky3gbHBCdZyUK7pdOugIc4PL18JUyFxCeJL8xOo6dqtUNrUcBZ65GbTZ1o%2BLxdrV4lUaxzFRLZeFF%2B%2BiwohdNeMxA2qXuuVVwrVxDOBqKXbhj%2F4jt1jLEL2bdZheHIqWTxerFldJKtynsMYMw7h0TQuk2uVXrgA4ob8ythXu%2BY3idbjN%2FUbQ3WHiBa4dfIBpsxE%2BgllO8uE2oDShXJ6ztcWOTHziKPzYy%2F6VscCGcTvqDsB1XiDPXrTMatoxLSOCudSawXJxeIG4ceY0sBBYQCNsZwwoF9WpTpxw7IHh2p1PnI1we7%2BFckdHT%2BfDIAFBFXFAch9yHFJR9SJPwISfxzNEsH2QfoUyz5As0vmpBE9eXn7JIMUBHwwIREnI%2BMzr%2FamGE9FKO8Dq%2F%2FcZO4LgmZzekshL3WMJSwtElh%2BKKAVl5rcaFZ2uDduUgbHG9ApdK83UT1R%2F3JyvVzxaSPojplvgUpkOYi6B8iQ0yyc6AsDZC%2FXgqnEHToCfRAffKbxvlb9M98sqN02rCuJ6AcUk8XzS21bCJTUvOWJCBhT%2B%2FXGGOdS1nffmP%2Fto7UF0fz4AN0V%2F8%2FkOK2LKjTHiSCR5assbLKccqcFYAttxYTFAFqYaZrcFkvCyBGP6ZbY90T3NuVpsvWaZUq2O2UU53VgGHVVnqFhJWmt9GyPwHFv1LIsLkYo8ZzDroabMBjqZAei%2FxlzRBA0dkjveZmnU13lsN2JZvWZP7%2BDcunDusOrMtbCI8t1gJ1jqJG64sP1u3dKp2Ez2qqHixZKGo2%2BNBCFeykq53l20Q%2BgVekA1mIuSnikcjIRVytzJD3G%2FiTkMpcDP%2F0tdAjGinDO7ivp9OXz86vTlPulgtu7h%2FJxo631gy%2BqAZojF6RFrrcnm0z%2F0yUyKNrbJKMjriw%3D%3D&Expires=1770628000) - from typing import List, Dict, Any

import json

import re

import os

from llm.local_llm_client imp...

6. [executor.py](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/58908120/2b23743e-7b11-4558-a650-d0007349fa09/executor.py?AWSAccessKeyId=ASIA2F3EMEYERORSWIQR&Signature=viooeYrmllcL3EUZqP1%2Fdq%2BmB8E%3D&x-amz-security-token=IQoJb3JpZ2luX2VjEMD%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJGMEQCIHKdXphv%2BMmkknx1a4PCN%2FpEthWMxNgB19%2BpO9b%2F9tJtAiAPmeVjU59oX7mmNfrhwjSTtyIseWQn0gf8%2FHdBdHmfQir8BAiJ%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F8BEAEaDDY5OTc1MzMwOTcwNSIMGFYlSVRDOvm11tfVKtAEyeX%2FjZFIrYPMAMDbH3P5tnooBdQ%2Br4vdConwYPzzCKGxTNG7r9kFcQWkjvSXsWC1%2F6eEoYxE46%2BWEFkd2b%2Bauv2sPyRotX3R9Pz5Ky3gbHBCdZyUK7pdOugIc4PL18JUyFxCeJL8xOo6dqtUNrUcBZ65GbTZ1o%2BLxdrV4lUaxzFRLZeFF%2B%2BiwohdNeMxA2qXuuVVwrVxDOBqKXbhj%2F4jt1jLEL2bdZheHIqWTxerFldJKtynsMYMw7h0TQuk2uVXrgA4ob8ythXu%2BY3idbjN%2FUbQ3WHiBa4dfIBpsxE%2BgllO8uE2oDShXJ6ztcWOTHziKPzYy%2F6VscCGcTvqDsB1XiDPXrTMatoxLSOCudSawXJxeIG4ceY0sBBYQCNsZwwoF9WpTpxw7IHh2p1PnI1we7%2BFckdHT%2BfDIAFBFXFAch9yHFJR9SJPwISfxzNEsH2QfoUyz5As0vmpBE9eXn7JIMUBHwwIREnI%2BMzr%2FamGE9FKO8Dq%2F%2FcZO4LgmZzekshL3WMJSwtElh%2BKKAVl5rcaFZ2uDduUgbHG9ApdK83UT1R%2F3JyvVzxaSPojplvgUpkOYi6B8iQ0yyc6AsDZC%2FXgqnEHToCfRAffKbxvlb9M98sqN02rCuJ6AcUk8XzS21bCJTUvOWJCBhT%2B%2FXGGOdS1nffmP%2Fto7UF0fz4AN0V%2F8%2FkOK2LKjTHiSCR5assbLKccqcFYAttxYTFAFqYaZrcFkvCyBGP6ZbY90T3NuVpsvWaZUq2O2UU53VgGHVVnqFhJWmt9GyPwHFv1LIsLkYo8ZzDroabMBjqZAei%2FxlzRBA0dkjveZmnU13lsN2JZvWZP7%2BDcunDusOrMtbCI8t1gJ1jqJG64sP1u3dKp2Ez2qqHixZKGo2%2BNBCFeykq53l20Q%2BgVekA1mIuSnikcjIRVytzJD3G%2FiTkMpcDP%2F0tdAjGinDO7ivp9OXz86vTlPulgtu7h%2FJxo631gy%2BqAZojF6RFrrcnm0z%2F0yUyKNrbJKMjriw%3D%3D&Expires=1770628000) - from typing import List, Dict, Any

import json

import os

from tools import repo_scanner, code_sea...

8. [query_router.py](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/58908120/72bf8877-4644-4116-96df-238a7969bbc9/query_router.py?AWSAccessKeyId=ASIA2F3EMEYERORSWIQR&Signature=lDkM64KAn8FHbIyEk7ZYvk0W9EI%3D&x-amz-security-token=IQoJb3JpZ2luX2VjEMD%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJGMEQCIHKdXphv%2BMmkknx1a4PCN%2FpEthWMxNgB19%2BpO9b%2F9tJtAiAPmeVjU59oX7mmNfrhwjSTtyIseWQn0gf8%2FHdBdHmfQir8BAiJ%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F8BEAEaDDY5OTc1MzMwOTcwNSIMGFYlSVRDOvm11tfVKtAEyeX%2FjZFIrYPMAMDbH3P5tnooBdQ%2Br4vdConwYPzzCKGxTNG7r9kFcQWkjvSXsWC1%2F6eEoYxE46%2BWEFkd2b%2Bauv2sPyRotX3R9Pz5Ky3gbHBCdZyUK7pdOugIc4PL18JUyFxCeJL8xOo6dqtUNrUcBZ65GbTZ1o%2BLxdrV4lUaxzFRLZeFF%2B%2BiwohdNeMxA2qXuuVVwrVxDOBqKXbhj%2F4jt1jLEL2bdZheHIqWTxerFldJKtynsMYMw7h0TQuk2uVXrgA4ob8ythXu%2BY3idbjN%2FUbQ3WHiBa4dfIBpsxE%2BgllO8uE2oDShXJ6ztcWOTHziKPzYy%2F6VscCGcTvqDsB1XiDPXrTMatoxLSOCudSawXJxeIG4ceY0sBBYQCNsZwwoF9WpTpxw7IHh2p1PnI1we7%2BFckdHT%2BfDIAFBFXFAch9yHFJR9SJPwISfxzNEsH2QfoUyz5As0vmpBE9eXn7JIMUBHwwIREnI%2BMzr%2FamGE9FKO8Dq%2F%2FcZO4LgmZzekshL3WMJSwtElh%2BKKAVl5rcaFZ2uDduUgbHG9ApdK83UT1R%2F3JyvVzxaSPojplvgUpkOYi6B8iQ0yyc6AsDZC%2FXgqnEHToCfRAffKbxvlb9M98sqN02rCuJ6AcUk8XzS21bCJTUvOWJCBhT%2B%2FXGGOdS1nffmP%2Fto7UF0fz4AN0V%2F8%2FkOK2LKjTHiSCR5assbLKccqcFYAttxYTFAFqYaZrcFkvCyBGP6ZbY90T3NuVpsvWaZUq2O2UU53VgGHVVnqFhJWmt9GyPwHFv1LIsLkYo8ZzDroabMBjqZAei%2FxlzRBA0dkjveZmnU13lsN2JZvWZP7%2BDcunDusOrMtbCI8t1gJ1jqJG64sP1u3dKp2Ez2qqHixZKGo2%2BNBCFeykq53l20Q%2BgVekA1mIuSnikcjIRVytzJD3G%2FiTkMpcDP%2F0tdAjGinDO7ivp9OXz86vTlPulgtu7h%2FJxo631gy%2BqAZojF6RFrrcnm0z%2F0yUyKNrbJKMjriw%3D%3D&Expires=1770628000) - import re

from typing import Literal

from enum import Enum

class QueryType(Enum):

"""Types of qu...

16. [cli.py](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/58908120/69859644-91e8-45f5-a73d-4815dc0300db/cli.py?AWSAccessKeyId=ASIA2F3EMEYERORSWIQR&Signature=C8NOQ4ii7paCmepJ7WYaCg%2BGbZE%3D&x-amz-security-token=IQoJb3JpZ2luX2VjEMD%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJGMEQCIHKdXphv%2BMmkknx1a4PCN%2FpEthWMxNgB19%2BpO9b%2F9tJtAiAPmeVjU59oX7mmNfrhwjSTtyIseWQn0gf8%2FHdBdHmfQir8BAiJ%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F8BEAEaDDY5OTc1MzMwOTcwNSIMGFYlSVRDOvm11tfVKtAEyeX%2FjZFIrYPMAMDbH3P5tnooBdQ%2Br4vdConwYPzzCKGxTNG7r9kFcQWkjvSXsWC1%2F6eEoYxE46%2BWEFkd2b%2Bauv2sPyRotX3R9Pz5Ky3gbHBCdZyUK7pdOugIc4PL18JUyFxCeJL8xOo6dqtUNrUcBZ65GbTZ1o%2BLxdrV4lUaxzFRLZeFF%2B%2BiwohdNeMxA2qXuuVVwrVxDOBqKXbhj%2F4jt1jLEL2bdZheHIqWTxerFldJKtynsMYMw7h0TQuk2uVXrgA4ob8ythXu%2BY3idbjN%2FUbQ3WHiBa4dfIBpsxE%2BgllO8uE2oDShXJ6ztcWOTHziKPzYy%2F6VscCGcTvqDsB1XiDPXrTMatoxLSOCudSawXJxeIG4ceY0sBBYQCNsZwwoF9WpTpxw7IHh2p1PnI1we7%2BFckdHT%2BfDIAFBFXFAch9yHFJR9SJPwISfxzNEsH2QfoUyz5As0vmpBE9eXn7JIMUBHwwIREnI%2BMzr%2FamGE9FKO8Dq%2F%2FcZO4LgmZzekshL3WMJSwtElh%2BKKAVl5rcaFZ2uDduUgbHG9ApdK83UT1R%2F3JyvVzxaSPojplvgUpkOYi6B8iQ0yyc6AsDZC%2FXgqnEHToCfRAffKbxvlb9M98sqN02rCuJ6AcUk8XzS21bCJTUvOWJCBhT%2B%2FXGGOdS1nffmP%2Fto7UF0fz4AN0V%2F8%2FkOK2LKjTHiSCR5assbLKccqcFYAttxYTFAFqYaZrcFkvCyBGP6ZbY90T3NuVpsvWaZUq2O2UU53VgGHVVnqFhJWmt9GyPwHFv1LIsLkYo8ZzDroabMBjqZAei%2FxlzRBA0dkjveZmnU13lsN2JZvWZP7%2BDcunDusOrMtbCI8t1gJ1jqJG64sP1u3dKp2Ez2qqHixZKGo2%2BNBCFeykq53l20Q%2BgVekA1mIuSnikcjIRVytzJD3G%2FiTkMpcDP%2F0tdAjGinDO7ivp9OXz86vTlPulgtu7h%2FJxo631gy%2BqAZojF6RFrrcnm0z%2F0yUyKNrbJKMjriw%3D%3D&Expires=1770628000) - import argparse

import os

import sys

from agent.planner import Planner

from agent.executor impor...

18. [config.py](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/58908120/ab630554-03a6-4857-a889-e770bc280e1f/config.py?AWSAccessKeyId=ASIA2F3EMEYERORSWIQR&Signature=tbSIcDw8dkV43QIfGE5y5SQbQfg%3D&x-amz-security-token=IQoJb3JpZ2luX2VjEMD%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJGMEQCIHKdXphv%2BMmkknx1a4PCN%2FpEthWMxNgB19%2BpO9b%2F9tJtAiAPmeVjU59oX7mmNfrhwjSTtyIseWQn0gf8%2FHdBdHmfQir8BAiJ%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F8BEAEaDDY5OTc1MzMwOTcwNSIMGFYlSVRDOvm11tfVKtAEyeX%2FjZFIrYPMAMDbH3P5tnooBdQ%2Br4vdConwYPzzCKGxTNG7r9kFcQWkjvSXsWC1%2F6eEoYxE46%2BWEFkd2b%2Bauv2sPyRotX3R9Pz5Ky3gbHBCdZyUK7pdOugIc4PL18JUyFxCeJL8xOo6dqtUNrUcBZ65GbTZ1o%2BLxdrV4lUaxzFRLZeFF%2B%2BiwohdNeMxA2qXuuVVwrVxDOBqKXbhj%2F4jt1jLEL2bdZheHIqWTxerFldJKtynsMYMw7h0TQuk2uVXrgA4ob8ythXu%2BY3idbjN%2FUbQ3WHiBa4dfIBpsxE%2BgllO8uE2oDShXJ6ztcWOTHziKPzYy%2F6VscCGcTvqDsB1XiDPXrTMatoxLSOCudSawXJxeIG4ceY0sBBYQCNsZwwoF9WpTpxw7IHh2p1PnI1we7%2BFckdHT%2BfDIAFBFXFAch9yHFJR9SJPwISfxzNEsH2QfoUyz5As0vmpBE9eXn7JIMUBHwwIREnI%2BMzr%2FamGE9FKO8Dq%2F%2FcZO4LgmZzekshL3WMJSwtElh%2BKKAVl5rcaFZ2uDduUgbHG9ApdK83UT1R%2F3JyvVzxaSPojplvgUpkOYi6B8iQ0yyc6AsDZC%2FXgqnEHToCfRAffKbxvlb9M98sqN02rCuJ6AcUk8XzS21bCJTUvOWJCBhT%2B%2FXGGOdS1nffmP%2Fto7UF0fz4AN0V%2F8%2FkOK2LKjTHiSCR5assbLKccqcFYAttxYTFAFqYaZrcFkvCyBGP6ZbY90T3NuVpsvWaZUq2O2UU53VgGHVVnqFhJWmt9GyPwHFv1LIsLkYo8ZzDroabMBjqZAei%2FxlzRBA0dkjveZmnU13lsN2JZvWZP7%2BDcunDusOrMtbCI8t1gJ1jqJG64sP1u3dKp2Ez2qqHixZKGo2%2BNBCFeykq53l20Q%2BgVekA1mIuSnikcjIRVytzJD3G%2FiTkMpcDP%2F0tdAjGinDO7ivp9OXz86vTlPulgtu7h%2FJxo631gy%2BqAZojF6RFrrcnm0z%2F0yUyKNrbJKMjriw%3D%3D&Expires=1770628000) - import os

from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY...

19. [Context Window Management in Agentic Systems - jroddev](https://blog.jroddev.com/context-window-management-in-agentic-systems/) - LLMs have a fixed context window (token limit) that constrains how much conversation history, retrie...

22. [LLM context windows: what they are & how they work - Redis](https://redis.io/blog/llm-context-windows/) - Context windows determine how much text your LLM can process at once. Learn what they are, why they ...

29. [MemGPT – LLMs with self-editing memory for unbounded context](https://news.ycombinator.com/item?id=37901902) - A post-processor takes over, automatically generating memories from the conversation and saving them...

31. [Conversational Memory for LLMs with Langchain - Pinecone](https://www.pinecone.io/learn/series/langchain/langchain-conversational-memory/) - ' The LLM can clearly remember the history of the conversation. Let's take a look at how this conver...

34. [MemGPT](https://research.memgpt.ai) - A system that intelligently manages different storage tiers in order to effectively provide extended...

36. [Effective context engineering for AI agents - Anthropic](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) - Context engineering refers to the set of strategies for curating and maintaining the optimal set of ...

37. [MemGPT: Towards LLMs as Operating Systems - Leonie Monigatti](https://www.leoniemonigatti.com/papers/memgpt.html) - The MemGPT agent design pattern has a two tier memory architecture which differentiates between two ...

39. [Evaluating context compression in AI agents - Tessl](https://tessl.io/blog/factory-publishes-framework-for-evaluating-context-compression-in-ai-agents/) - Context compression, for the uninitiated, refers to techniques that reduce the amount of prior inter...

42. [Context Engineering for AI Agents: Part 2 - Philschmid](https://www.philschmid.de/context-engineering-part-2) - Context Engineering is the discipline of designing a system that provides the right information and ...

