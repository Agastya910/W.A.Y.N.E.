# OPTIMIZATION GUIDE for RepoPilot

## Current Performance
- Embeddings: 30% GPU utilization (underutilized)
- Response time: ~5 minutes for medium repos
- Bottleneck: CPU-bound embedding and LLM inference

## Safe GPU Optimizations (Non-Breaking)

### 1. Force Ollama to Use GPU
Already should be on GPU by default, but verify:

```bash
# Check if Ollama is using GPU
curl http://localhost:11434/api/tags | jq .

# If models show "load_duration", check logs
grep "GPU" ~/.ollama/logs/server.log | tail -20
```

If using CPU, explicitly load on GPU:

```bash
# Stop Ollama
sudo systemctl stop ollama

# Edit config (create if missing)
sudo nano /etc/systemd/system/ollama.service.d/override.conf
```

Add:
```ini
[Service]
Environment="CUDA_VISIBLE_DEVICES=0"
Environment="OLLAMA_NUM_PARALLEL=4"
```

Then:
```bash
sudo systemctl daemon-reload
sudo systemctl start ollama
```

### 2. Batch Embeddings (Minimal Code Change)

In `core/indexer.py`, change embedding batch size:

```python
# Current (single embedding at a time - SLOW)
self.embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url="http://localhost:11434"
)

# Change to:
self.embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url="http://localhost:11434",
    show_progress=True  # Visual feedback
)
```

The FAISS library already batches behind the scenes, but you can add explicit batching:

```python
# In build_index(), replace FAISS.from_texts with:
def embed_with_batching(texts, batch_size=32):
    """Embed texts in batches for better GPU utilization."""
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        batch_embeddings = self.embeddings.embed_documents(batch)
        all_embeddings.extend(batch_embeddings)
    return all_embeddings

# Then use:
embeddings = embed_with_batching(texts, batch_size=32)
self.vector_store = FAISS.from_embeddings(
    text_embeddings=list(zip(texts, embeddings)),
    embedding=self.embeddings,
    metadatas=metadatas
)
```

### 3. Use Faster Embedding Model

Current: `nomic-embed-text` (good quality, slower)  
Options:
- `all-minilm` - 40x faster, 90% as good
- `bge-small-en-v1.5` - 30x faster

```bash
# Pull lighter model
ollama pull all-minilm

# Update indexer.py line 32:
self.embeddings = OllamaEmbeddings(
    model="all-minilm",  # Changed
    base_url="http://localhost:11434"
)
```

### 4. Parallel Inference

In `llm/local_llm_client.py`, enable parallel requests (if you have multi-GPU):

```python
# Currently sequential, can parallelize searches:
import asyncio

async def async_search(query):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, self.indexer.search, query)

# But this requires async refactor - SKIP for now
```

### 5. Pre-cache Embeddings

The first embedding run takes time; subsequent runs load from disk instantly.  
Make sure users understand: **First indexing is slow, then instant.**

Document in README:
```markdown
## Performance

**First-time indexing**: 3-10 minutes (one-time cost)
**Subsequent queries**: <30 seconds (embeddings cached)

For large repos:
- Go/React: ~3 min
- Linux Kernel: ~15 min (92k files)
```

## Recommended Action (Safest)

1. **Switch embedding model** → `all-minilm` (fastest safe change)
2. **Force GPU in Ollama config** → Already likely using it
3. **Add batching** → Marginal improvement
4. **Document expectations** → Users know it's normal

## What NOT to Do

❌ Quantize model aggressively (drops accuracy)  
❌ Reduce chunk size dramatically (loses context)  
❌ Disable embeddings (breaks search)  
❌ Run multiple Ollama instances (conflicts)

## Testing After Changes

```bash
cd ~/RepoPilot
rm -rf .repopilot_index ./analyzed_repo
python cli.py .

# Time first query (will index)
time python -c "
from core.indexer import CodeIndexer
i = CodeIndexer('.')
"

# Time subsequent query (cached)
time python -c "
from core.indexer import CodeIndexer
i = CodeIndexer('.')
results = i.search('authentication')
"
```

Expected:
- Before: 5 minutes
- After (all-minilm): 2-3 minutes
- With GPU properly configured: 1-2 minutes
