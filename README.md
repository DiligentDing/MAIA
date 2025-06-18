# MAIA Benchmark

**MAIA** (*Medical Agent Intelligence Assessment*) gauges how well an autonomous medical agent can **plan**, **call external tools**, and **reason clinically**.\
All items follow a unified schema so that an LLM‑based agent can decide *whether*, *when*, and *how* to invoke the provided APIs.

## Composition

| Task family        | Items   | Evaluated skill                                                       |
| ------------------ | ------- | --------------------------------------------------------------------- |
| Retrieval          | **100** | Retrieve clinically relevant information from trusted medical sources |
| KG Reasoning       | **466** | Multi‑hop reasoning abilities in medical knowledge‑graph settings     |
| Diagnostic Pathway | **448** | Reasoning capabilities in authentic clinical scenarios                |

**Total items: 1 014** (single *full* split).

## Data format

```jsonc
{
  "id": "ret_cacfe0e74802",
  "question": "What is the PMID of …?",
  "tool_calls": [
    {
      "tool": "pubmed.search",
      "params": { "term": "...", "retmax": 1 }
    }
  ],
  "answer": ["40360142"],
  "type": "retrieval"
}
```

---

## Access via Hugging Face

The full dataset is hosted on **Hugging Face Datasets**: [https://huggingface.co/datasets/maia-benchmark/maia](https://huggingface.co/datasets/maia-benchmark/maia)\
*(Replace this URL with your actual repository if different.)*

### Quick load with `datasets`

```python
from datasets import load_dataset

ds = load_dataset("maia-benchmark/maia", split="full")  # loads the entire benchmark
print(ds[0])
```

### Git‑based download (with LFS)

```bash
# Ensure Git‑LFS is installed
git lfs install

git clone https://huggingface.co/datasets/maia-benchmark/maia
```

---



