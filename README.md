# MAIA Benchmark

**MAIA** evaluates how well an autonomous medical agent can **plan**, **call external tools**, and **reason clinically**.\
All items follow a unified schema so that an LLM‑based agent can decide *whether*, *when*, and *how* to invoke the provided APIs.

## Composition

| Task family        | Items   | Evaluated skill                                                       |
| ------------------ | ------- | --------------------------------------------------------------------- |
| Retrieval          | **471** | Retrieve clinically relevant information from trusted medical sources |
| KG Reasoning       | **2068** | Multi‑hop reasoning abilities in medical knowledge‑graph settings     |
| Clinical Pathway | **1937** | Reasoning capabilities in authentic clinical scenarios                |

**Total items: 4476** (single *full* split).

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

The full dataset is hosted on **Hugging Face Datasets**: [https://huggingface.co/datasets/DiligentDing/MAIA](https://huggingface.co/datasets/DiligentDing/MAIA)

### Quick load with `datasets`

```python
from datasets import load_dataset

ds = load_dataset("DiligentDing/MAIA", split="full")  # loads the entire benchmark
print(ds[0])
```




