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
  "id": "kg_474ba465454f",
  "question": "A 54-year-old woman with chronic hepatitis C develops laboratory evidence of reduced platelet count on routine screening. Her physician initiates a biologic therapy that triggers the JAK-STAT pathway after binding to a specific cell-surface receptor complex. Which molecular complex is directly responsible for mediating the cellular immunomodulatory effects of this therapy in her case?",
  "tool_calls": [
    {
      "tool": "umls.concept_lookup",
      "params": { "name": "Thrombocytopenia, unspecified" }
    },
    {
      "tool": "umls.get_related",
      "params": { "from_cui": "C0040034", "rela": "may_be_treated_by" }
    },
    {
      "tool": "umls.get_related",
      "params": { "from_cui": "C0021735", "rela": "has_target" }
    }
  ],
  "answer": "Interferon alpha receptor complex",
  "type": "kg_reasoning",
  "source": "UMLS",
  "reasoning_path": "Recombinant interferon alfa-2b binds to the interferon alpha receptor complex, activating the JAK-STAT pathway and modulating immune and hematopoietic cell function.",
  "reasoning_depth": 3
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


## Local installation and usage

1) Install dependencies (use a virtualenv if possible):

```bash
pip install -r requirements.txt
```

2) Prepare API keys and optional databases:

- OpenAI API: set environment variable `OPENAI_API_KEY`.
- Optional UMLS MySQL for UMLS tools: set `UMLS_DB_HOST`, `UMLS_DB_PORT`, `UMLS_DB_USER`, `UMLS_DB_PASSWORD`, `UMLS_DB_NAME`.

3) Run evaluation over the bundled JSON or a custom slice:

```bash
python eval.py \
  --input dataset/MAIA.json \
  --outdir ./res \
  --responder-model gpt-4o-mini \
  --judge-model gpt-4o-mini \
  --temperature 0.1 \
  --rate-limit-s 1
```

Useful flags:

- `--start/--end`: evaluate a slice for quick smoke tests.
- `--skip-generate` or `--skip-judge`: run one phase only.

Outputs:

- `res/model_answers.json` – generated answers keyed by index.
- `res/judge_scores.json` – judge scores and brief rationale.

Notes:

- The judge expects the reference answer to be a string or list of strings in the dataset.
- If using UMLS tools in `tools/impl.py`, ensure DB connectivity; the connection is created lazily when the first UMLS function is invoked.




