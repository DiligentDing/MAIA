from __future__ import annotations

import argparse
import json
import pathlib
import time
import re
from typing import Any, Dict, Iterable, List, Optional

try:
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover - optional at import time
    OpenAI = None  # type: ignore

try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover - allow --help without tqdm installed
    def tqdm(x, **kwargs):
        return x


def load_items(path: pathlib.Path) -> List[Dict[str, Any]]:
    """Load MAIA items from a JSON file.

    Accepts either a list of QA dicts or a dict with key 'dataset'.
    """
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "dataset" in data:
        return data["dataset"]
    if isinstance(data, list):
        return data
    raise ValueError("Unsupported dataset format; expected list or {'dataset': [...]}")


def ensure_dir(p: pathlib.Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def save_json(obj: Any, path: pathlib.Path) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def generate_answers(
    items: List[Dict[str, Any]],
    client: Any,
    model: str,
    temperature: float,
    rate_limit_s: float,
    out_path: pathlib.Path,
    start: int = 0,
    end: int | None = None,
) -> Dict[str, str]:
    """Generate model answers for the given slice of items.

    Returns a dict mapping string index to answer text and persists to out_path.
    """
    # resume if exists
    if out_path.exists():
        try:
            model_answers: Dict[str, str] = json.loads(out_path.read_text(encoding="utf-8"))
        except Exception:
            model_answers = {}
    else:
        model_answers = {}

    rng = range(start, len(items) if end is None else min(end, len(items)))
    for idx in tqdm(rng, desc="Generating answers"):
        if str(idx) in model_answers:
            continue

        q = items[idx]["question"]
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an experienced oncologist answering exam-style "
                            "clinical questions concisely and accurately."
                        ),
                    },
                    {"role": "user", "content": q},
                ],
            )
            model_answers[str(idx)] = resp.choices[0].message.content.strip()
            time.sleep(rate_limit_s)
        except Exception as e:
            print(f"[Responder error @ {idx}] {e}. Retrying in 10 s…")
            time.sleep(10)
            continue

        if idx % 10 == 9:
            save_json(model_answers, out_path)

    save_json(model_answers, out_path)
    return model_answers


JUDGE_PROMPT_TPL = """You are an impartial medical board examiner.
Score the model answer against the reference answer on a 0–5 scale.
If unsure between two scores, pick the lower one.

Rubric:
5 = Covers all key clinical facts; any extra content is correct & relevant.
4 = ≥90% key facts correct; at most one minor omission or wording issue.
3 = 70–89% key facts covered; may include a few minor errors; none unsafe.
2 = 40–69% key facts or ≥1 moderate error/omission; some irrelevant content.
1 = <40% key facts or major inaccuracies; mostly irrelevant or confusing.
0 = Blank, nonsense, or clearly unsafe recommendation.

Penalty rules:
• Extra correct & relevant content → no penalty.
• Extra irrelevant or wrong content → lower the score.
• Any unsafe or potentially harmful statement → max score = 1.

Return exactly one line:
"<score 0-5>: <concise 1–2 sentence justification>"

Question:
{question}

Reference answer:
{ref_answer}

Model answer:
{model_answer}
"""


def judge_answers(
    items: List[Dict[str, Any]],
    client: Any,
    model: str,
    rate_limit_s: float,
    answers: Dict[str, str],
    out_path: pathlib.Path,
    start: int = 0,
    end: int | None = None,
) -> Dict[str, Dict[str, Any]]:
    """Score model answers and persist to out_path.

    Returns a dict mapping index str to {score, explanation}.
    """
    if out_path.exists():
        try:
            judge_scores: Dict[str, Dict[str, Any]] = json.loads(out_path.read_text(encoding="utf-8"))
        except Exception:
            judge_scores = {}
    else:
        judge_scores = {}

    rng = range(start, len(items) if end is None else min(end, len(items)))
    for idx in tqdm(rng, desc="Judging answers"):
        if str(idx) in judge_scores:
            continue

        ref_answer = items[idx].get("answer", "")
        if isinstance(ref_answer, list):
            ref_answer_str = ", ".join(map(str, ref_answer))
        else:
            ref_answer_str = str(ref_answer)

        prompt = JUDGE_PROMPT_TPL.format(
            question=items[idx]["question"],
            ref_answer=ref_answer_str,
            model_answer=answers.get(str(idx), ""),
        )

        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=0,
                max_tokens=120,
                messages=[
                    {"role": "system", "content": "You are an expert clinical examiner."},
                    {"role": "user", "content": prompt},
                ],
            )
            raw = resp.choices[0].message.content.strip()

            m = re.search(r"\b([0-5](?:\.\d+)?)\b", raw)
            if not m:
                print(f"[Judge format error @ {idx}] {raw}")
                continue

            score = float(m.group(1))
            judge_scores[str(idx)] = {"score": score, "explanation": raw}
            time.sleep(rate_limit_s)
        except Exception as e:
            print(f"[Judge error @ {idx}] {e}. Retrying in 5 s…")
            time.sleep(5)
            continue

        if idx % 10 == 9:
            save_json(judge_scores, out_path)

    save_json(judge_scores, out_path)
    return judge_scores


def mean_score(scores: Dict[str, Dict[str, Any]]) -> float:
    vals = [v["score"] for v in scores.values() if isinstance(v, dict) and "score" in v]
    return sum(vals) / len(vals) if vals else 0.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate MAIA model answers with an LLM judge.")
    p.add_argument("--input", default="dataset/MAIA.json", help="Path to dataset JSON")
    p.add_argument("--outdir", default="./res", help="Output directory for results")
    p.add_argument("--responder-model", default="answer_model", help="Model name for answer generation")
    p.add_argument("--judge-model", default="judge_model", help="Model name for judging")
    p.add_argument("--temperature", type=float, default=0.1, help="Temperature for responder model")
    p.add_argument("--rate-limit-s", type=float, default=1.0, help="Delay between API calls in seconds")
    p.add_argument("--start", type=int, default=0, help="Start index (inclusive)")
    p.add_argument("--end", type=int, default=None, help="End index (exclusive)")
    p.add_argument("--skip-generate", action="store_true", help="Skip answer generation phase")
    p.add_argument("--skip-judge", action="store_true", help="Skip judge phase")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    in_path = pathlib.Path(args.input)
    out_dir = pathlib.Path(args.outdir)
    ensure_dir(out_dir)

    items = load_items(in_path)
    if OpenAI is None:
        raise RuntimeError("openai package not installed. Please `pip install openai`.")
    client = OpenAI()

    answers_path = out_dir / "model_answers.json"
    scores_path = out_dir / "judge_scores.json"

    if not args.skip_generate:
        answers = generate_answers(
            items=items,
            client=client,
            model=args.responder_model,
            temperature=args.temperature,
            rate_limit_s=args.rate_limit_s,
            out_path=answers_path,
            start=args.start,
            end=args.end,
        )
    else:
        answers = json.loads(answers_path.read_text(encoding="utf-8")) if answers_path.exists() else {}

    if not args.skip_judge:
        scores = judge_answers(
            items=items,
            client=client,
            model=args.judge_model,
            rate_limit_s=args.rate_limit_s,
            answers=answers,
            out_path=scores_path,
            start=args.start,
            end=args.end,
        )
        print(f"Average score: {mean_score(scores):.3f} over {len(scores)} items")
    else:
        print("Judge phase skipped.")


if __name__ == "__main__":
    main()
