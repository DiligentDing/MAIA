import os, json, time, pathlib, tqdm
from collections import defaultdict
from openai import OpenAI

RESPONDER_MODEL = "answer_model"  # 替换为实际的模型名称      
JUDGE_MODEL     ="judge_model"     
TEMP            = 0.1              
RATE_LIMIT_S    = 1      
out_dir = pathlib.Path("../res")
out_dir.mkdir(exist_ok=True)         

# ========= 1. 加载数据 =========
with open("dataset/MAIA.json", "r", encoding="utf-8") as f:
    qa_pairs = json.load(f)

# ========= 2. 生成模型答案 =========
answers_path = out_dir / "model_answers.json"
if answers_path.exists():
    model_answers = json.load(open(answers_path))
else:
    model_answers = {}

for idx, qa in enumerate(tqdm.tqdm(qa_pairs['dataset'], desc="Generating answers")):
    q = qa["question"]
    if str(idx) in model_answers:
        continue  

    try:
        client = OpenAI()
        resp = client.chat.completions.create(
            model=RESPONDER_MODEL,
            messages=[
                {"role": "system", "content": "You are an experienced oncologist answering exam-style clinical questions concisely and accurately."},
                {"role": "user",    "content": q}
            ],
        )
        model_answers[str(idx)] = resp.choices[0].message.content.strip()
        time.sleep(RATE_LIMIT_S)
    except Exception as e:
        print(f"[Responder error @ {idx}] {e}. Retrying in 10 s…")
        time.sleep(10)
        continue

    
    if idx % 10 == 9:
        json.dump(model_answers, open(answers_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)


json.dump(model_answers, open(answers_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
# ========= 3. 调用 Judge 评分 =========
import re
scores_path = out_dir / "judge_scores.json"
if scores_path.exists():
    judge_scores = json.load(open(scores_path))
else:
    judge_scores = {}

judge_prompt_tpl = """You are an impartial medical board examiner.
Score the model answer against the reference answer on a 0–5 scale,
using the *refined* rubric below.  If unsure between two scores, pick
the **lower** one.

Rubric:
5 = Covers **all** key clinical facts in the reference; any additional
    explanations are factually correct *and clinically relevant*; no
    inaccuracies, unsafe statements, or major omissions.
4 = ≥90 % of key facts correct; extra content is correct; at most one
    minor omission **or** wording inaccuracy that does not alter meaning.
3 = 70-89 % of key facts covered; may include a few minor errors or
    omissions, but no clinically dangerous advice.
2 = 40-69 % of key facts **or** ≥1 moderate factual error/omission; some
    irrelevant or redundant statements allowed.
1 = <40 % of key facts **or** major inaccuracies; content mostly
    irrelevant or confusing.
0 = Blank, nonsense, or any clearly unsafe recommendation.

Penalty rules:
• Extra content that is factually correct & relevant → **no penalty**.
• Extra but irrelevant OR factually wrong content → lower the score.
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

for idx, qa in enumerate(tqdm.tqdm(qa_pairs['dataset'], desc="Judging answers")):
    if str(idx) in judge_scores:
        continue

    prompt = judge_prompt_tpl.format(
        question=qa["question"],
        ref_answer=qa["answer"],
        model_answer=model_answers.get(str(idx), "")
    )

    try:

        resp = client.chat.completions.create(
            model=JUDGE_MODEL,
            temperature=0,
            max_tokens=120,
            messages=[
                {"role": "system", "content": "You are an expert clinical examiner."},
                {"role": "user",    "content": prompt}
            ],
        )

        raw = resp.choices[0].message.content.strip()

        m = re.search(r"\b([0-5](?:\.\d+)?)\b", raw)
        if not m:
            print(f"[Judge format error @ {idx}] {raw}")
            continue                      # 或者 retry

        score = float(m.group(1))
        judge_scores[str(idx)] = {"score": score, "explanation": raw}
        time.sleep(RATE_LIMIT_S)
    except Exception as e:
        print(f"[Judge error @ {idx}] {e}. Retrying in 5 s…")
        time.sleep(5)
        continue

    if idx % 10 == 9:
        json.dump(judge_scores, open(scores_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

json.dump(judge_scores, open(scores_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
