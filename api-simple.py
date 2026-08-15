import os
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

app = Flask(__name__)
CORS(app)  # allow the standalone index.html to call this API


# ── Helpers ───────────────────────────────────────────────────────────────────
def _ensure_v1(host: str) -> str: 
    host = host.rstrip("/")
    if not host.endswith("/v1"):
        host += "/v1"
    return host


# ── Default connections (can be overridden per-request) ───────────────────────
SCORING_HOST  = _ensure_v1(os.getenv("SCORING_MODEL_HOST", "http://localhost:1234"))
SCORING_MODEL = os.getenv("SCORING_MODEL_NAME", "codellama-7b-instruct")

FEEDBACK_HOST  = _ensure_v1(os.getenv("FEEDBACK_MODEL_HOST", "http://localhost:11434"))
FEEDBACK_MODEL = os.getenv("FEEDBACK_MODEL_NAME", "codegemma:7b-instruct")


# ── Scoring prompt ─────────────────────────────────────────────────────────────
RUBRIC = """
SCORING RUBRIC
(use integer multiples of 5 ONLY: 0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100)

FUNCTIONALITY:
  100=output perfect; 95=tiny edge case; 90=correct with minor weakness;
  85=mostly correct; 80=correct for main case; 75=mostly works but inconsistent;
  70=mostly correct with visible issues; 65=partially correct; 60=core runs incomplete;
  55=significant issues; 50=half requirements met; 45=some logic runs;
  40=wrong but shows basic understanding; 35=little progress; 30=runs with major errors;
  25=very limited correctness; 20=runs but mostly wrong; 15=barely functional;
  10=very serious errors; 5=almost unusable; 0=crashes or no valid output.

CODE_STYLE:
  100=perfect PEP8; 95=nearly perfect; 90=very clean; 85=good naming and format;
  80=minor style issues; 75=generally readable; 70=some issues; 65=readability suffers;
  60=understandable but errors; 55=many inconsistencies; 50=average; 45=hard to read;
  40=messy; 35=very inconsistent; 30=little readability; 25=very messy; 20=chaotic;
  15=nearly unreadable; 10=major issues; 5=almost impossible; 0=unreadable.

DOCUMENTATION:
  100=full docstrings+comments; 95=nearly complete; 90=very good; 85=mostly documented;
  80=most code documented; 75=decent but gaps; 70=present but uneven;
  65=inconsistent quality; 60=basic comments; 55=very minimal; 50=limited;
  45=very few; 40=1-2 comments; 35=very weak; 30=almost none; 25=little benefit;
  20=unclear; 15=almost none; 10=very poor; 5=fragments; 0=none.

LOGIC:
  100=optimal and elegant; 95=very good; 90=very strong; 85=correct minor gap;
  80=correct but improvable; 75=mostly correct; 70=works with weaknesses;
  65=inconsistencies; 60=mostly correct with flaws; 55=significant weakness;
  50=partial understanding; 45=many errors; 40=partially correct;
  35=weak structure; 30=minimal flow; 25=major misunderstanding;
  20=fundamental error; 15=barely makes sense; 10=almost nothing correct;
  5=random; 0=no logical structure.

SYNTAX:
  100=perfect idiomatic Python; 95=nearly flawless; 90=very good; 85=correct minor issues;
  80=correct not fully idiomatic; 75=slight inconsistencies; 70=mostly valid;
  65=some errors program runs; 60=small non-breaking errors; 55=noticeable errors;
  50=frequent weaknesses; 45=occasional runtime issues; 40=1-2 errors;
  35=many errors; 30=needs many fixes; 25=frequently fails; 20=many errors;
  15=barely runnable; 10=very serious; 5=almost impossible; 0=cannot parse.

CONCEPT:
  100=best concept applied; 95=very good choice; 90=very appropriate;
  85=correct with minor gap; 80=correct but improvable; 75=adequately applied;
  70=mostly appropriate; 65=partially fitting; 60=runs but not ideal;
  55=weak application; 50=moderate understanding; 45=confused in places;
  40=misused; 35=very weak; 30=minimal; 25=major misunderstanding;
  20=wrong concept; 15=nearly irrelevant; 10=very serious errors; 5=almost none; 0=none.

PROFICIENCY LEVEL (choose exactly one):
  Expert          : average >= 90, nothing below 85
  Competent       : average >= 75, nothing below 65
  Advance         : average >= 60, nothing below 50
  Advance Beginner: average >= 45, nothing below 35
  Beginner        : average >= 30, nothing below 20
  Novice          : average <  30 or fundamental errors present
"""


def build_scoring_prompt(soal, code, expected_out, simulated_in) -> str:
    return f"""You are a strict Python programming assessor.

### SOAL (Problem):
{soal}

### STUDENT CODE:
```python
{code}
```

### SIMULATED INPUT:
{simulated_in if simulated_in else '(none)'}

### EXPECTED OUTPUT:
{expected_out if expected_out else '(not specified)'}

{RUBRIC}

For each of the 6 aspects (functionality, code_style, documentation, logic, syntax, concept):
- Give a score (multiple of 5, 0–100) based strictly on the rubric.
- If score < 80, quote the exact line or issue and explain how to fix it.
- If score >= 80, write "All good!"

Also provide:
- overall_feedback: short encouraging summary in friendly English (max 60 words)
- inferred_proficiency: one of Novice / Beginner / Advance Beginner / Advance / Competent / Expert
- reasoning: why this proficiency level (max 30 words)

Respond with ONLY valid JSON, no markdown fences, no extra text:
{{
  "feedback": "overall encouraging summary",
  "inferred_proficiency": "level",
  "reasoning": "why this level",
  "scores": {{
    "functionality": 0,
    "code_style": 0,
    "documentation": 0,
    "logic": 0,
    "syntax": 0,
    "concept": 0,
    "overall_score": 0
  }},
  "critique": {{
    "functionality": "critique or All good!",
    "code_style": "critique or All good!",
    "documentation": "critique or All good!",
    "logic": "critique or All good!",
    "syntax": "critique or All good!",
    "concept": "critique or All good!"
  }}
}}
overall_score = average of the 6 aspect scores, rounded to nearest integer."""


def _parse_json_response(raw: str) -> dict | None:
    """Strip markdown fences then attempt JSON parse."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.startswith("json"):
            raw = raw[4:].strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def _normalize_scores(scores: dict) -> dict:
    """If all scores <= 10, auto-scale to 100."""
    skeys = ["functionality", "code_style", "documentation", "logic", "syntax", "concept"]
    values = [scores.get(k, 0) for k in skeys]
    if values and max(values) <= 10:
        for k in skeys:
            scores[k] = scores.get(k, 0) * 10
    scores["overall_score"] = round(sum(scores.get(k, 0) for k in skeys) / len(skeys))
    return scores


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "scoring_model": SCORING_MODEL,
        "feedback_model": FEEDBACK_MODEL,
    })


@app.route("/defaults", methods=["GET"])
def defaults():
    return jsonify({
        "scoring_model_host": SCORING_HOST.replace("/v1", ""),
        "scoring_model_name": SCORING_MODEL,
        "feedback_model_host": FEEDBACK_HOST.replace("/v1", ""),
        "feedback_model_name": FEEDBACK_MODEL,
        "temperature": 0.0,
        "max_tokens":  1024,
        "top_p":       1.0,
    })


@app.route("/evaluate", methods=["POST"])
def evaluate():
    """CodeLlama scoring endpoint — returns scores + critique."""
    data = request.get_json(force=True, silent=True) or {}

    soal         = data.get("soal",         "").strip()
    code         = data.get("code",         "").strip()
    expected_out = data.get("expected_out", "").strip()
    simulated_in = data.get("simulated_in", "").strip()

    if not soal or not code:
        return jsonify({"error": "Fields 'soal' and 'code' are required."}), 400

    # Allow per-request override; fall back to env defaults
    model_host  = _ensure_v1(data.get("model_host", SCORING_HOST))
    model_name  = data.get("model_name",  SCORING_MODEL)
    temperature = float(data.get("temperature", 0.0))
    max_tokens  = int(data.get("max_tokens",    1024))
    top_p       = float(data.get("top_p",       1.0))

    prompt = build_scoring_prompt(soal, code, expected_out, simulated_in)

    try:
        client = OpenAI(base_url=model_host, api_key="lm-studio")
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )
        raw = response.choices[0].message.content.strip()
        result = _parse_json_response(raw)

        if result is None:
            result = {
                "feedback": "Could not parse AI response.",
                "inferred_proficiency": "N/A",
                "reasoning": "",
                "scores": {k: 0 for k in ["functionality", "code_style", "documentation",
                                           "logic", "syntax", "concept", "overall_score"]},
                "critique": {},
                "_raw_response": raw,
            }
        else:
            if "scores" in result:
                result["scores"] = _normalize_scores(result["scores"])

        result["_meta"] = {
            "scoring_model_host": model_host,
            "scoring_model_name": model_name,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
        }
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/evaluate-feedback", methods=["POST"])
def evaluate_feedback():
    """CodeGemma feedback endpoint — uses pre-computed scores from CodeLlama."""
    data = request.get_json(force=True, silent=True) or {}

    soal         = data.get("soal",         "").strip()
    code         = data.get("code",         "").strip()
    expected_out = data.get("expected_out", "").strip()
    simulated_in = data.get("simulated_in", "").strip()
    scores       = data.get("scores",       {})

    if not soal or not code:
        return jsonify({"error": "Fields 'soal' and 'code' are required."}), 400

    # Allow per-request override; fall back to env defaults
    model_host  = _ensure_v1(data.get("model_host", FEEDBACK_HOST))
    model_name  = data.get("model_name",  FEEDBACK_MODEL)
    temperature = float(data.get("temperature", 0.0))
    max_tokens  = int(data.get("max_tokens",    1024))
    top_p       = float(data.get("top_p",       1.0))

    scores_summary = "\n".join(f"  {k}: {v}" for k, v in scores.items()) if scores \
        else "  (no scores provided — evaluate based on code only)"

    feedback_prompt = f"""You are a friendly, encouraging Python programming tutor.

### PROBLEM:
{soal}

### EXPECTED OUTPUT:
{expected_out if expected_out else '(not specified)'}

### SIMULATED INPUT:
{simulated_in if simulated_in else '(none)'}

### STUDENT CODE:
```python
{code}
```

### PRE-COMPUTED SCORES (from scoring model — do NOT change these):
{scores_summary}

Based on the scores above, provide:
1. An encouraging overall_feedback (max 70 words, friendly English or Bahasa Indonesia).
2. The inferred_proficiency level (one of: Novice / Beginner / Advance Beginner / Advance / Competent / Expert).
3. A short reasoning for that proficiency (max 30 words).
4. Per-rubric critique: for each of the 6 aspects, if score < 80 quote the exact issue and how to fix it; if score >= 80 write "All good!".

Respond with ONLY valid JSON, no markdown fences, no extra text:
{{
  "feedback": "overall encouraging summary here",
  "inferred_proficiency": "level here",
  "reasoning": "why this level",
  "critique": {{
    "functionality": "critique or All good!",
    "code_style": "critique or All good!",
    "documentation": "critique or All good!",
    "logic": "critique or All good!",
    "syntax": "critique or All good!",
    "concept": "critique or All good!"
  }}
}}"""

    try:
        client = OpenAI(base_url=model_host, api_key="lm-studio")
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": feedback_prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )
        raw = response.choices[0].message.content.strip()
        result = _parse_json_response(raw)

        if result is None:
            result = {
                "feedback": "(CodeGemma response could not be parsed as JSON)",
                "inferred_proficiency": "N/A",
                "reasoning": "",
                "critique": {},
                "_raw_response": raw,
            }

        result["_meta"] = {
            "feedback_model_host": model_host,
            "feedback_model_name": model_name,
        }
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


# ── Fine-tune sample endpoints ─────────────────────────────────────────────────
FINETUNE_PATH = os.getenv("FINETUNE_OUTPUT_PATH", "finetune_samples.jsonl")


@app.route("/samples-count", methods=["GET"])
def samples_count():
    try:
        with open(FINETUNE_PATH, encoding="utf-8") as f:
            count = sum(1 for _ in f)
        return jsonify({"count": count})
    except FileNotFoundError:
        return jsonify({"count": 0})


@app.route("/save-sample", methods=["POST"])
def save_sample():
    data = request.get_json(force=True, silent=True) or {}
    if not data:
        return jsonify({"error": "No data provided."}), 400
    with open(FINETUNE_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")
    try:
        with open(FINETUNE_PATH, encoding="utf-8") as f:
            total = sum(1 for _ in f)
    except Exception:
        total = "?"
    return jsonify({"ok": True, "total_samples": total})


@app.route("/export-samples", methods=["GET"])
def export_samples():
    from flask import send_file
    try:
        return send_file(FINETUNE_PATH, as_attachment=True,
                         download_name="finetune_samples.jsonl",
                         mimetype="application/x-ndjson")
    except FileNotFoundError:
        return jsonify({"error": "No samples collected yet."}), 404


@app.route("/clear-samples", methods=["DELETE"])
def clear_samples():
    try:
        open(FINETUNE_PATH, "w").close()
        return jsonify({"ok": True})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    port = int(os.getenv("API_PORT", 5050))
    print(f"[api-simple] Scoring  → {SCORING_HOST}  ({SCORING_MODEL})")
    print(f"[api-simple] Feedback → {FEEDBACK_HOST}  ({FEEDBACK_MODEL})")
    print(f"[api-simple] Listening on http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=True)
