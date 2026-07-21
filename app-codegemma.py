from ollama import chat

RUBRIC = """SCORING RUBRIC
(use integer multiples of 5 ONLY: 0,5,10,...,100)

FUNCTIONALITY : 100=perfect output; 80=correct main case; 60=core runs incomplete; 40=wrong but basic understanding; 0=crashes.
CODE_STYLE    : 100=perfect PEP8; 80=minor style issues; 60=readable but errors; 40=messy; 0=unreadable.
DOCUMENTATION : 100=full docstrings+comments; 80=most code documented; 60=basic comments; 40=1-2 comments; 0=none.
LOGIC         : 100=optimal and elegant; 80=correct but improvable; 60=mostly correct with flaws; 40=partially correct; 0=no logical structure.
SYNTAX        : 100=perfect idiomatic Python; 80=correct not fully idiomatic; 60=small non-breaking errors; 40=1-2 errors; 0=cannot parse.
CONCEPT       : 100=best concept applied; 80=correct but improvable; 60=runs but not ideal; 40=misused; 0=none.

PROFICIENCY (for feedback model only — do NOT output this in scoring):
  Expert          : avg >= 90, nothing below 85
  Competent       : avg >= 75, nothing below 65
  Advance         : avg >= 60, nothing below 50
  Advance Beginner: avg >= 45, nothing below 35
  Beginner        : avg >= 30, nothing below 20
  Novice          : avg <  30 or fundamental errors
"""

def build_feedback_prompt(soal: str, code: str, expected_out: str,
                           simulated_in: str, scores: dict) -> str:
    """
    Feedback prompt for CodeGemma.
    Receives the scores already computed by CodeLlama, adds narrative.
    Output: feedback, proficiency, reasoning, per-rubric critique.

    Args:
        soal: the problem statement
        code: the student's submitted code
        expected_out: the expected program output
        simulated_in: simulated stdin used when running the code
        scores: dict of per-rubric scores computed upstream, e.g.
                {"functionality": 80, "code_style": 80, "documentation": 60,
                 "logic": 80, "syntax": 90, "concept": 80}
    """
    scores_summary = "\n".join(
        f"  {k}: {v}" for k, v in scores.items()
    )
    return f"""You are a friendly, encouraging Python programming tutor providing detailed feedback.

### PROBLEM:
{soal}

### EXPECTED OUTPUT:
{expected_out if expected_out else "(not specified)"}

### SIMULATED INPUT:
{simulated_in if simulated_in else "(none)"}

### STUDENT CODE:
```python
{code}
```

{RUBRIC}

### COMPUTED SCORES:
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


soal = 'Buatlah program Python untuk mencari nama seorang murid pada daftar nama menggunakan Linear Search dan Binary Search. Bandingkan efisiensi kedua algoritma dan jelaskan kapan masing-masing lebih tepat digunakan.'
code = '''
def linear_search(names, target):
    comparisons = 0
    for i, name in enumerate(names):
        comparisons += 1
        if name == target:
            return i, comparisons
    return -1, comparisons

def binary_search(names, target):
    names_sorted = sorted(names)
    low, high = 0, len(names_sorted) - 1
    comparisons = 0
    while low <= high:
        mid = (low + high) // 2
        comparisons += 1
        if names_sorted[mid] == target:
            return mid, comparisons
        elif names_sorted[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
    return -1, comparisons

names = ['Andi', 'Budi', 'Citra', 'Dewi', 'Eka']
target = 'Citra'

ls_idx, ls_cmp = linear_search(names, target)
bs_idx, bs_cmp = binary_search(sorted(names), target)

print(f"Linear Search: ditemukan di indeks {ls_idx} ({ls_cmp} kali perbandingan)")
print(f"Binary Search: ditemukan di indeks {bs_idx} ({bs_cmp} kali perbandingan)")
print("Linear Search cocok untuk data kecil/tidak terurut.")
print("Binary Search lebih efisien untuk data besar yang sudah terurut.")
'''
expected_out = '''
Linear Search: ditemukan di indeks 2 (3 kali perbandingan)
Binary Search: ditemukan di indeks 2 (2 kali perbandingan)
Linear Search cocok untuk data kecil/tidak terurut.
Binary Search lebih efisien untuk data besar yang sudah terurut.
'''
simulated_in = ''

# PLACEHOLDER ONLY karena seharusnya ini dapet dari CODELLAMA-7B
scores = {
    "functionality": 100,
    "code_style": 95,
    "documentation": 80,
    "logic": 95,
    "syntax": 95,
    "concept": 95,
    "overall_score": 93,
}

prompt = build_feedback_prompt(soal, code, expected_out, simulated_in, scores)

response = chat(
    model='codegemma:7b-instruct-q3_K_M',
    messages=[{'role': 'user', 'content': prompt}],
)
print(response.message.content)