# adjustment normalisasi skor pada dataset

import json

soal = json.load(open("./soal_ujian.json"))
nilai = json.load(open("./nilai_ujian.json"))

soal_by_id = {s["id"]: s for s in soal}

def normalize_scores(nilai_dict):
    # if max value <= 10, it's on a 0-10 scale -> rescale to 0-100
    if max(nilai_dict.values()) <= 10:
        return {k: v * 10 for k, v in nilai_dict.items()}
    return nilai_dict

dataset = []
skipped = 0

for n in nilai:
    q = soal_by_id.get(n["id_soal"])
    if q is None:
        skipped += 1
        continue

    scores = normalize_scores(n["nilai"])
    # recompute avg to stay consistent with normalized scale
    avg = round(sum(scores.values()) / len(scores), 2)

    dataset.append({
        "soal": q["soal"],
        "expected_output": q["expected_output"],
        "kode_siswa": n["kode_siswa"],
        "level_siswa": n["level_siswa"],
        "nilai": scores,
        "nilai_avg": avg,
        "feedback": n["feedback"],
    })

print(f"Usable examples: {len(dataset)} | Skipped (no matching question): {skipped}")