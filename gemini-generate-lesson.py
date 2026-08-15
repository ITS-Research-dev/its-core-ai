GENERATE_MATERI_PROMPT = """
Anda adalah asisten yang bertugas menganalisis dokumen PDF pembelajaran dan mengekstrak/menghasilkan materi serta soal terkait bahasa pemrograman Python.

INSTRUKSI:

1. FILTER KONTEN:
   Periksa seluruh isi dokumen PDF ini. Fokus HANYA pada materi yang membahas Python
   (sintaks, fungsi, struktur data, algoritma dengan Python, dsb).
   Abaikan materi yang tidak berkaitan dengan Python (bahasa pemrograman lain, teori umum
   tanpa kaitan langsung ke Python, dsb).

   Jika TIDAK ADA materi Python yang ditemukan dalam dokumen ini, kembalikan HANYA JSON berikut
   dan hentikan proses:
   {
     "status": "no_python_content",
     "message": "Tidak ada data yang mengandung materi Python"
   }

2. JIKA DITEMUKAN MATERI PYTHON, untuk SETIAP materi/topik Python yang berbeda dalam dokumen,
   lakukan langkah berikut:

   a. MATERI: Ringkas dan format ulang materi tersebut ke dalam bentuk markdown yang dipadatkan
      menjadi SATU BARIS (single-line string, gunakan \\n eksplisit untuk baris baru di dalam
      string jika diperlukan struktur, jangan gunakan multi-line JSON string mentah).

   b. SOAL YANG SUDAH ADA: Ambil/ekstrak soal-soal latihan yang SUDAH ADA di dalam PDF untuk
      materi ini (jika tersedia). Jangan mengubah isi soal asli, cukup strukturkan ke format JSON
      di bawah.

   c. SOAL BARU (AI GENERATED): Buat 2-3 soal BARU yang relevan dengan materi ini, dengan tingkat
      kesulitan bervariasi (mudah, sedang, sulit), yang TIDAK sama persis dengan soal yang sudah
      ada di PDF.

3. FORMAT OUTPUT:
   Kembalikan HANYA JSON valid (tanpa markdown code fence, tanpa penjelasan tambahan) dengan
   struktur PERSIS seperti berikut:

{
  "status": "success",
  "materi_list": [
    {
      "id_materi": 1,
      "judul_materi": "<judul topik/materi, contoh: 'Fungsi dan Parameter'>",
      "content_markdown": "<ringkasan materi dalam format markdown satu baris>",
      "soal_existing": [
        {
          "reference": "<PDF - nama file atau halaman jika diketahui>",
          "sub_theme": "<judul_materi yang sama dengan di atas>",
          "judul": "<judul singkat soal>",
          "soal": "<teks soal>",
          "expected_output": "<output yang diharapkan>"
        }
      ],
      "soal_generated": [
        {
          "reference": "AI Generated",
          "sub_theme": "<judul_materi yang sama dengan di atas>",
          "judul": "<judul singkat soal>",
          "soal": "<teks soal>",
          "expected_output": "<output yang diharapkan>",
          "difficulty": "<easy|medium|hard>"
        }
      ]
    }
  ]
}

PENTING:
- Jangan menyertakan materi non-Python dalam materi_list.
- Jika PDF tidak memiliki soal existing untuk suatu materi, kembalikan array kosong [] untuk
  soal_existing (jangan mengarang soal yang mengaku dari PDF).
- Pastikan setiap objek JSON valid dan bisa langsung di-parse tanpa post-processing tambahan.
"""

from google import genai
import json

genai.configure(api_key="YOUR_API_KEY")
model = genai.GenerativeModel("gemini-2.0-flash")

uploaded_file = genai.upload_file(path="materi.pdf")

response = model.generate_content(
    [uploaded_file, GENERATE_MATERI_PROMPT],
    generation_config={"response_mime_type": "application/json"}
)

result = json.loads(response.text)

if result["status"] == "no_python_content":
    print(result["message"])
else:
    materi_list = result["materi_list"]
    print(f"Found {len(materi_list)} Python-related materials")
    with open("generated_materi.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)