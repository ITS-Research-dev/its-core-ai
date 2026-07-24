const questions = [
  {
    id: 22,
    checked: true,
    source_valid: "valid",
    reference: "AI_REF_1 Hal 139",
    "sub-theme":
      "Algoritma Pencarian - Linear Search vs Binary Search (Python)",
    judul: "Studi Kasus: Mencari Nama dalam Daftar Murid",
    soal: "Buatlah program Python untuk mencari nama seorang murid pada daftar nama menggunakan Linear Search dan Binary Search. Bandingkan efisiensi kedua algoritma dan jelaskan kapan masing-masing lebih tepat digunakan.",
    code: `def linear_search(names, target):
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
bs_idx, bs_cmp = binary_search(names, target)

print(f"Linear Search: ditemukan di indeks {ls_idx} ({ls_cmp} kali perbandingan)")
print(f"Binary Search: ditemukan di indeks {bs_idx} ({bs_cmp} kali perbandingan)")
print("Linear Search cocok untuk data kecil/tidak terurut.")
print("Binary Search lebih efisien untuk data besar yang sudah terurut.")`,
    expected_output: `Linear Search: ditemukan di indeks 2 (3 kali perbandingan)
Binary Search: ditemukan di indeks 2 (2 kali perbandingan)
Linear Search cocok untuk data kecil/tidak terurut.
Binary Search lebih efisien untuk data besar yang sudah terurut.`,
    simulated_input: "",
    level: "medium",
    hints: ["pseudocode", "code"],
  },
  {
    id: 23,
    checked: true,
    source_valid: "valid",
    reference: "AI_REF_1 Hal 141",
    "sub-theme": "Algoritma Pencarian - Binary Search (Python)",
    judul: "Studi Kasus: Mencari Harga Produk pada E-Commerce",
    soal: "Buatlah program Python yang menggunakan algoritma Binary Search untuk mencari harga produk pada daftar harga yang telah diurutkan. Program harus menampilkan indeks data apabila harga ditemukan, atau pesan bahwa data tidak ditemukan.",
    code: `harga = [10000, 15000, 20000, 25000, 30000]
cari = 20000
posisi = -1

for i in range(len(harga)):
    if harga[i] == cari:
        posisi = i

print(posisi)`,
    expected_output: `Daftar harga: [10000, 15000, 20000, 25000, 30000]
Harga 20000 ditemukan di indeks 2`,
    simulated_input: "",
    level: "medium",
    hints: ["pseudocode", "code"],
  },
];
