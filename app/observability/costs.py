"""Estimasi biaya token (FR-8, PRD §12 'Biaya API membengkak').

Tarif per 1 juta token, USD. Angka ini berubah -- perbarui dari halaman harga
resmi, jangan tebak. Estimasi disimpan per pesan agar AD-5 bisa menampilkan
biaya berjalan tanpa memanggil API penagihan.
"""

from __future__ import annotations

from dataclasses import dataclass

PRICES_PER_MTOK: dict[str, tuple[float, float]] = {
    # model: (input, output)
    "cx/gpt-5.5": (5, 30),
    "openrouter/openai/text-embedding-3-small": (0.02, 0.0),
}


@dataclass(frozen=True)
class CostEstimate:
    input_tokens: int
    output_tokens: int
    usd: float


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> CostEstimate:
    """Hitung perkiraan biaya satu panggilan.

    Raises:
        KeyError: model tak dikenal -- lebih baik gagal daripada melaporkan
        biaya nol yang menyesatkan.
    """
    return _hitung(PRICES_PER_MTOK[model], input_tokens, output_tokens)


def price_for(model: str) -> tuple[float, float] | None:
    """Tarif (input, output) per 1 juta token, atau None bila belum terdaftar.

    ID dari gateway OpenAI-compatible sering diberi awalan penyedia, mis.
    `openrouter/openai/gpt-4o-mini`. Bila ID lengkap tidak terdaftar, bagian
    setelah garis miring terakhir dicoba. Tidak ada tebakan lebih jauh dari
    itu: tarif yang salah lebih menyesatkan daripada tarif yang kosong.
    """
    if model in PRICES_PER_MTOK:
        return PRICES_PER_MTOK[model]
    return PRICES_PER_MTOK.get(model.rsplit("/", 1)[-1])


def try_estimate_cost(
    model: str, input_tokens: int, output_tokens: int
) -> CostEstimate | None:
    """Seperti `estimate_cost`, tetapi None untuk model tanpa tarif.

    Dipakai saat mencatat percakapan: tarif yang belum didaftarkan tidak boleh
    menggagalkan jawaban ke mahasiswa. AD-5 menghitung pesan seperti ini dan
    menampilkannya sebagai peringatan, bukan sebagai biaya nol.
    """
    harga = price_for(model)
    if harga is None:
        return None
    return _hitung(harga, input_tokens, output_tokens)


def _hitung(harga: tuple[float, float], input_tokens: int, output_tokens: int) -> CostEstimate:
    if input_tokens < 0 or output_tokens < 0:
        raise ValueError("jumlah token tidak boleh negatif")
    price_in, price_out = harga
    usd = (input_tokens / 1_000_000) * price_in + (output_tokens / 1_000_000) * price_out
    return CostEstimate(input_tokens, output_tokens, round(usd, 6))
