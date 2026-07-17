from __future__ import annotations

# IG = Information Geometry (informaciona geometrija) 

"""
Geometrija prostora kombinacija — v1 Hebbian

Prostor: brojevi 1…39 kao čvorovi.
Hebbian:
  W_ij += 1  ako i,j zajedno u istom kolu (co-fire)
  W_ij += λ  ako i u kolu t, j u kolu t+1 (susedna kola)

Geometrija = simetrična težina → row-stochastic D (distribucija energije
sa čvora, ne frekvencija count).

Skor: energija od last (maska) kroz D — gde masa ide posle last.
Ban last; next. CSV ceo, seed=39.
Ime: ig_hebbian_v1_manifold.py
"""

import csv
from itertools import combinations
from pathlib import Path

import numpy as np

SEED = 39
FRONT_N = 39
FRONT_SELECT = 7
LAMBDA_TEMP = 0.35  # težina Hebbian veze t → t+1
CSV_PATH = Path(__file__).resolve().parents[1] / "data" / "loto7_4652_k57.csv"

np.random.seed(SEED)


def load_draws(csv_path: Path = CSV_PATH) -> np.ndarray:
    draws = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        for row in csv.reader(f):
            if len(row) < FRONT_SELECT:
                continue
            try:
                draw = sorted(int(x.strip()) for x in row[:FRONT_SELECT])
            except ValueError:
                continue
            if len(draw) == FRONT_SELECT and all(1 <= x <= FRONT_N for x in draw):
                if len(set(draw)) == FRONT_SELECT:
                    draws.append(draw)
    if not draws:
        raise ValueError(f"Nema validnih kola u {csv_path}")
    return np.array(draws, dtype=int)


def hebbian_weights(draws: np.ndarray, lam: float = LAMBDA_TEMP) -> np.ndarray:
    """W[i,j] Hebbian (0-based indeksi). Dijagonala 0."""
    W = np.zeros((FRONT_N, FRONT_N), dtype=float)
    for d in draws:
        idx = [int(x) - 1 for x in d.tolist()]
        for a, b in combinations(idx, 2):
            W[a, b] += 1.0
            W[b, a] += 1.0
    for t in range(len(draws) - 1):
        a_idx = [int(x) - 1 for x in draws[t].tolist()]
        b_idx = [int(x) - 1 for x in draws[t + 1].tolist()]
        for a in a_idx:
            for b in b_idx:
                if a == b:
                    continue
                W[a, b] += lam
                W[b, a] += lam
    np.fill_diagonal(W, 0.0)
    return W


def energy_distribution(W: np.ndarray) -> np.ndarray:
    """Row-stochastic: D_ij = W_ij / sum_j W_ij  (distribucija energije sa i)."""
    D = W.copy()
    row = D.sum(axis=1, keepdims=True)
    row = np.where(row < 1e-18, 1.0, row)
    D = D / row
    return D


def number_scores(D: np.ndarray, last: np.ndarray, ban: set[int]) -> dict[int, float]:
    """
    Energija na j: prosek D[i→j] preko i ∈ last (gde masa ide od last).
    To je distribucioni skor, ne count frekvencije.
    """
    idx = [int(x) - 1 for x in last.tolist()]
    mass = D[idx].mean(axis=0)
    out = {}
    for j in range(FRONT_N):
        n = j + 1
        if n in ban:
            out[n] = -1e18
        else:
            out[n] = float(mass[j])
    return out


def _combo_fit(combo, score, target_sum, pos_means, target_odd, ban):
    nums = sorted(combo)
    if any(x in ban for x in nums):
        return -1e18
    s = sum(score[x] for x in nums)
    s -= 0.08 * abs(sum(nums) - target_sum)
    s -= 0.04 * sum(abs(nums[i] - pos_means[i]) for i in range(FRONT_SELECT))
    odd = sum(1 for x in nums if x % 2)
    s -= 0.3 * abs(odd - target_odd)
    return s


def predict_next(draws, score, ban):
    ranked = sorted((n for n in score if n not in ban), key=lambda n: (-score[n], n))
    target_sum = float(draws.sum(axis=1).mean())
    pos_means = [float(draws[:, i].mean()) for i in range(FRONT_SELECT)]
    target_odd = float(np.mean([sum(1 for x in d if x % 2) for d in draws]))
    candidates = [sorted(ranked[:FRONT_SELECT])]
    for start in range(0, min(20, len(ranked) - FRONT_SELECT + 1)):
        candidates.append(sorted(ranked[start : start + FRONT_SELECT]))
    best, best_fit = None, -1e18
    for base in candidates:
        fit = _combo_fit(base, score, target_sum, pos_means, target_odd, ban)
        if fit > best_fit:
            best_fit, best = fit, list(base)
        for i in range(FRONT_SELECT):
            for repl in ranked[:30]:
                cand = sorted(set(base[:i] + base[i + 1 :] + [repl]))
                if len(cand) != FRONT_SELECT:
                    continue
                fit = _combo_fit(cand, score, target_sum, pos_means, target_odd, ban)
                if fit > best_fit:
                    best_fit, best = fit, cand
    return best if best is not None else sorted(ranked[:FRONT_SELECT])


def run_v1(csv_path: Path = CSV_PATH) -> None:
    draws = load_draws(csv_path)
    last = draws[-1]
    ban = set(int(x) for x in last.tolist())
    W = hebbian_weights(draws)
    D = energy_distribution(W)
    score = number_scores(D, last, ban)
    combo = predict_next(draws, score, ban)

    print(f"CSV: {csv_path.name}")
    print(
        f"Kola: {len(draws)} | seed={SEED} | λ_temp={LAMBDA_TEMP} | ig_hebbian_v1"
    )
    print(f"last: {last.tolist()}")
    print()
    print("=== Hebbian geometrija ===")
    print(
        {
            "W_nnz": int(np.count_nonzero(W)),
            "W_mean": round(float(W.mean()), 6),
            "D_row_sums_ok": bool(np.allclose(D.sum(axis=1), 1.0)),
        }
    )
    print()
    ranked = sorted(
        ((n, float(score[n])) for n in range(1, FRONT_N + 1) if n not in ban),
        key=lambda t: (-t[1], t[0]),
    )
    print("=== top12 skor (energija od last) ===")
    print([(n, round(sc, 6)) for n, sc in ranked[:12]])
    print()
    print("=== next (ig_hebbian_v1) ===")
    print("next:", combo)


if __name__ == "__main__":
    run_v1()



"""
CSV: loto7_4652_k57.csv
Kola: 4652 | seed=39 | λ_temp=0.35 | ig_hebbian_v1
last: [7, 8, 14, 15, 17, 23, 32]

=== Hebbian geometrija ===
{'W_nnz': 1482, 'W_mean': 230.700855, 'D_row_sums_ok': True}

=== top12 skor (energija od last) ===
[(26, 0.028106), (34, 0.027738), (37, 0.027466), (33, 0.027273), (11, 0.027213), (39, 0.027109), (22, 0.027041), (29, 0.02698), (35, 0.026978), (38, 0.02684), (10, 0.026791), (24, 0.026783)]

=== next (ig_hebbian_v1) ===
next: [3, 10, 12, 24, 25, 31, 35]
"""



"""
Hebbian W (co-fire + t→t+1) → D distribucija energije → skor od last → next.
"""



"""
0. Granica
Loto i.i.d. → nema prediktivnog transporta kao u 03–05.
Ovde: algoritam uči geometriju prostora kombinacija i traži putanju energije (distribucija) → next. Ne LLM.
1. Prostor
Tačka = 7-torica (ili simplex masa na {1…39}).
Manifold = geometrija naučena iz istorije CSV (sličnost / metrika među kolima), ne nametnuti Fisher/Γ.
2. „Nebo“ (Perez intuicija)
Polje „osvetljenosti“ na prostoru brojeva/zona — analog Perez (zenit / sunce / turbidnost → parametri iz podataka).
Cirkadijalni sloj: periodična modulacija polja kroz vreme (indeks kola / faza).
3. Hebbian
Jačanje veza između ko-pojavljivanja / susednih kola na manifoldu („fire together → wire together“).
Matrica / težine = lokalna geometrija.
4. RLM (ne LLM)
Rekurzivno / lokalno učenje putanje na tom grafu/manifoldu (stanje → korak → ažuriranje).
Ekscitacija: mali perturbatori da se održi observabilnost geometrije (Stošić intuicija).
5. Energija = distribucija
Cilj: pomeraj mase/energije (distribucija na simpleksu), ne rang frekvencije.
Putanja ≈ diskretni OT korak na naučenoj metriki (Hebbian+RLM), ne sirovi Sinkhorn kao „predikcija“.
6. next
Kraj putanje / maksimum energije pod zabranom last → jedna kombinacija.
Merilo: gde putanja prati empiriju vs gde odstupa → tada nadogradnja (ne novi šum).

v1 — prostor + Hebbian težine + next
v2 — Perez-polje + cirkadijalna faza
v3 — RLM koraci na manifoldu + ekscitacija
v4 — energija/distribucija OT na naučenoj metriki → next + dijagnostika odstupanja

Seed 39, CSV loto7_4650_k56, samo simulator/.
"""
