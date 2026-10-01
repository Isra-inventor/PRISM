"""Generate the synthetic test fixtures A-F (and a messy-parsing file).

Run:  python tests/fixtures/make_fixtures.py
Deterministic (seeded), so the committed files can be regenerated exactly.
"""

import csv
import math
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent


def w(name, header, rows, delimiter=","):
    with open(HERE / name, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f, delimiter=delimiter, lineterminator="\n")
        wr.writerow(header)
        wr.writerows(rows)


def a_maxquant(rng):
    samples = [f"S{i:02d}" for i in range(1, 7)]
    header = (["Protein IDs", "Majority protein IDs", "Protein names", "Gene names", "Fasta headers",
               "Peptides", "Razor + unique peptides", "Unique peptides", "Sequence coverage [%]",
               "Mol. weight [kDa]", "Q-value", "Score", "Intensity"]
              + [f"Intensity {s}" for s in samples]
              + [f"LFQ intensity {s}" for s in samples]
              + [f"iBAQ {s}" for s in samples]
              + [f"Peptides {s}" for s in samples]
              + ["Only identified by site", "Reverse", "Potential contaminant", "id"])
    genes = ["ALB", "APOA1", "HP", "TF", "A2M", "C3", "FGB", "SERPINA1", "APOB", "IGHG1", "CP", "HPX",
             "ORM1", "AGT", "KNG1", "VTN", "GC", "FN1", "CFB", "PLG", "APOH", "ITIH2", "AMBP", "LRG1",
             "CLU", "AHSG", "FGA", "FGG", "C4A", "C4B", "APOE", "SAA1", "CRP", "TTR", "RBP4", "APOC3",
             "GSN", "LPA", "F2", "SERPINC1"]
    rows = []
    for k in range(60):
        g = genes[k % len(genes)] + ("" if k < len(genes) else str(k))
        acc = f"P{10000 + 37 * k}"
        rev = k in (55, 56, 57)
        con = k in (50, 51)
        pid = ("REV__" if rev else "CON__" if con else "") + acc
        base = 10 ** rng.uniform(6, 9.5)
        inten = [0 if rng.random() < 0.12 else int(base * rng.lognormvariate(0, 0.3)) for _ in samples]
        lfq = [0 if v == 0 else int(v * rng.uniform(0.8, 1.2)) for v in inten]
        ibaq = [0 if v == 0 else int(v / rng.randint(5, 40)) for v in inten]
        peps = [0 if v == 0 else rng.randint(1, 40) for v in inten]
        rows.append([pid, pid, f"{g} protein", g, f">sp|{acc}|{g}_HUMAN {g} protein OS=Homo sapiens",
                     max(peps), max(peps), max(0, max(peps) - 1), round(rng.uniform(2, 80), 1),
                     round(rng.uniform(10, 250), 3), round(rng.uniform(0, 0.01), 5),
                     round(rng.uniform(5, 323), 3), sum(inten)]
                    + inten + lfq + ibaq + peps
                    + ["+" if k == 20 else "", "+" if rev else "", "+" if con else "", k])
    w("A_maxquant_proteinGroups.txt", header, rows, "\t")


def b_diann(rng):
    samples = [f"S{i:02d}.raw" for i in range(1, 9)]
    header = ["PG.ProteinGroups", "PG.ProteinNames", "PG.Genes"] + samples
    rows = []
    for k in range(50):
        acc = f"Q{20000 + 11 * k}"
        base = rng.uniform(14, 28)
        vals = ["NaN" if rng.random() < 0.08 else f"{base + rng.gauss(0, 0.6):.4f}" for _ in samples]
        rows.append([acc, f"PROT{k}_HUMAN", f"GENE{k}"] + vals)
    w("B_diann_pg_matrix.tsv", header, rows, "\t")


def c_mzmine(rng):
    samples = ["QC_01", "QC_02", "QC_03", "blank_01", "blank_02"] + [f"S{i:02d}" for i in range(1, 13)]
    header = (["row ID", "row m/z", "row retention time", "compound_name", "formula", "adduct", "HMDB_ID"]
              + [f"{s} Peak area" for s in samples])
    names = ["Glucose", "Lactate", "Alanine", "Citrate", "Creatinine", "Glutamine", "Valine", "Leucine",
             "Isoleucine", "Tyrosine", "Tryptophan", "Phenylalanine", "Serine", "Glycine", "Proline",
             "Histidine", "Lysine", "Arginine", "Urea", "Uric acid"]
    rows = []
    for k in range(45):
        mz = round(rng.uniform(80, 900), 4)
        rt = round(rng.uniform(0.5, 18), 2)
        known = k < len(names)
        base = 10 ** rng.uniform(3.5, 7)
        vals = []
        for s in samples:
            if s.startswith("blank"):
                vals.append("" if rng.random() < 0.6 else f"{base * 0.01 * rng.random():.1f}")
            else:
                vals.append("" if rng.random() < 0.05 else f"{base * rng.lognormvariate(0, 0.3):.1f}")
        rows.append([k + 1, mz, rt, names[k] if known else "", ("C6H12O6" if known else ""),
                     rng.choice(["[M+H]+", "[M+Na]+", "[M-H]-"]), (f"HMDB{122 + 7 * k:07d}" if known else "")]
                    + vals)
    w("C_mzmine_feature_table.csv", header, rows)


def d_samples_in_rows(rng):
    proteins = [f"P{rng.randint(0, 9)}{rng.randint(1000, 9999)}" for _ in range(25)]
    mets = ["Glucose", "Lactate", "Alanine", "Citrate", "Creatinine", "Glutamine", "Valine", "Leucine",
            "Isoleucine", "Tyrosine", "Tryptophan", "Phenylalanine", "Serine", "Glycine", "Proline",
            "Histidine", "Lysine", "Arginine", "Urea", "Pyruvate"]
    header = (["sample_id", "subject_id", "visit", "severity_group", "age", "CD4_count", "iron", "batch"]
              + proteins + mets)
    pbase = [10 ** rng.uniform(5.5, 9) for _ in proteins]
    mbase = [10 ** rng.uniform(-0.5, 2.5) for _ in mets]
    rows = []
    for s in range(40):
        subj = s // 2 + 1
        rows.append([f"S{s + 1:03d}", f"PT{subj:02d}", "T1" if s % 2 == 0 else "T2",
                     ["mild", "moderate", "severe"][subj % 3], 30 + (subj * 7) % 40,
                     200 + (subj * 53) % 900, f"{rng.uniform(5, 30):.1f}", 1 if s < 20 else 2]
                    + [f"{b * rng.lognormvariate(0, 0.4):.1f}" for b in pbase]
                    + [f"{b * rng.lognormvariate(0, 0.3):.3f}" for b in mbase])
    w("D_samples_in_rows_multiomics.csv", header, rows)


def e_somascan(rng, n_features=1500):
    seqs = [f"seq.{10000 + 3 * k}.{rng.randint(1, 99)}" for k in range(n_features)]
    header = (["PlateId", "SlideId", "SampleId", "SampleType", "PlateScale_Scalar", "HybControlNormScale",
               "NormScale_20", "NormScale_0_005", "NormScale_0_5", "RowCheck"] + seqs)
    types = ["Sample"] * 30 + ["QC"] * 4 + ["Calibrator"] * 4 + ["Buffer"] * 2
    base = [10 ** rng.uniform(2, 4.5) for _ in seqs]
    rows = []
    for s, t in enumerate(types):
        rows.append([f"Set{1 + s // 20}", 258495800000 + s, f"{t[:3].upper()}{s + 1:03d}", t,
                     f"{rng.uniform(0.8, 1.2):.4f}", f"{rng.uniform(0.7, 1.3):.4f}",
                     f"{rng.uniform(0.9, 1.1):.4f}", f"{rng.uniform(0.9, 1.1):.4f}", f"{rng.uniform(0.9, 1.1):.4f}",
                     "PASS" if rng.random() > 0.1 else "FLAG"]
                    + [f"{b * rng.lognormvariate(0, 0.25):.1f}" for b in base])
    w("E_somascan_adat_like.csv", header, rows)


def f_long(rng):
    feats = [f"PROT{k}" for k in range(12)]
    samples = [f"S{i:02d}" for i in range(1, 7)]
    header = ["protein", "gene", "sample", "abundance"]
    rows = [[f, f"G{f[4:]}", s, f"{10 ** rng.uniform(5, 8):.1f}"] for f in feats for s in samples]
    w("F1_long_unique.csv", header, rows)
    dup = [[f, f"G{f[4:]}", s, f"{10 ** rng.uniform(5, 8):.1f}"] for f in feats for s in samples for _ in range(2)]
    w("F2_long_duplicates.csv", header, dup)


def messy():
    rows = [
        ["P1", "1,5", "12", "NA", "abc"],
        ["P2", "2,25", "0", "N/A", " padded "],
        ["P3", "", "7", "#N/A", "x"],
        ["", "", "", "", ""],
        ["P4", "3,0", "Filtered", "-", "y"],
    ]
    w("G_messy_parsing.csv", ["id", "value", "value", "", "note"], rows, ";")


def h_16s_otu(rng):
    """QIIME-style 16S OTU count table: out of PRISM's current scope."""
    ranks = ["k__Bacteria", "p__Firmicutes", "c__Clostridia", "o__Clostridiales", "f__Lachnospiraceae"]
    genera = ["g__Blautia", "g__Roseburia", "g__Dorea", "g__Coprococcus", "g__Faecalibacterium", "g__"]
    samples = [f"Stool.D{d}.{s}" for s in ("A", "B", "C", "D") for d in (0, 7, 14)]
    rows = []
    for k in range(80):
        counts = [0 if rng.random() < 0.45 else int(10 ** rng.uniform(0, 3.5)) for _ in samples]
        rows.append([f"OTU_{k + 1}"] + counts + ["; ".join(ranks + [genera[k % len(genera)]])])
    w("H_16S_otu_table.tsv", ["#OTU ID"] + samples + ["taxonomy"], rows, "\t")


def i_methylation(rng):
    """Illumina-style methylation beta-value matrix: out of PRISM's current scope."""
    samples = [f"GSM{2100450 + i}" for i in range(12)]
    rows = []
    for k in range(200):
        base = rng.choice([0.05, 0.5, 0.9])
        rows.append([f"cg{rng.randrange(10 ** 7, 3 * 10 ** 7):08d}"]
                    + ["NA" if rng.random() < 0.01 else f"{min(0.999, max(0.001, rng.gauss(base, 0.05))):.4f}"
                       for _ in samples])
    w("I_methylation_beta.csv", ["ID_REF"] + samples, rows)


def j_subject_code_blocks(rng):
    """SomaScan-NHP shape (v2.3): ONE measurement whose columns are subject codes with a
    trailing time point (DEAB_0 ... DEAB_136), near-identical statistics everywhere,
    plus subjects with a single column. Over-fragmenting it per code is the bug."""
    design = {"DEAB": [0, 4, 16, 52, 136], "GA43": [4, 16, 136], "H7K2": [0, 4, 136], "R12X": [0, 52, 136],
              "PB09": [0, 16, 52], "KL3M": [4, 52, 136], "Z81Q": [0, 4, 16], "W5TA": [16, 136],
              "T623": [0], "M88A": [4]}
    samples = [f"{code}_{tp}" for code, tps in design.items() for tp in tps]
    rows = []
    for k in range(200):
        base = rng.gauss(2.9, 0.45)
        rows.append([f"seq.{20000 + k}.{k % 97}", f"Target{k}"] + [f"{10 ** (base + rng.gauss(0, 0.08)):.1f}" for _ in samples])
    w("J_subject_code_blocks.csv", ["SeqId", "Target"] + samples, rows)


if __name__ == "__main__":
    rng = random.Random(42)
    a_maxquant(rng)
    b_diann(rng)
    c_mzmine(rng)
    d_samples_in_rows(rng)
    e_somascan(rng)
    f_long(rng)
    messy()
    h_16s_otu(random.Random(16))
    i_methylation(random.Random(450))
    j_subject_code_blocks(random.Random(623))
    print("fixtures written to", HERE)
