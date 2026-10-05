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


# ---------------------------------------------------------------- v2.4: shapes of the two real runs

NHP = {"DEAB": [0, 4, 16, 52], "GA43": [0, 4, 16, 136], "H7K2": [0, 4, 16, 52, 136], "R12X": [0, 4, 16, 52, 136],
       "PB09": [0, 4, 8, 16, 52, 136], "T623": [0], "M88A": [4], "W5TA": [16]}   # 4, 4, 5, 5, 6, 1, 1, 1 samples
NHP_SAMPLES = [f"{a}_{tp}" for a, tps in NHP.items() for tp in tps]


def k_somascan_nhp(rng, n_features=400):
    """SomaScan NHP shape (v2.4 §22): 76 columns = 1 ID + 48 annotations + 27 sample columns
    named <animal>_<number>; near-duplicate annotation pairs; a Type column with non-human rows.
    Plus a 33-column sample metadata file (nhp_id, time_point, TimePoint, study_group, SubjectID ...)."""
    ann = ["Target", "Target Name", "TargetFullName", "UniProt", "UniProt ID", "EntrezGeneID", "EntrezGeneSymbol",
           "Organism", "Units", "Type_meta", "Type_anno", "Dilution", "Dilution2", "PlateScale_Reference",
           "CalReference", "Cal_Set_A", "ColCheck", "QC_CV_plasma", "QC_CV_serum", "SignalToNoise",
           "LoD_plasma", "LoD_serum", "MedianSignal_buffer", "Aptamer_Length", "SomaId", "TargetType",
           "Protein_Class", "Pathway", "Subcellular", "Secreted", "HPA_Tissue", "Panel", "Panel_Version",
           "Seq_Notes", "Flag_Crossreactive", "Flag_HighCV", "Assay_Plate_Pos", "KD_nM", "Mol_Weight_kDa",
           "GO_Process", "GO_Function", "Chromosome", "Gene_Start", "Gene_End", "Strand", "Ensembl",
           "Annotation_Date", "Comment"]
    assert len(ann) == 48 and len(NHP_SAMPLES) == 27
    header = ["SeqId"] + ann[:24] + NHP_SAMPLES + ann[24:]
    types = ["Protein"] * 18 + ["Non-Human"] + ["Hybridization Control Elution", "Spuriomer"]
    rows = []
    for k in range(n_features):
        gene = f"G{k % 350}"
        uni = f"P{10000 + k:05d}"
        typ = types[k % len(types)] if k % 7 == 0 else "Protein"
        base = rng.gauss(3.1, 0.5)
        a = {"Target": f"T{k}", "Target Name": f"T{k}", "TargetFullName": f"Target protein {k}", "UniProt": uni,
             "UniProt ID": uni if k % 50 else "", "EntrezGeneID": str(1000 + k % 350), "EntrezGeneSymbol": gene,
             "Organism": "Human" if typ == "Protein" else "", "Units": "RFU", "Type_meta": typ, "Type_anno": typ,
             "Dilution": rng.choice(["20", "0.5", "0.005"]), "Dilution2": rng.choice(["20", "0.5", "0.005"]),
             "PlateScale_Reference": f"{rng.uniform(500, 5000):.1f}", "CalReference": f"{rng.uniform(500, 5000):.1f}",
             "Cal_Set_A": f"{rng.uniform(0.8, 1.2):.3f}", "ColCheck": rng.choice(["PASS", "PASS", "FLAG"]),
             "QC_CV_plasma": f"{rng.uniform(2, 15):.2f}", "QC_CV_serum": f"{rng.uniform(2, 15):.2f}",
             "SignalToNoise": f"{rng.uniform(1, 40):.2f}", "LoD_plasma": f"{rng.uniform(10, 200):.1f}",
             "LoD_serum": f"{rng.uniform(10, 200):.1f}", "MedianSignal_buffer": f"{rng.uniform(50, 400):.1f}",
             "Aptamer_Length": str(rng.randint(40, 50)), "SomaId": f"SL{k:06d}", "TargetType": "Protein",
             "Protein_Class": rng.choice(["enzyme", "receptor", "cytokine", "other"]),
             "Pathway": rng.choice(["immune", "metabolism", "signalling", ""]), "Subcellular": rng.choice(["secreted", "membrane", "cytoplasm"]),
             "Secreted": rng.choice(["yes", "no"]), "HPA_Tissue": rng.choice(["liver", "blood", "brain", "kidney"]),
             "Panel": "7k", "Panel_Version": "4.1", "Seq_Notes": "", "Flag_Crossreactive": rng.choice(["", "", "X"]),
             "Flag_HighCV": rng.choice(["", "", "", "Y"]), "Assay_Plate_Pos": f"{rng.choice('ABCDEFGH')}{rng.randint(1, 12)}",
             "KD_nM": f"{rng.uniform(0.01, 50):.3f}", "Mol_Weight_kDa": f"{rng.uniform(8, 300):.1f}",
             "GO_Process": rng.choice(["GO:0006955", "GO:0008152", ""]), "GO_Function": rng.choice(["GO:0005515", ""]),
             "Chromosome": str(rng.randint(1, 22)), "Gene_Start": str(rng.randint(1000, 9000000)),
             "Gene_End": str(rng.randint(9000000, 9900000)), "Strand": rng.choice(["+", "-"]),
             "Ensembl": f"ENSG{rng.randint(10 ** 10, 10 ** 11 - 1)}", "Annotation_Date": "2021-06-01", "Comment": ""}
        vals = {sm: f"{10 ** (base + rng.gauss(0, 0.12)):.1f}" for sm in NHP_SAMPLES}
        rows.append([f"{10000 + k}-{k % 90 + 1}_3"] + [a[h] if h in a else vals[h] for h in header[1:]])
    w("K_somascan_nhp.csv", header, rows)
    groups = {"DEAB": "SIV+ART", "GA43": "SIV+ART", "H7K2": "SIV", "R12X": "SIV", "PB09": "SIV+ART", "T623": "control",
              "M88A": "control", "W5TA": "SIV"}
    mh = ["SampleId", "nhp_id", "time_point", "TimePoint", "study_group", "SubjectID", "nhp_week_index", "plate",
          "slide", "scanner", "run_order", "batch", "sex", "age_years", "weight_kg", "viral_load", "cd4_count",
          "hemolysis", "sample_volume_ul", "collection_site", "freeze_thaw", "storage_days", "operator",
          "RowCheck", "NormScale_20", "NormScale_0_5", "NormScale_0_005", "SampleType", "SampleMatrix",
          "Barcode", "Notes", "ART_start_week", "necropsy"]
    assert len(mh) == 33
    mrows = []
    for a, tps in NHP.items():
        sex, age = rng.choice(["F", "M"]), rng.randint(3, 9)
        for k, tp in enumerate(tps):
            mrows.append([f"{a}_{tp}", a, str(tp), f"W{tp}", groups[a], f"NHP-{a}", str(k + 1), f"P{1 + k % 3}",
                          f"S{rng.randint(1, 9)}", "SC1", str(len(mrows) + 1), str(1 + len(mrows) // 14), sex, str(age),
                          f"{rng.uniform(4, 9):.1f}", f"{10 ** rng.uniform(1, 6):.0f}", str(rng.randint(200, 1400)),
                          rng.choice(["none", "slight"]), "50", rng.choice(["A", "B"]), str(rng.randint(0, 2)),
                          str(rng.randint(10, 900)), rng.choice(["ab", "cd"]), "PASS", f"{rng.uniform(0.8, 1.2):.3f}",
                          f"{rng.uniform(0.8, 1.2):.3f}", f"{rng.uniform(0.8, 1.2):.3f}", "Sample", "plasma",
                          f"BC{rng.randint(10 ** 6, 10 ** 7)}", "", "8" if groups[a] == "SIV+ART" else "",
                          "yes" if k == len(tps) - 1 else "no"])
    w("K_somascan_nhp_metadata.csv", mh, mrows)


METAB_CLASSES = [("Amino Acid", 200), ("Carbohydrate", 30), ("Cofactors and Vitamins", 40), ("Energy", 10),
                 ("Lipid", 450), ("Nucleotide", 40), ("Peptide", 30), ("Xenobiotics", 200),
                 ("Partially Characterized Molecules", 14), ("", 160)]


def l_metabolon_like(rng):
    """Metabolomics run shape (v2.4 §22): 27 rows (the K sample names, same order), an ID column and
    1,174 numeric columns <class>_<number> (nine classes + 160 with an empty class), one export
    median-scaled to ~1 everywhere; one column at its minimum in >= 14 of 27 samples."""
    cols = []
    used = set()
    for cls, n in METAB_CLASSES:
        for _ in range(n):
            while True:
                num = rng.choice([rng.randint(30, 9999), rng.randint(100000000, 100020000), rng.randint(999900000, 999999999)])
                if num not in used:
                    used.add(num)
                    break
            cols.append(f"{cls}_{num}")
    for fixed in ("Amino Acid_100010863", "Xenobiotics_100010955"):
        if fixed not in cols:
            cols[[c.split("_")[0] for c in cols].index(fixed.split("_")[0])] = fixed
    assert len(cols) == 1174
    data = {c: [rng.lognormvariate(0, 0.35) for _ in NHP_SAMPLES] for c in cols}
    x = data["Xenobiotics_100010955"]
    for k in range(15):
        x[k * 27 // 15] = 0.124948
    data["Xenobiotics_100010955"] = [max(v, 0.124948) for v in x]
    rows = [[sm] + [f"{data[c][r]:.6g}" for c in cols] for r, sm in enumerate(NHP_SAMPLES)]
    w("L_metabolon_like.csv", ["ID"] + cols, rows)


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
    k_somascan_nhp(random.Random(4412))
    l_metabolon_like(random.Random(151))
    print("fixtures written to", HERE)
