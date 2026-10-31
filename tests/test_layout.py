"""Step 2 tests: orientation, ID column, measurement and annotation columns, the matrix."""

import numpy as np
import pytest

import prism


def somascan_like(tmp_path, n_feat=300, n_samp=12, seed=0):
    """Proteins in rows: text and numeric annotation columns, then the sample columns."""
    rng = np.random.default_rng(seed)
    level = rng.normal(8, 1.5, n_feat)                                  # each protein's abundance (log scale)
    lines = ["SeqId,Target,UniProt,Kd,LoD,CV," + ",".join(f"S{j:02d}" for j in range(n_samp))]
    for i in range(n_feat):
        samples = np.exp(level[i] + rng.normal(0, 0.2, n_samp))
        ann = [f"{1000 + i}-{i % 7}", f"Protein {i}", f"P{i:05d}",
               f"{rng.uniform(1e-12, 1e-9):.3e}", f"{rng.uniform(10, 200):.1f}", f"{rng.uniform(0.02, 0.2):.3f}"]
        vals = [f"{v:.2f}" for v in samples]
        if i == 5:
            vals[3] = ""                                                  # one missing cell
        lines.append(",".join(ann + vals))
    p = tmp_path / "soma.csv"
    p.write_text("\n".join(lines) + "\n")
    return p


def metabolomics_like(tmp_path, n_samp=20, n_feat=150, seed=1):
    """Samples in rows, metabolites as columns, a group code column."""
    rng = np.random.default_rng(seed)
    lines = ["ID,Group," + ",".join(f"Lipid_{k}" for k in range(n_feat))]
    for j in range(n_samp):
        lines.append(f"Sample_{j:02d},{j % 2}," + ",".join(f"{v:.3f}" for v in rng.lognormal(2, 1, n_feat)))
    p = tmp_path / "metab.csv"
    p.write_text("\n".join(lines) + "\n")
    return p


def test_features_in_rows_finds_the_samples(tmp_path):
    layout = prism.propose_layout(prism.read(somascan_like(tmp_path)))
    assert layout.orientation == "features_in_rows"
    assert layout.id_column == "SeqId"
    # every numeric column is proposed; numeric annotations are moved by the user
    assert layout.measurement_columns == ["Kd", "LoD", "CV"] + [f"S{j:02d}" for j in range(12)]
    assert set(layout.annotation_columns) == {"Target", "UniProt"}
    assert any("set(annotations=" in r for r in layout.reasons)
    layout = layout.set(annotations=["Kd", "LoD", "CV"])
    assert layout.measurement_columns == [f"S{j:02d}" for j in range(12)]
    X = layout.matrix()
    assert X.shape == (300, 12) and X.index[0] == "1000-0" and np.isnan(X.iloc[5, 3])
    assert layout.annotations().loc["1000-0", "Target"] == "Protein 0"


def test_samples_in_rows_is_turned_into_features_x_samples(tmp_path):
    layout = prism.propose_layout(prism.read(metabolomics_like(tmp_path)))
    assert layout.orientation == "samples_in_rows" and layout.id_column == "ID"
    assert "Group" in layout.annotation_columns                       # a 0/1 code, not a metabolite
    X = layout.matrix()
    assert X.shape == (150, 20) and list(X.columns[:2]) == ["Sample_00", "Sample_01"]
    assert layout.annotations().loc["Sample_01", "Group"] == "1"


def test_set_corrects_the_proposal(tmp_path):
    layout = prism.propose_layout(prism.read(somascan_like(tmp_path)))
    fixed = layout.set(annotations=["S00"], measurements=["LoD"])
    assert "S00" in fixed.annotation_columns and "LoD" in fixed.measurement_columns
    assert fixed.measurement_columns[:2] == ["Kd", "LoD"]              # file order is kept
    assert "S00" in layout.measurement_columns                         # the original proposal is unchanged
    assert fixed.reasons[-1].startswith("set by you")
    with pytest.raises(KeyError):
        layout.set(measurements=["nope"])
    with pytest.raises(ValueError):
        layout.set(orientation="sideways")
    roles = fixed.roles().set_index("column")["role"]
    assert roles["SeqId"] == "id" and roles["S00"] == "annotation"


def test_no_id_column_numbers_the_rows(tmp_path):
    p = tmp_path / "noid.csv"
    p.write_text("a,b\n1,2\n1,3\n2,5\n")
    layout = prism.propose_layout(prism.read(p))
    assert layout.id_column is None and list(layout.matrix().index) == ["1", "2", "3"]


def test_a_numeric_first_column_of_whole_numbers_can_be_the_id(tmp_path):
    p = tmp_path / "numid.csv"
    p.write_text("id,x,y\n10,1.5,2.5\n11,1.7,2.1\n12,3.2,9.9\n")
    assert prism.propose_layout(prism.read(p)).id_column == "id"
