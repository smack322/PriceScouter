import pytest
from comparison.metrics import cluster_purity, pairwise_f1

# --- Helpers: tiny fixtures of labels/predicted clusters ---

def labels_ab_cd():
    # Gold: {A,B} belong to X; {C,D} belong to Y
    return {"A": "X", "B": "X", "C": "Y", "D": "Y"}

# -------------------- cluster_purity --------------------

def test_purity_perfect_two_clusters():
    labels = labels_ab_cd()
    pred = [["A", "B"], ["C", "D"]]
    assert cluster_purity(pred, labels) == pytest.approx(1.0)

def test_purity_mixed_cluster():
    # One mixed cluster [A,B,C] (A,B -> X, C -> Y), one pure [D] (Y)
    labels = labels_ab_cd()
    pred = [["A", "B", "C"], ["D"]]
    # Correct counts: max(2,1)=2 from first + 1 from second = 3 correct / 4 total = 0.75
    assert cluster_purity(pred, labels) == pytest.approx(0.75)

def test_purity_all_singletons_equals_label_accuracy():
    labels = labels_ab_cd()
    pred = [["A"], ["B"], ["C"], ["D"]]
    # Each singleton is "pure": 1 correct per cluster -> 4/4 = 1.0
    assert cluster_purity(pred, labels) == pytest.approx(1.0)

def test_purity_empty_pred_is_zero():
    labels = labels_ab_cd()
    pred = []
    assert cluster_purity(pred, labels) == 0.0

# -------------------- pairwise_f1 --------------------

def test_f1_perfect_match():
    labels = labels_ab_cd()
    pred = [["A", "B"], ["C", "D"]]
    assert pairwise_f1(pred, labels) == pytest.approx(1.0)

def test_f1_all_singletons_vs_grouped_gold_is_zero():
    labels = labels_ab_cd()
    pred = [["A"], ["B"], ["C"], ["D"]]  # no positive pairs predicted
    assert pairwise_f1(pred, labels) == pytest.approx(0.0)

def test_f1_partial_merge():
    # Gold: [A,B], [C,D]; Pred: [A,B,C], [D]
    labels = labels_ab_cd()
    pred = [["A", "B", "C"], ["D"]]
    # Pairs: P={(A,B),(A,C),(B,C)}, G={(A,B),(C,D)}
    # tp=1, fp=2, fn=1 -> precision=1/3, recall=1/2, F1=0.4
    assert pairwise_f1(pred, labels) == pytest.approx(0.4, rel=1e-12)

def test_f1_order_invariance():
    labels = labels_ab_cd()
    pred1 = [["A", "B"], ["C", "D"]]
    pred2 = [["B", "A"], ["D", "C"]]  # same clusters, different order
    assert pairwise_f1(pred1, labels) == pytest.approx(pairwise_f1(pred2, labels))

def test_f1_empty_everything_is_zero():
    labels = {}
    pred = []
    assert pairwise_f1(pred, labels) == 0.0

def test_f1_pred_has_pairs_but_no_labels_is_zero():
    labels = {}
    pred = [["A", "B"], ["C"]]  # predicted positives, but no gold labels → precision=0
    assert pairwise_f1(pred, labels) == pytest.approx(0.0)
