from typing import Dict, List, Set, Tuple
from collections import defaultdict

def cluster_purity(pred_clusters: List[List[str]], labels: Dict[str, str]) -> float:
    # labels maps listing_id -> gold_brand or gold_entity
    total = sum(len(c) for c in pred_clusters)
    correct = 0
    for c in pred_clusters:
        counts = defaultdict(int)
        for lid in c:
            counts[labels[lid]] += 1
        correct += max(counts.values()) if counts else 0
    return correct / max(total, 1)

def pairwise_f1(pred_clusters: List[List[str]], labels: Dict[str, str]) -> float:
    def pairs(clusters):
        s = set()
        for c in clusters:
            for i in range(len(c)):
                for j in range(i+1, len(c)):
                    s.add((min(c[i], c[j]), max(c[i], c[j])))
        return s
    # gold by label
    gold_map = defaultdict(list)
    for lid, lab in labels.items():
        gold_map[lab].append(lid)
    gold_clusters = list(gold_map.values())
    P = pairs(pred_clusters)
    G = pairs(gold_clusters)
    tp = len(P & G)
    fp = len(P - G)
    fn = len(G - P)
    precision = tp / (tp + fp) if (tp+fp)>0 else 0.0
    recall = tp / (tp + fn) if (tp+fn)>0 else 0.0
    return (2*precision*recall)/(precision+recall) if (precision+recall)>0 else 0.0