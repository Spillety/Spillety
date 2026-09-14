def pairwise_precision_recall(
    predicted: dict[str, str], ground_truth: dict[str, str]
) -> dict[str, float]:
    addrs = [a for a in predicted if a in ground_truth]
    tp = fp = fn = 0
    for i in range(len(addrs)):
        for j in range(i + 1, len(addrs)):
            pred_link = predicted[addrs[i]] == predicted[addrs[j]]
            true_link = ground_truth[addrs[i]] == ground_truth[addrs[j]]
            if pred_link and true_link:
                tp += 1
            elif pred_link:
                fp += 1
            elif true_link:
                fn += 1
    # No predicted links means no false merges; no true links means nothing to miss.
    precision = tp / (tp + fp) if tp + fp > 0 else 1.0
    recall = tp / (tp + fn) if tp + fn > 0 else 1.0
    return {"precision": precision, "recall": recall}
