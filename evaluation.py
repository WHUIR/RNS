import numpy as np
def _compute_apk(targets, predictions, k):
    if len(predictions) > k:
        predictions = predictions[:k]
    score = 0.0
    num_hits = 0.0
    for i, p in enumerate(predictions):
        if p in targets and p not in predictions[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)
    if not list(targets):
        return 0.0
    return score / min(len(targets), k)
def _compute_precision_recall(targets, predictions, k):
    pred = predictions[:k]
    num_hit = len(set(pred).intersection(set(targets)))
    precision = float(num_hit) / len(pred)
    recall = float(num_hit) / len(targets)
    return precision, recall
def _compute_ndcg(targets, predictions, k):
    k1 = min(len(targets), k)
    if len(predictions) > k:
        predictions = predictions[:k]
    # compute idcg
    idcg = np.sum(1 / np.log2(np.arange(2, k1 + 2)))
    dcg = 0.0
    for i, p in enumerate(predictions):
        if p in targets:
            dcg += 1 / np.log2(i + 2)
    ndcg = dcg / idcg
    return ndcg
def _compute_hr(targets, predictions, k):
    pred = predictions[:k]
    for i in pred:
        if i in targets:
            return 1
    return 0
def evaluate_ranking(model, test, train=None, k=10):
    all_item = np.arange(test.num_items)
    test = test.tocsr()
    if train is not None:
        train = train.tocsr()
    if not isinstance(k, list):
        ks = [k]
    else:
        ks = k
    precisions = [list() for _ in range(len(ks))]
    recalls = [list() for _ in range(len(ks))]
    ndcgs = [list() for _ in range(len(ks))]
    hrs = [list() for _ in range(len(ks))]
    f1s = [list() for _ in range(len(ks))]
    apks = list()
    for user_id, row in enumerate(test):
        if not len(row.indices):
            continue
        test_neg_candidate = list(set(all_item) - set(row.indices) - set(train[user_id].indices))
        test_neg_nums = len(row.indices) * model._test_neg
        test_neg = np.random.choice(a=test_neg_candidate, size=test_neg_nums, p=None)
        item_ids = np.array(list(set(test_neg) | set(row.indices))).reshape(-1, 1)
        predictions = -model.predict(user_id, item_ids)
        item_ids = item_ids.flatten()
        predictions = item_ids[predictions.argsort()]
        if train is not None:
            rated = set(train[user_id].indices)
        else:
            rated = []
        predictions = [p for p in predictions if p not in rated]
        targets = row.indices
        for i, _k in enumerate(ks):
            precision, recall = _compute_precision_recall(targets, predictions, _k)
            if precision != 0 or recall != 0:
                f1 = 2.0 * precision * recall / (precision + recall)
            else:
                f1 = 0.0
            precisions[i].append(precision)
            recalls[i].append(recall)
            f1s[i].append(f1)
            ndcg = _compute_ndcg(targets, predictions, _k)
            hr = _compute_hr(targets, predictions, _k)
            ndcgs[i].append(ndcg)
            hrs[i].append(hr)
        apks.append(_compute_apk(targets, predictions, k=np.inf))
    precisions = [np.array(i) for i in precisions]
    recalls = [np.array(i) for i in recalls]
    ndcgs = [np.array(i) for i in ndcgs]
    hrs = [np.array(i) for i in hrs]
    f1s = [np.array(i) for i in f1s]
    if not isinstance(k, list):
        precisions = precisions[0]
        recalls = recalls[0]
        ndcgs = ndcgs[0]
        hrs = hrs[0]
        f1s = f1s[0]
    mean_aps = np.mean(apks)
    return precisions, recalls, mean_aps, ndcgs, hrs, f1s
