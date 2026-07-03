import numpy as np
from munkres import Munkres
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.cluster import KMeans


def evaluation(y_true, y_pred):
    """
    evaluate the clustering performance
    :param y_true: ground truth
    :param y_pred: prediction
    :returns acc, nmi, ari, f1:
    - accuracy
    - normalized mutual information
    - adjust rand index
    - f1 score
    """
    nmi = nmi_score(y_true, y_pred, average_method='arithmetic')
    ari = ari_score(y_true, y_pred)

    y_true = y_true - np.min(y_true)
    l1 = list(set(y_true))
    num_class1 = len(l1)
    l2 = list(set(y_pred))
    num_class2 = len(l2)
    ind = 0
    if num_class1 != num_class2:
        for i in l1:
            if i in l2:
                pass
            else:
                y_pred[ind] = i
                ind += 1
    l2 = list(set(y_pred))
    num_class2 = len(l2)
    if num_class1 != num_class2:
        print('error: class mismatch, return default metrics')
        return 0.0, nmi, ari, 0.0
    cost = np.zeros((num_class1, num_class2), dtype=int)
    for i, c1 in enumerate(l1):
        mps = [i1 for i1, e1 in enumerate(y_true) if e1 == c1]
        for j, c2 in enumerate(l2):
            mps_d = [i1 for i1 in mps if y_pred[i1] == c2]
            cost[i][j] = len(mps_d)
    m = Munkres()
    cost = cost.__neg__().tolist()
    indexes = m.compute(cost)
    new_predict = np.zeros(len(y_pred))
    for i, c in enumerate(l1):
        c2 = l2[indexes[i][1]]
        ai = [ind for ind, elm in enumerate(y_pred) if elm == c2]
        new_predict[ai] = c
    acc = accuracy_score(y_true, new_predict)
    f1 = f1_score(y_true, new_predict, average='macro')

    return acc, nmi, ari, f1


def eva(k, labels, emb):
    embeddings = emb.cpu().data.numpy()
    model = KMeans(n_clusters=k, n_init=20)
    cluster_id = model.fit_predict(embeddings)
    acc, nmi, ari, f1 = evaluation(labels, cluster_id)
    return acc, nmi, ari, f1


def _to_numpy_emb(emb):
    if hasattr(emb, "detach"):
        return emb.detach().cpu().numpy()
    if hasattr(emb, "cpu"):
        return emb.cpu().data.numpy()
    return np.asarray(emb)


def _align_cluster_to_ref(ref, pred, k):
    ref = np.asarray(ref)
    pred = np.asarray(pred)
    cost = np.zeros((k, k), dtype=int)
    for i in range(k):
        ref_mask = (ref == i)
        for j in range(k):
            cost[i, j] = np.sum(ref_mask & (pred == j))
    m = Munkres()
    indexes = m.compute((-cost).tolist())
    mapping = {}
    for ref_label, pred_label in indexes:
        mapping[pred_label] = ref_label
    aligned = np.array([mapping.get(x, x) for x in pred])
    return aligned, mapping


def eva_with_diagnostics(k, labels, emb, prev_cluster_id=None, prev_centers=None,
                         sample_size=3000, random_state=42):
    embeddings = _to_numpy_emb(emb)

    model = KMeans(n_clusters=k, n_init=20, random_state=random_state)
    cluster_id = model.fit_predict(embeddings)
    centers = model.cluster_centers_

    if labels is not None:
        acc, nmi, ari, f1 = evaluation(np.asarray(labels).copy(), cluster_id.copy())
    else:
        acc, nmi, ari, f1 = 0.0, 0.0, 0.0, 0.0

    counts = np.bincount(cluster_id, minlength=k)
    empty_clusters = int(np.sum(counts == 0))
    max_cluster_ratio = float(np.max(counts) / max(1, len(cluster_id)))
    cluster_size_cv = float(np.std(counts) / (np.mean(counts) + 1e-12))

    unique_num = len(np.unique(cluster_id))
    if unique_num > 1 and len(cluster_id) > unique_num:
        if len(cluster_id) > sample_size:
            rng = np.random.RandomState(random_state)
            idx = rng.choice(len(cluster_id), size=sample_size, replace=False)
            emb_sample = embeddings[idx]
            cluster_sample = cluster_id[idx]
        else:
            emb_sample = embeddings
            cluster_sample = cluster_id

        if len(np.unique(cluster_sample)) > 1:
            silhouette = float(silhouette_score(emb_sample, cluster_sample))
        else:
            silhouette = float("nan")

        dbi = float(davies_bouldin_score(embeddings, cluster_id))
        ch = float(calinski_harabasz_score(embeddings, cluster_id))
    else:
        silhouette = float("nan")
        dbi = float("nan")
        ch = float("nan")

    if prev_cluster_id is not None and len(prev_cluster_id) == len(cluster_id):
        aligned_cluster_id, mapping = _align_cluster_to_ref(prev_cluster_id, cluster_id, k)
        epoch_nmi = float(nmi_score(prev_cluster_id, cluster_id, average_method='arithmetic'))
        switch_rate = float(np.mean(aligned_cluster_id != prev_cluster_id))

        if prev_centers is not None and prev_centers.shape == centers.shape:
            aligned_centers = np.zeros_like(centers)
            for cur_label, ref_label in mapping.items():
                if ref_label < k and cur_label < k:
                    aligned_centers[ref_label] = centers[cur_label]
            center_shift = float(np.mean(np.linalg.norm(aligned_centers - prev_centers, axis=1)))
        else:
            center_shift = float("nan")
    else:
        epoch_nmi = float("nan")
        switch_rate = float("nan")
        center_shift = float("nan")

    diag = {
        "silhouette": silhouette,
        "dbi": dbi,
        "ch": ch,
        "empty_clusters": empty_clusters,
        "max_cluster_ratio": max_cluster_ratio,
        "cluster_size_cv": cluster_size_cv,
        "epoch_nmi": epoch_nmi,
        "switch_rate": switch_rate,
        "center_shift": center_shift,
    }

    return acc, nmi, ari, f1, diag, cluster_id, centers