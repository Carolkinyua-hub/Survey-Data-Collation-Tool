import prince
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def run_famd(X, n_components=8):
    famd = prince.FAMD(
        n_components=n_components,
        random_state=42
    )

    X_famd = famd.fit_transform(X)
    return famd, X_famd


def auto_k(X, k_min=2, k_max=8):
    best_k = 2
    best_score = -1

    for k in range(k_min, k_max + 1):
        km = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )

        labels = km.fit_predict(X)
        score = silhouette_score(X, labels)

        if score > best_score:
            best_score = score
            best_k = k

    return best_k, best_score


def cluster_data(X_famd, k):
    km = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = km.fit_predict(X_famd)

    return km, labels