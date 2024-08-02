"""Clustering algorithm visualizations adapted from Géron, Chapter 9.

Géron (2019): Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow.
"""
import matplotlib.pyplot as plt
import numpy as np


def plot_birchs(X, birchs, figsize=(8, 6)):
    """Plot multiple BIRCH cluster results alongside.

    Args:
        X (np.array): 2D array to use for clustering.
        birchs (list[sklearn.cluster.Birch]): Instance(s) of fitted Birch estimators.
        figsize (tuple, optional): Size of the figure plotted. Defaults to (8, 6).
    """
    fig = plt.figure(figsize=figsize)
    rows, cols = (int(round(len(birchs) / 2, 0)), 2)
    for ix, bir in enumerate(birchs):
        labels = bir.labels_
        centroids = bir.subcluster_centers_
        n_clusters = np.unique(labels).size
        ax = fig.add_subplot(rows, cols, ix + 1)
        colors_ = plt.cm.jet(np.linspace(0, 1, n_clusters))
        for _, k, col in zip(centroids, range(n_clusters), colors_):
            mask = labels == k
            ax.scatter(X[mask, 0], X[mask, 1], c="w", edgecolor=col, marker=".", alpha=0.9)
        ax.set_autoscaley_on(False)
        ax.set_title(f"{n_clusters} clusters")
    fig.suptitle("BIRCH clustering")
    plt.tight_layout()


def plot_kmeans(X, kmeans, figsize=(8, 8)):
    """Plot muliple k-means side-by-side.

    Args:
        X (np.array): 2D array to use for clustering.
        kmeans (sklearn.cluster.KMeans): Instance(s) of fitted KMeans estimators.
        figsize (tuple, optional): Size of the figure plotted. Defaults to (8, 8).
    """
    fig = plt.figure(figsize=figsize)
    rows, cols = (int(round(len(kmeans) / 2, 0)), 2)
    for ix, km in enumerate(kmeans):
        plt.subplot(rows, cols, ix + 1)
        _plot_decision_boundaries(km, X)
        plt.title(f"{km.n_clusters} clusters")
    fig.suptitle("K-Means clustering")
    plt.tight_layout()


def _plot_data(X):
    plt.plot(X[:, 0], X[:, 1], "k.", markersize=2)


def _plot_centroids(centroids, weights=None, circle_color="w", cross_color="k"):
    if weights is not None:
        centroids = centroids[weights > weights.max() / 10]
    plt.scatter(
        centroids[:, 0],
        centroids[:, 1],
        marker="o",
        s=35,
        linewidths=8,
        color=circle_color,
        zorder=10,
        alpha=0.9,
    )
    plt.scatter(
        centroids[:, 0],
        centroids[:, 1],
        marker="x",
        s=2,
        linewidths=12,
        color=cross_color,
        zorder=11,
        alpha=1,
    )


def _plot_decision_boundaries(
    clusterer, X, resolution=1000, show_centroids=True, show_xlabels=True, show_ylabels=True
):
    mins = X.min(axis=0) - 0.1
    maxs = X.max(axis=0) + 0.1
    xx, yy = np.meshgrid(
        np.linspace(mins[0], maxs[0], resolution), np.linspace(mins[1], maxs[1], resolution)
    )
    Z = clusterer.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(Z, extent=(mins[0], maxs[0], mins[1], maxs[1]), cmap="Pastel2")
    plt.contour(Z, extent=(mins[0], maxs[0], mins[1], maxs[1]), linewidths=1, colors="k")
    _plot_data(X)
    if show_centroids:
        _plot_centroids(clusterer.cluster_centers_)

    if show_xlabels:
        plt.xlabel("$x_1$", fontsize=14)
    else:
        plt.tick_params(labelbottom=False)
    if show_ylabels:
        plt.ylabel("$x_2$", fontsize=14, rotation=0)
    else:
        plt.tick_params(labelleft=False)
