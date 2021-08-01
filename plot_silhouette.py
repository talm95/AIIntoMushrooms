from sklearn.metrics import silhouette_samples, silhouette_score

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np


def plot_silhouette_graph(data, k_means_clusters, spectral_clusters, agglomerative_clusters, n_clusters):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    fig.set_size_inches(18, 7)
    k_means_silhouette_score = plot_silhouette_sub_graph(data, k_means_clusters, n_clusters, ax1, 'k_means')
    spectral_clusters_silhouette_score = plot_silhouette_sub_graph(data, spectral_clusters, n_clusters, ax2,
                                                                  'spectral_clustering')
    agglomerative_clusters_silhouette_score = plot_silhouette_sub_graph(data, agglomerative_clusters, n_clusters, ax3,
                                                                        'agglomerative_clustering')
    plt.show()
    return [k_means_silhouette_score, spectral_clusters_silhouette_score, agglomerative_clusters_silhouette_score]


def plot_silhouette_sub_graph(data, predicted_clusters, n_clusters, ax, cluster_type):

    ax.set_xlim([-0.1, 1])
    ax.set_ylim([0, len(data) + (n_clusters + 1) * 10])

    silhouette_avg = silhouette_score(data, predicted_clusters)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score of " + cluster_type + " is :", silhouette_avg)

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(data, predicted_clusters)

    y_lower = 10
    for i in range(n_clusters):
        ith_cluster_silhouette_values = \
            sample_silhouette_values[predicted_clusters == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax.set_title("The silhouette plot for " + cluster_type)
    ax.set_xlabel("The silhouette coefficient values")
    ax.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax.set_yticks([])  # Clear the yaxis labels / ticks
    ax.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    plt.suptitle(("Silhouette score "
                  "with %d clusters" % n_clusters),
                 fontsize=14, fontweight='bold')
    return silhouette_avg


def plot_silhouette_score_per_cluster_num(clusters_nums, silhouette_scores):
    plt.plot(clusters_nums, silhouette_scores[:, 0], 'b', label="k means")
    plt.plot(clusters_nums, silhouette_scores[:, 1], 'r', label="spectral clustering")
    plt.plot(clusters_nums, silhouette_scores[:, 2], 'k', label="agglomertaive clustering")
    plt.legend(loc='upper left')
    plt.xlabel('clusters number')
    plt.ylabel('silhouette score')
    plt.title('silhouette scores by clusters number')
    plt.show()
