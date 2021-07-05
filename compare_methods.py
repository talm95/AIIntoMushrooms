import pandas as pd


def cluster_confusion_matrix(clusters, real_labels):
    df = pd.DataFrame({'Labels': real_labels, 'Clusters': clusters})
    ct = pd.crosstab(df['Labels'], df['Clusters'])
    return ct
