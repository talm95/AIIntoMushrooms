from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np


def cluster_confusion_matrix(clusters, real_labels):
    # real_labels = np.array(real_labels[0:1000], dtype=object)
    df = pd.DataFrame({'Labels': real_labels, 'Clusters': clusters})
    ct = pd.crosstab(df['Labels'], df['Clusters'])
    print(ct)
