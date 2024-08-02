from __future__ import annotations

from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder


def check_for_components(max_components, data, prep_pipeline, variance) -> int:
    """Check ellbow curve for appropriate number of components.

    Args:
        max_components (int): Maximum number of components to investigate.
        data (pd.DataFrame): Data used with the pipeline.
        prep_pipeline (Pipeline): Pipeline used to transform the data.
        variance (float): Desired explained variance.

    Returns:
        int: Number of components needed to achieve the desired variance.
    """
    pre_transform = data.shape
    data_post_transform = prep_pipeline.fit_transform(data)
    post_transform = data_post_transform.shape
    print(f"Dimension vorher (R,C): ({pre_transform[0]}, {pre_transform[1]})")
    print(f"Dimension nach der Transformation (R,C): ({post_transform[0]}, {post_transform[1]})")

    pca = PCA(n_components=max_components, random_state=42)
    pca.fit_transform(data_post_transform)

    cumsum = np.cumsum(pca.explained_variance_ratio_)
    print("Cumulative sum of explained variance:", cumsum)
    d = np.argmax(cumsum >= variance) + 1
    if np.max(cumsum) < variance:
        print(f"{max_components} do not explain {variance} of the variance, only {np.max(cumsum)}.")
        d = max_components
    print(f"{d} components are required to achieve >{variance} explained variance.")
    plt.plot(cumsum, linewidth=3)
    plt.axis([0, max_components - 1, 0, 1])
    plt.xlabel("Dimensions")
    plt.ylabel("Explained Variance")
    if not np.max(cumsum) < variance:
        plt.plot([d, d], [0, variance], "k:")
        plt.plot([0, d], [variance, variance], "k:")
        plt.plot(d, variance, "ko")
    plt.grid(True)
    plt.show()

    return d
