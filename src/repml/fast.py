from typing import Tuple

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from repml.datasets.trees_ber import prepare_trees_ber


def get_transformed_tree_data(components) -> Tuple[np.array]:
    """Transforms labeled Berlin tree data.

    Closely aligned with notebook 3F, the data is split, fitted and
    transformed, returning all data sets required for model training.

    Arguements:
        components (int): Number of components to keep in PCA.

    Returns:
        Tuple[np.array]: 2D Arrays x_train, y_train, x_test, y_test.
    """
    _, labeled, _ = prepare_trees_ber()

    y = "pflanzjahr"
    X = labeled.columns.to_list()
    X.remove(y)
    train_data, test_data = train_test_split(
        labeled, test_size=0.2, random_state=42, stratify=labeled["gattung_deutsch"]
    )

    num_cols = [
        "kronedurch",
        "stammumfg",
        "baumhoehe",
        "hoehe_zu_krone",
        "hoehe_zu_stamm",
    ]

    cat_cols = [
        "bezirk",
        "baumart",
        "art_dtsch",
        "art_bot",
        "gattung_deutsch",
        "gattung",
        "art_dtsch_infrequent",
        "art_bot_infrequent",
        "gattung_deutsch_infrequent",
        "gattung_infrequent",
        "namenr",
        "namenr_nonum",
        "lat_lon_tile",
    ]
    cols_set = [*num_cols, *cat_cols]

    ct_a = ColumnTransformer(
        transformers=[
            ("imp", IterativeImputer(random_state=42), [0, 1, 2]),
        ],
        remainder="passthrough",
    )
    cat_cols_indices = [
        cols_set.index(col) for col in cat_cols
    ]  # create list of indices for categorical columns
    ct_b = ColumnTransformer(
        transformers=[
            ("scale", StandardScaler(), [0, 1, 2]),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols_indices),
        ],
    )
    prep = Pipeline(
        steps=[
            ("prep_a", ct_a),
            ("prep_b", ct_b),
        ]
    )
    full_prep = Pipeline([("preprocessing", prep), ("ct_pca", PCA(n_components=components))])
    # 71 components contain 85% explained variance -> outdated
    # 19 components contain 75% explained variance

    x_train = full_prep.fit_transform(train_data[cols_set])
    y_train = train_data[y]
    x_test = full_prep.transform(test_data[cols_set])
    y_test = test_data[y]

    return x_train, y_train, x_test, y_test
