"""Should be more sophisticated"""
# TODO make this less brute force. Use WanDB's hyperopt capabilities!

import numpy as np
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier
from dataloading import get_all_data


PATH_DATA = '../data/all_vectors.csv'
TEST_SIZE = .2
UNIQUE_COLUMNS = False

def iter(max_depth, max_features, n_estimators):
    (X_train, X_test, y_train, y_test), (_, _) = get_all_data(
        path_all_vectors=PATH_DATA,
        test_size=TEST_SIZE,
        unique=UNIQUE_COLUMNS
        )


    # MODEL INIT
    model = RandomForestClassifier(
        verbose=True,
        max_depth=max_depth,
        max_features=max_features,
        n_estimators=n_estimators,
        bootstrap=True,
        oob_score=False,
        )

    # TRAINING

    _ = model.fit(X_train, y_train)

    # LOGGING
    count = defaultdict(int)
    for tree in model.estimators_:
        count[tree.get_depth()] += 1
    count = dict(count)
    for item, key in count.items():
        print(f'{key} trees have length {item}')

    # get predictions
    y_pred = model.predict(X_test)

    matches = np.count_nonzero(y_test == y_pred)
    print(f'Accuracy: {100 * matches / len(y_test)} %')
