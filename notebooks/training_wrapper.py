"""Should be more sophisticated"""
# TODO make this less brute force. Use WanDB's hyperopt capabilities!

from wandb.sklearn import plot_precision_recall, plot_feature_importances
import wandb
from prettytable import PrettyTable
import numpy as np
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier
from dataloading import get_all_data


PATH_DATA = '../data/all_vectors.csv'
TEST_SIZE = .2
UNIQUE_COLUMNS = False

def iter():
    (X_train, X_test, y_train, y_test), (info_data, info_label) = get_all_data(
        path_all_vectors=PATH_DATA,
        test_size=TEST_SIZE,
        unique=UNIQUE_COLUMNS
        )


    # MODEL INIT
    model = RandomForestClassifier(
        verbose=True,
        max_depth=8,
        max_features=0.4,
        n_estimators=100,
        bootstrap=True,
        oob_score=False,
        )

    # TRAINING

    _ = model.fit(X_train, y_train)


    count = defaultdict(int)
    for tree in model.estimators_:
        # print(tree.get_depth(), end=', ')
        count[tree.get_depth()] += 1
    count = dict(count)
    for item, key in count.items():
        print(f'{key} trees have length {item}')


    # get predictions
    y_pred = model.predict(X_test)

    matches = np.count_nonzero(y_test == y_pred)
    print(f'Accuracy: {100 * matches / len(y_test)} %')


    table = PrettyTable()
    table.field_names = ["Property", "Importance (%)"]

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    for i in indices:
        imp = 100 * importances[i]

        if imp > 0:
            if int(imp) == 0:
                imp = '<1 %'
            else:
                imp = f'{int(imp)} %'
        else:
            imp = '0 %'


        table.add_row([info_data[i], imp])

    print(table)

    # # WanDB init

    try:


        # start a new wandb run and add your model hyperparameters
        wandb.init(project='Halado_Adatelemzes_Labor', config=model.get_params())

        # Add additional configs to wandb
        wandb.config.update({"test_size" : TEST_SIZE,
                            "train_len" : len(X_train),
                            "test_len" : len(X_test)})

        from wandb.sklearn import plot_class_proportions, plot_learning_curve, plot_roc

        y_probas = model.predict_proba(X_test)

        # log additional visualisations to wandb
        plot_class_proportions(y_train, y_test, info_data)
        plot_learning_curve(model, X_train, y_train)
        plot_roc(y_test, y_probas, info_data)
        plot_precision_recall(y_test, y_probas, info_data)
        plot_feature_importances(model)

        # Finish the wandb run
        wandb.finish()
    except:
        pass
    try:
        wandb.finish()
    except:
        pass
