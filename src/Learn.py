import warnings
from time import time

from sklearn.model_selection import GridSearchCV

from DbReader import DbReader
from Model import LinearRegressionModel, LogisticRegressionModel, SVMModel, RandomForestModel
from Plotter import Plotter

stats_file_path = ".//stat_log.txt"


def main():
    warnings.simplefilter("ignore")
    warnings.warn("deprecated", DeprecationWarning)

    # INIT
    reader = DbReader()
    plotter = Plotter(reader)

    # OBA RODZAJE
    # 3 klasy
    train_X, val_X, test_X, train_y, val_y, test_y = reader.get_packed_data()
    # 10 klas
    # train_X, val_X, test_X, train_y, val_y, test_y = reader.get_splitted_data()

    # ONLY RED
    # 3 klasy
    # train_X, val_X, test_X, train_y, val_y, test_y = reader.get_red_packed_data()
    # 10 klas
    # train_X, val_X, test_X, train_y, val_y, test_y = reader.get_red_data()

    # ONLY WHITE
    # 3 klasy
    # train_X, val_X, test_X, train_y, val_y, test_y = reader.get_white_packed_data()
    # 10 klas
    # train_X, val_X, test_X, train_y, val_y, test_y = reader.get_white_data()

    models = [
        LinearRegressionModel(),
        LogisticRegressionModel(),
        SVMModel(),
        RandomForestModel()
    ]
    model_names = [
        'LinRegModel',
        'LogRegModel',
        'SVMModel',
        'RandomForestModel'
    ]
    params = [
        dict(fit_intercept=[True, False], normalize=[True, False]),  # true, 1, false
        dict(tol=[1e-3, 1e-4, 1e-5], C=[1, 10, 20, 30], fit_intercept=[True, False], warm_start=[True, False]),
        # 1e-4, 20, true, true
        dict(C=[2, 5, 10, 20, 30, 50], gamma=[0.1, 0.01, 0.0001, 0.00001]),
        # gamma:rb, pl, sim
        # 20, 1e-5
        dict(n_estimators=[50, 100, 165, 200, 300, 500, 700], max_depth=[10, 20, 33, 40])  # 33, 165
    ]
    best_params = []
    times = []

    # GSCV
    for i, model in enumerate(models):
        clf = GridSearchCV(models[i].model, params[i], cv=5, refit=False)
        clf.fit(train_X, train_y)
        print(f"{model_names[i]}")
        print(f"Best params: {clf.best_params_}")
        print(f"Best score:  {clf.best_score_}")
        print(f"Worst score: {clf.cv_results_['mean_test_score'].min()}\n")
        best_params.append(clf.best_params_)

    # LEARN WITH BEST_PARAMS
    for i, model in enumerate(models):
        model.set_estimator(best_params[i])
        t_start = time()
        model.fit(train_X, train_y)
        t_end = time()
        mae = model.get_mae(val_X, val_y)
        mse = model.get_mse(val_X, val_y)
        times.append(t_end - t_start)
        model.save()
        print(f"Model name: {model.name:27}  MAE: {mae:{6}.{4}}  MSE: {mse:{6}.{4}}  t: {times[i]}")

    # SAVE LOGS TO FILE
    with open(stats_file_path, "a") as stat_file:
        stat_file.write("Duration times of model fitting:\n")
        for i, model in enumerate(model_names):
            stat_file.write((f"\t{model_names[i]:{22}}: {times[i]}\n"))
        stat_file.write('\n')


if __name__ == "__main__":
    main()
