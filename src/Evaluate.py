from DbReader import DbReader
from Model import LinearRegressionModel, LogisticRegressionModel, SVMModel, RandomForestModel
from Plotter import Plotter

stats_file_path = ".//stat_log.txt"


def main():
    reader = DbReader()
    plotter = Plotter(reader)

    # LABELS
    labels = list(range(3))
    # labels = list(range(1, 11))

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

    # scaler = StandardScaler()
    # scaler.fit_transform(train_X, train_y)
    # scaler.transform(val_X, val_y)

    models = [LinearRegressionModel(), LogisticRegressionModel(), SVMModel(),
              RandomForestModel()]

    model_names = [
        'LinRegModel',
        'LogRegModel',
        'SVMModel',
        'RandomForestModel'
    ]
    # test_X, test_y = train_X, train_y
    # zapis do pliku MAE, MSE dla test
    with open(stats_file_path, "a") as stat_file:
        stat_file.write("Model errors:\n")
        for i, model in enumerate(models):
            model.load()
            mae = model.get_mae(test_X, test_y)
            mse = model.get_mse(test_X, test_y)
            print(f"Model name: {model.name:27}   MAE: {mae:{6}.{4}}    MSE: {mse:{6}.{4}}")
            stat_file.write((f"\t{model_names[i]:{22}}: "))
            stat_file.write(f"MAE = {mae:{8}.{4}}  ")
            stat_file.write(f"MSE = {mse:{8}.{4}}  ")
            stat_file.write(f"SCORE = {model.score(test_X, test_y)}\n")

    plotter.heatmap()
    plotter.kdeplot()
    plotter.pairplot()
    plotter.confusion_matrix(test_y, model.predict(test_X))
    plotter.classification_report(test_y, model.predict(test_X), labels)


if __name__ == "__main__":
    main()
