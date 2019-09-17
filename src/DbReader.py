import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_STATE = 1
TRAIN_SIZE = 0.8


class DbReader:
    """do pobierania i wczytywania danych"""

    def __init__(self, train_size=TRAIN_SIZE):
        self.red_wine_path = ".//Datasets//winequality-red.csv"
        self.white_wine_path = ".//Datasets//winequality-white.csv"

        self.red_wine_data = pd.read_csv(self.red_wine_path, sep=';')  # 1599 lines
        self.white_wine_data = pd.read_csv(self.white_wine_path, sep=';')  # 4898 lines

        # Podzial 80 : 20
        red_sep_point = int(train_size * len(self.red_wine_data))
        white_sep_point = int(train_size * len(self.white_wine_data))

        self.train_red = self.red_wine_data[:red_sep_point]
        self.test_red = self.red_wine_data[red_sep_point:]

        self.train_white = self.white_wine_data[:white_sep_point]
        self.test_white = self.white_wine_data[white_sep_point:]

        # Scalanie
        train_frames = [self.train_red, self.train_white]
        train = pd.concat(train_frames)

        test_frames = [self.test_red, self.test_white]
        test = pd.concat(test_frames)

        features_all = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
                        'chlorides', 'free sulfur dioxide', 'total sulfur dioxide',
                        'density', 'pH', 'sulphates', 'alcohol']
        self.features = ['volatile acidity', 'residual sugar', 'density', 'alcohol']
        self.output = 'quality'

        # Podzial na train, test i val
        train_X, val_X, train_y, val_y = train_test_split(train[self.features], train[self.output],
                                                          train_size=TRAIN_SIZE, random_state=RANDOM_STATE)

        test_X = test[self.features]
        test_y = test[self.output]

        self.train_X, self.val_X, self.test_X = train_X, val_X, test_X
        self.train_y, self.val_y, self.test_y = train_y, val_y, test_y

    def get_splitted_data(self):
        """:returns: train_X, val_X, test_X, train_y, val_y, test_y"""
        return self.train_X, self.val_X, self.test_X, self.train_y, self.val_y, self.test_y

    def get_full_data(self):
        return pd.concat([self.red_wine_data, self.white_wine_data])

    @staticmethod
    def pack_data(y):
        new_y = pd.Series([])
        new_y_list = list(y)
        for i, elem in enumerate(new_y_list):
            if elem <= 3:
                new_y[i] = 0
            elif elem <= 7:
                new_y[i] = 1
            else:
                new_y[i] = 2
        return new_y

    def get_red_data(self):
        features = self.features
        output = self.output
        train_X, val_X, train_y, val_y = train_test_split(self.train_red[features], self.train_red[output],
                                                          train_size=TRAIN_SIZE, random_state=RANDOM_STATE)
        return train_X, val_X, self.test_red[features], \
               train_y, val_y, self.test_red[output]

    def get_white_data(self):
        features = self.features
        output = self.output
        train_X, val_X, train_y, val_y = train_test_split(self.train_white[features], self.train_white[output],
                                                          train_size=TRAIN_SIZE, random_state=RANDOM_STATE)
        return train_X, val_X, self.test_white[features], \
               train_y, val_y, self.test_white[output]

    def get_packed_data(self):
        train_y = self.pack_data(self.train_y)
        val_y = self.pack_data(self.val_y)
        test_y = self.pack_data(self.test_y)

        return self.train_X, self.val_X, self.test_X, train_y, val_y, test_y

    def get_red_packed_data(self):
        train_X, val_X, test_X, train_y, val_y, test_y = self.get_red_data()

        train_y = self.pack_data(self.train_y)
        val_y = self.pack_data(self.val_y)
        test_y = self.pack_data(self.test_y)

        return train_X, val_X, test_X, train_y, val_y, test_y

    def get_white_packed_data(self):
        train_X, val_X, test_X, train_y, val_y, test_y = self.get_white_data()

        train_y = self.pack_data(self.train_y)
        val_y = self.pack_data(self.val_y)
        test_y = self.pack_data(self.test_y)

        return train_X, val_X, test_X, train_y, val_y, test_y
