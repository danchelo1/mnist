import numpy as np
import pandas as pd
from typing import Optional, List
from sklearn.model_selection import train_test_split
from sklearn.base import TransformerMixin,RegressorMixin
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler


def root_mean_squared_logarithmic_error(y_true, y_pred, a_min=1.):
    """
    Вычисляет Root Mean Squared Logarithmic Error (RMSLE).

    Формула: sqrt(mean((log(y_pred + 1) - log(y_true + 1))^2))

    :param y_true: истинные значения цен (должны быть положительными)
    :param y_pred: предсказанные значения
    :param a_min: минимальное значение для клиппинга предсказаний (по умолчанию 1.0)
    :return: значение RMSLE метрики
    """

    # Проверка: цены не могут быть отрицательными
    if np.any(y_true < 0):
        raise ValueError("y_true содержит отрицательные значения. Цена не может быть отрицательной!")

    # Клиппинг предсказаний: заменяем значения меньше a_min на a_min
    y_pred_clipped = np.clip(y_pred, a_min, None)

    # Вычисляем логарифмическую ошибку
    # Формула: log(y_true) - log(y_pred)
    log_error = np.log(y_true) - np.log(y_pred_clipped)

    # Вычисляем среднеквадратичную ошибку
    squared_log_error = log_error ** 2
    mean_squared_log_error = np.mean(squared_log_error)

    # Берем квадратный корень
    metric = np.sqrt(mean_squared_log_error)

    return metric

data = pd.read_csv('./data.csv')
target_column = "Sale_Price"
np.random.seed(24)

test_size = 0.2
data_train, data_test, Y_train, Y_test = train_test_split(
    data[data.columns.drop("Sale_Price")],
    np.array(data["Sale_Price"]),
    test_size=test_size,
    random_state=24)

print(f"Train : {data_train.shape} {Y_train.shape}")
print(f"Test : {data_test.shape} {Y_test.shape}")
continuous_columns = [key for key in data.keys() if data[key].dtype in ("int64", "float64")]
categorical_columns = [key for key in data.keys() if data[key].dtype == "object"]

continuous_columns.remove(target_column)


class BaseDataPreprocessor(TransformerMixin):
    def __init__(self, needed_columns = None):
        """
        :param needed_columns: if not None select these columns from the dataframe
        """
        self.scaler = StandardScaler()
        self.needed_columns = needed_columns

    def fit(self, data, *args):
        """
        Prepares the class for future transformations
        :param data: pd.DataFrame with all available columns
        :return: self
        """
        if self.needed_columns is None:
            self.columns_ = data.columns
        else:
            self.columns_ = self.needed_columns


        self.scaler.fit(data[self.columns_])
        return self

    def transform(self, data) :
        """
        Transforms features so that they can be fed into the regressors
        :param data: pd.DataFrame with all available columns
        :return: np.array with preprocessed features
        """
        X = data[self.columns_]

        X_scaled = self.scaler.transform(X)

        return X_scaled

preprocessor = BaseDataPreprocessor(needed_columns=continuous_columns)

X_train = preprocessor.fit_transform(data_train)
X_test = preprocessor.transform(data_test)


class SGDLinearRegressor(RegressorMixin):
    def __init__(self, lr=0.01, regularization=1., delta_converged=1e-2, max_steps=1000, batch_size=64):
        self.lr = lr
        self.regularization = regularization
        self.max_steps = max_steps
        self.delta_converged = delta_converged
        self.batch_size = batch_size

        self.W = None
        self.b = None

    def fit(self, X, Y):
        X = np.asarray(X)
        Y = np.asarray(Y).reshape(-1, 1)
        n_samples, n_features = X.shape
        self.b = 0
        self.W = np.zeros((n_features, 1))

        for i in range(self.max_steps):
            old_W = self.W.copy()
            old_b = self.b
            idx = np.arange(n_samples)
            np.random.shuffle(idx)

            for start_idx in range(0, n_samples, self.batch_size):
                end_idx = min(start_idx + self.batch_size, n_samples)
                batch = idx[start_idx:end_idx]
                X_b = X[batch]
                Y_b = Y[batch]
                batch_size = len(batch)
                Y_pred = X_b @ self.W + self.b
                error = Y_pred - Y_b
                grad_W = (2.0 / batch_size) * (X_b.T @ error) + 2 * self.regularization * self.W
                grad_b = (2.0 / batch_size) * np.sum(error)
                self.W = self.W - self.lr * grad_W
                self.b = self.b - self.lr * grad_b

            diff_W = self.W - old_W
            diff_b = self.b - old_b

            norm_diff = np.sqrt(np.sum(diff_W ** 2) + diff_b ** 2)

            if norm_diff < self.delta_converged:
                break

    def predict(self, X):
        X_np = np.asarray(X)

        Pred = X_np @ self.W + self.b
        return np.ravel(Pred)


model = SGDLinearRegressor()
model.fit(X_train, Y_train)

prediction = model.predict(X_test)
print(Y_test.shape, prediction.shape)
print("MAE : ", mean_absolute_error(Y_test, prediction))
print("Mean log : ", root_mean_squared_logarithmic_error(Y_test, prediction))