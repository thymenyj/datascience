from turtle import pen
import yaml
import pickle
from pandas import read_csv, DataFrame
from sklearn.linear_model import LogisticRegression


def train(config):
    data_train_csv_path = config['featurize']['data_train_csv_path']
    iris_train = _load_iris_train(data_train_csv_path)

    X = iris_train[['petal_length', 'petal_width']].to_numpy()
    y = iris_train['target'].to_numpy()

    penalty = config['train']['penalty']
    random_state = config['base']['seed']
    solver = config['train']['solver']
    logistic_regression = LogisticRegression(penalty=penalty, solver=solver, random_state=random_state)
    logistic_regression.fit(X, y)

    logistic_regression_path = config['train']['logistic_regression_path'] 
    pickle.dump(logistic_regression, open(logistic_regression_path, 'wb'))


def _load_iris_train(path):
    iris_train = read_csv(path, index_col=0)
    return DataFrame(iris_train)

if __name__=="__main__":
    with open('params.yaml') as config_file:
        config = yaml.safe_load(config_file)

    train(config)