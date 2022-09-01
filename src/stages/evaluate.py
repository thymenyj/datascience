from yaml import safe_load
from json import dump
from pickle import load
from pandas import read_csv, DataFrame
from sklearn.metrics import accuracy_score


def evaluate(config):
    logistic_regression_path = config['train']['logistic_regression_path']
    model = _load_model(logistic_regression_path)

    train_data_path = config['featurize']['data_train_csv_path']
    X_train, y_train = _load_train_data(train_data_path)

    test_data_path = config['featurize']['data_test_csv_path']
    X_test, y_test = _load_test_data(test_data_path)
    
    train_prediction = model.predict(X_train)
    test_prediction = model.predict(X_test)

    train_metrics = {
        "train": {
            "accuracy": accuracy_score(y_train, train_prediction)
        },
        "test": {
            "accuracy": accuracy_score(y_test, test_prediction)
        }
    }

    metrics_path = config['evaluate']['metrics_path']
    with open(metrics_path, 'w') as file:
        dump(train_metrics, file)


    
def _load_iris_train(path):
    iris_train = read_csv(path, index_col=0)
    return DataFrame(iris_train)

def _load_model(path):
    with open(path, 'rb') as model_file:
        model = load(model_file)
    return model


def _load_train_data(path):
    train_data = read_csv(path, index_col=0)
    X_train = train_data[['petal_length','petal_width']].to_numpy()
    y_train = train_data['target'].to_numpy()
    return X_train, y_train


def _load_test_data(path):
    test_data = read_csv(path, index_col=0)
    X_test = test_data[['petal_length','petal_width']].to_numpy()
    y_test = test_data['target'].to_numpy()
    return X_test, y_test


if __name__=="__main__":
    with open('params.yaml') as config_file:
        config = safe_load(config_file)

    evaluate(config)