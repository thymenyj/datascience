import yaml
from pandas import read_csv, DataFrame
from numpy import c_
from sklearn.model_selection import train_test_split


def featurize(config):
    seed = config['base']['seed']

    data_raw_csv_path = config['data_load']['data_raw_csv_path']
    iris_dataframe =_load_iris_dataframe(data_raw_csv_path)

    X = _get_training_data(iris_dataframe)
    y = _get_testing_data(iris_dataframe)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=seed)

    train_csv_path = config['featurize']['data_train_csv_path'] 
    test_csv_path = config['featurize']['data_test_csv_path'] 
    _save_train_data_to_csv(X_train, y_train, train_csv_path)
    _save_test_data_to_csv(X_test, y_test, test_csv_path)


def _get_training_data(iris_dataframe):
    X = iris_dataframe.drop(['target','species'], axis=1)
    return X.to_numpy()[:, (2,3)]


def _load_iris_dataframe(path):
    iris_dataset = read_csv(path,index_col=0)
    iris_dataframe = DataFrame(iris_dataset)
    return iris_dataframe


def _save_train_data_to_csv(X_train, y_train, path):
    data = c_[X_train, y_train]
    columns = ['petal_length','petal_width', 'target']
    train_data = DataFrame(data=data, columns=columns)
    train_data.to_csv(path)


def _save_test_data_to_csv(X_test, y_test, path):
    data = c_[X_test, y_test]
    columns = ['petal_length','petal_width', 'target']
    test_data = DataFrame(data=data, columns=columns)
    test_data.to_csv(path)


def _get_testing_data(iris_dataframe):
    return iris_dataframe['target']


if __name__=="__main__":
    with open('params.yaml') as config_file:
        config = yaml.safe_load(config_file)

    featurize(config)