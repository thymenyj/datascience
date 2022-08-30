import yaml
from sklearn.datasets import load_iris
from pandas import DataFrame
from numpy import c_


def data_load(config):
    iris_dataset = load_iris(as_frame=True)

    data = c_[iris_dataset['data'], iris_dataset['target']]
    columns = iris_dataset['feature_names'] + ['target']
    iris_dataframe = DataFrame(data=data, columns=columns)

    _name_target(iris_dataframe)

    data_raw_csv_path = config['data_load']['data_raw_csv_path']
    iris_dataframe.to_csv(data_raw_csv_path)


def _name_target(iris_dataframe):
    species = []

    for index, row in iris_dataframe.iterrows():
        target = row['target']
        if target == 0:
            species.append("setosa")
        elif target == 1:
            species.append('versicolor')
        else:
            species.append('virginica')

    iris_dataframe['species'] = species


if __name__=="__main__":
    with open('params.yaml') as config_file:
        config = yaml.safe_load(config_file)

    data_load(config)
