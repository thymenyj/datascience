import yaml
from sklearn.datasets import load_iris
from pandas import DataFrame


def data_load():
    with open('params.yaml') as config_file:
        config = yaml.safe_load(config_file)
    
    dataset = load_iris(as_frame=True)

    dataframe = DataFrame(data=dataset['data'], columns=dataset['feature_names'])

    data_raw_csv_path = config['data_load']['data_raw_csv_path']
    dataframe.to_csv(data_raw_csv_path)

if __name__=="__main__":
    data_load()