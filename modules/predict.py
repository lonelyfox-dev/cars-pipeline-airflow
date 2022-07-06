import pandas as pd
import dill
import json

import os
from datetime import datetime

import logging

root = os.environ.get('PROJECT_PATH', '.')


def assemble_data_from_dir(base_dir: str) -> pd.DataFrame:
    data = None
    for f in os.listdir(base_dir):
        entry_path = f'{base_dir}/{f}'

        with open(entry_path, 'r') as entry_file:
            entry = json.load(entry_file)
            entry = pd.DataFrame([entry])

        if data is None:
            data = entry
        else:
            data = pd.concat([data, entry], axis='index')

    return data


def get_oldest_file_from_dir(base_dir: str) -> str:
    file_timestamps = {}
    for f in os.listdir(base_dir):
        created_date = f.split('_')[-1].split('.')[0]
        timestamp = datetime.strptime(created_date, "%Y%m%d%H%M")

        file_timestamps[f] = timestamp

    latest_file_path = sorted(file_timestamps.items(), key=lambda item: item[1])[-1][0]

    return f'{base_dir}/{latest_file_path}'


def get_latest_model_instance(base_dir: str):
    latest_model_path = get_oldest_file_from_dir(base_dir)
    with open(latest_model_path, 'rb') as latest_model_file:
        latest_model = dill.load(latest_model_file)

    logging.info(f'Using model instance from path "{latest_model_path}"')
    return latest_model


def predict() -> None:
    model = get_latest_model_instance(f'{root}/data/models')

    test_files_path = f'{root}/data/test/'

    data = assemble_data_from_dir(test_files_path)

    predictions = model.predict(data)
    predictions = pd.DataFrame(
        {
            'car_id': data['id'].values,
            'pred': predictions
        }
    )

    predictions_path = f'{root}/data/predictions/preds_{datetime.now().strftime("%Y%m%d%H%M")}.csv'
    predictions.to_csv(predictions_path, index=False)

    logging.info(f'Saved predictions to path "{predictions_path}"')


if __name__ == '__main__':
    predict()
