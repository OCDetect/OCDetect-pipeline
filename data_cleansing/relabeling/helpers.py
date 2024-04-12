import os
import pandas as pd
import json
from data_cleansing.helpers.definitions import label_mapping
import shutil

def convert_df(studio_df):
    # create from label studio exported csv df as e.g.
    # 'file': OCDetect_03_recording_04_0a48395d-614f-497c-ab71-0d579d74ce27.csv, 'file_number': 1, 'start': 2022-04-05 10:17:45.080, 'end': 2022-04-05 10:18:23.100, 'label': Certain
    scheme = {'file': [], 'file_number': [], 'start': [], 'end': [], 'label': []}
    relabel_df = pd.DataFrame(scheme)
    for index in studio_df.index:
        if pd.isna(studio_df.iloc[index]['label']):
            continue

        file_base = os.path.basename(studio_df.iloc[index]['datetime']).split('-', 1)[1]
        file_name = file_base.rsplit('_', 1)[0] + ".csv"
        file_number = (file_base.rsplit('_', 1)[1]).rsplit('.', 1)[0]

        row_label = json.loads(studio_df.iloc[index]['label'])[0]
        start = row_label["start"][:23]
        end = row_label["end"][:23]
        label = label_mapping[row_label["timeserieslabels"][0]]

        new_row = {'file': file_name, 'file_number': file_number, 'start': start, 'end': end, 'label': label}
        relabel_df = pd.concat([relabel_df, pd.DataFrame([new_row])], ignore_index=True)

    relabel_df['start'] = pd.to_datetime(relabel_df['start']) #convert to datetime for fast comparison
    relabel_df['end'] = pd.to_datetime(relabel_df['end'])
    return relabel_df

# delete all files of the subject in target directory (preprocessed_relabeled)
def clean_merge_target_directory(subject_id, target_directory):
    for file in os.listdir(target_directory):
        if "OCDetect_"+str(subject_id) in file:
            os.remove(os.path.join(target_directory, file))


def clean_split_target_directory(target_directory):
    for filename in os.listdir(target_directory):
        file_path = os.path.join(target_directory, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
