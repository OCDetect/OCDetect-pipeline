import getpass
import os
import pandas as pd
import json
from datetime import datetime


relabeled_path = "/dhc/groups/ocdetect/relabeled_subjects"
origin_path = "/dhc/home/"+getpass.getuser()+"/datasets/OCDetect/preprocessed"
target_path = "/dhc/groups/ocdetect/preprocessed_relabeled"
def relabel(subject):
    relabeled = []
    for file in os.listdir(relabeled_path):
        if "_"+str(subject)+".csv" in file:
            relabeled.append(file)
    if len(relabeled) < 2 or len(relabeled) > 2:
        print ("Exactly 2 annotator files required")
        return
    df_first = convert_df(pd.read_csv(os.path.join(relabeled_path,relabeled[0])))
    df_second = convert_df(pd.read_csv(os.path.join(relabeled_path,relabeled[1])))


    for file in os.listdir(origin_path):
        if file.endswith('.csv') and file.split('_')[1]==subject and file=="OCDetect_03_recording_05_382535ec-9a0d-4359-b120-47f7605a22de.csv":
            origin_df = pd.read_csv(os.path.join(origin_path,file))
            origin_df.loc[:, 'relabeled'] = 0
            relabeled_df = two_columns(file, origin_df, df_first, df_second)
            relabeled_df.to_csv(os.path.join(target_path,file), index=False)


def two_columns(file_name, origin, annotator_frst, annotator_scnd):
    origin['annotator_1'] = 0
    origin['annotator_2'] = 0
    for index, row in annotator_frst.iterrows():
        if row['file']==file_name:
            origin.loc[(origin['datetime'] >= row['start']) & (origin['datetime'] <= row['end']), 'annotator_1'] = 1.0
    for index, row in annotator_scnd.iterrows():
        if row['file']==file_name:
            origin.loc[(origin['datetime'] >= row['start']) & (origin['datetime'] <= row['end']), 'annotator_2'] = 1.0
    return origin


def convert_df(df):

    scheme = {'file': [], 'file_number': [], 'start': [], 'end': [], 'label': []}
    df_new = pd.DataFrame(scheme)
    for index, row in df.iterrows():
        file_name = os.path.basename(row['datetime']).split('-', 1)[1]
        file = file_name.rsplit('_', 1)[0] + ".csv"
        file_number = (file_name.rsplit('_', 1)[1]).rsplit('.', 1)[0]

        row_label = json.loads(row['label'])[0]
        start = row_label["start"][:23]
        end = row_label["end"][:23]
        label = row_label["timeserieslabels"]

        new_row = {'file': file, 'file_number': file_number, 'start': start, 'end': end, 'label': label}
        df_new = pd.concat([df_new, pd.DataFrame([new_row])], ignore_index=True)

    return df_new

for file in os.listdir(target_path):
    os.remove(os.path.join(target_path,file))
relabel("03")

