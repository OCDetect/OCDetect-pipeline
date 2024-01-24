import getpass
import os
import pandas as pd
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from itertools import repeat

pd.set_option('display.max_rows', None)

labels = {"Certain": 1, "Begin uncertain": 2, "End uncertain": 3, "Begin AND End uncertain": 4}

relabel_method = "two_columns" # method whether annotator columns are merged or both adding both annotations "merged_column" or "two_columns"
run_subject = "03" #specify always as in files e.g. OCDetect_03 run_subject = "03", OCDetect_30 run_subject = "30"

relabeled_path = "/dhc/groups/ocdetect/relabeled_subjects" # path to exported files from label studio, always name then "a{number of annotator}_subject_{subject_number}.csv subject_number as fpr run_subject
# lea: 1, lorenz: 2, robin: 3, kristina: 4
origin_path = "/dhc/home/"+getpass.getuser()+"/datasets/OCDetect/preprocessed" # path to files getting relabeled
target_path = "/dhc/groups/ocdetect/preprocessed_relabeled_"+relabel_method # path to relabeled files, one for both methods

def process_file(file, subject, df_first, df_second=None):
    if file.endswith('.csv') and file.split('_')[1] == subject: # and file == "OCDetect_03_recording_05_382535ec-9a0d-4359-b120-47f7605a22de.csv":
        origin_df = pd.read_csv(os.path.join(origin_path, file))
        origin_df['datetime'] = pd.to_datetime(origin_df['datetime'])
        if relabel_method == "merged_column":
            relabel_df = merged_column(file, origin_df, df_first)
        if relabel_method == "two_columns":
            relabeled_df = two_columns(file, origin_df, df_first, df_second)

        relabeled_df.to_csv(os.path.join(target_path, file), index=False) #write relabeled files to target_path

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
    if relabel_method == "merged_column":
        merged_df = merge(df_first, df_second)
        args = (subject, merged_df)

    if relabel_method == "two_columns":
        args = (subject, df_first, df_second)

    with ProcessPoolExecutor(max_workers=63) as executor: #let run files in parallel
        futures = [executor.submit(process_file, file, *args) for file in os.listdir(origin_path)]

        for future in as_completed(futures):
            future.result()


def two_columns(file_name, origin, annotator_frst, annotator_scnd): #writing new labels to origin dateframe as two columns for both annotators
    origin['annotator_1'] = 0
    origin['annotator_2'] = 0
    annotators = [annotator_frst, annotator_scnd]
    for i, annotator in enumerate(annotators):
        for index, row in annotator.iterrows():
            if row['file']==file_name:
                label = labels[row['label']]
                print(label, row["label"], row['start'])
                origin.loc[(origin['datetime'] >= row['start']) & (origin['datetime'] <= row['end']), 'annotator_'+str(i+1)] = label
    return origin

def merge(df_first, df_second): # logic how cases certain-certain, certain-uncertain, uncertain-uncertain should be merged
    return df_first #TODO how to merge

def merged_column(file_name, origin, annotation): #writing new labels to origin dateframe as one column
    origin['annotation'] = 0
    for index, row in annotation.iterrows():
        if row['file']==file_name:
            origin.loc[(origin['datetime'] >= row['start']) & (origin['datetime'] <= row['end']), 'annotation'] = 1.0
    return origin


def convert_df(df):
    # create from label studio exported csv df as e.g.
    # 'file': OCDetect_03_recording_04_0a48395d-614f-497c-ab71-0d579d74ce27.csv, 'file_number': 1, 'start': 2022-04-05 10:17:45.080, 'end': 2022-04-05 10:18:23.100, 'label': Certain
    scheme = {'file': [], 'file_number': [], 'start': [], 'end': [], 'label': []}
    df_new = pd.DataFrame(scheme)
    for index, row in df.iterrows():
        file_name = os.path.basename(row['datetime']).split('-', 1)[1]
        file = file_name.rsplit('_', 1)[0] + ".csv"
        file_number = (file_name.rsplit('_', 1)[1]).rsplit('.', 1)[0]

        row_label = json.loads(row['label'])[0]
        start = row_label["start"][:23]
        end = row_label["end"][:23]
        label = row_label["timeserieslabels"][0]

        new_row = {'file': file, 'file_number': file_number, 'start': start, 'end': end, 'label': label}
        df_new = pd.concat([df_new, pd.DataFrame([new_row])], ignore_index=True)

    df_new['start'] = pd.to_datetime(df_new['start']) #convert to datetime for fast comparison
    df_new['end'] = pd.to_datetime(df_new['end'])

    return df_new


# run
for file in os.listdir(target_path): #TODO delete only files of the subject
    os.remove(os.path.join(target_path,file)) #delete all

relabel(run_subject)

