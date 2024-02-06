import getpass
import os
import pandas as pd
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

labels = {"Certain": 1, "Begin uncertain": 2, "End uncertain": 3, "Begin AND End uncertain": 4}

#relabel_method = "merged_column" # method whether annotator columns are merged or both adding both annotations "merged_column" or "two_columns"
run_subject = "03" # specify always as in files e.g. OCDetect_03 run_subject = "03", OCDetect_30 run_subject = "30"

relabeled_path = "/dhc/groups/ocdetect/relabeled_subjects" # path to exported files from label studio, always name then "a{number of annotator}_subject_{subject_number}.csv subject_number as fpr run_subject
# lea: 1, lorenz: 2, robin: 3, kristina: 4
origin_path = "/dhc/groups/ocdetect/preprocessed" # path to files getting relabeled
target_path = "/dhc/groups/ocdetect/preprocessed_relabeled" # path to relabeled files, one for both methods

# Merging settings
type_certain = "union" # intersection or union
type_uncertain = "union" # intersection or union
type_un_cert = "ignore_uncertain" # intersection or ignore_uncertain

def process_file(file, subject, df_first, df_second, annotation_df):
        if file.endswith('.csv') and file.split('_')[1] == subject:# and file == "OCDetect_03_recording_05_382535ec-9a0d-4359-b120-47f7605a22de.csv":
            origin_df = pd.read_csv(os.path.join(origin_path, file))
            origin_df['datetime'] = pd.to_datetime(origin_df['datetime'])

            annotation_df =  annotation_df.loc[annotation_df['file'] == file].copy()

            label_dates = origin_df.loc[origin_df['user yes/no'] == 1.0, ['datetime', 'compulsive']]

            # for annotation_index, annotation in annotations_df.iterrows():
            #     next_labels = label_dates.loc[(pd.Timedelta('5 minutes') >= (label_dates['datetime'] - annotation['end'])) & ((label_dates['datetime'] - annotation['end']) >= pd.Timedelta(0)), ['datetime','compulsive']]
            #     if len(next_labels) == 0:
            #         raise ValueError("No User Label found for label", annotation['file'], annotation['file_number'], annotation['start'], annotation['end'])
            #     annotation_df.loc[annotation_index, 'compulsive'] = next_labels.iloc[0, 'compulsive']
            #     annotation_df.loc[annotation_index, 'usr_label'] = next_labels.iloc[0, 'datetime']

            for annotation_index, annotation in annotation_df.iterrows():
                min_distance_index = (label_dates['datetime'] - annotation['end']).abs().idxmin()
                annotation_df.loc[annotation_index, 'compulsive'] = label_dates.loc[min_distance_index, 'compulsive']
                annotation_df.loc[annotation_index, 'usr_label'] = label_dates.loc[min_distance_index, 'datetime']
            origin_df['compulsive'] = 0
            print(annotation_df)
            relabeled_df = add_annotations(file, origin_df, df_first, df_second, annotation_df)

            relabeled_df.to_csv(os.path.join(target_path, file), index=False) #write relabeled files to target_path

def relabel(subject):
    relabeled = []
    for label_file in os.listdir(relabeled_path):
        if "_"+str(subject)+".csv" in label_file:
            relabeled.append(label_file)
    if len(relabeled) < 2 or len(relabeled) > 2:
        print ("Exactly 2 annotator files required")
        return

    df_first = convert_df(pd.read_csv(os.path.join(relabeled_path,relabeled[0])))
    df_second = convert_df(pd.read_csv(os.path.join(relabeled_path,relabeled[1])))

    annotations = merge(df_first, df_second)

    args = (subject, df_first, df_second, annotations)

    # for ls_file in os.listdir(origin_path): # for case that parallel processing not possible on atlas currently
    #     process_file(ls_file, *args)

    with ProcessPoolExecutor(max_workers=5) as executor: #let run files in parallel
        futures = [executor.submit(process_file, file, *args) for file in os.listdir(origin_path)]

        for future in as_completed(futures):
            future.result()

def add_annotations(file_name, origin, annotator_frst, annotator_scnd, merged_annotation):
    origin['annotator_1'] = 0
    origin['annotator_2'] = 0
    origin['merged_annotation'] = 0
    annotators = [annotator_frst, annotator_scnd]
    for i, annotator in enumerate(annotators):
        for index, row in annotator.iterrows():
            if row['file'] == file_name:
                label = labels[row['label']]
                origin.loc[(origin['datetime'] >= row['start']) & (origin['datetime'] <= row['end']), 'annotator_'+str(i+1)] = label
    for index, row in merged_annotation.iterrows():
        if row['file']==file_name:
            compulsive = row['compulsive']
            origin.loc[(origin['datetime'] >= row['start']) & (origin['datetime'] <= row['end']) & (origin['datetime'] <= row['usr_label']), 'merged_annotation'] = 1.0
            origin.loc[(origin['datetime'] >= row['start']) & (origin['datetime'] <= row['end']) & (origin['datetime'] <= row['usr_label']), 'compulsive'] = compulsive
    return origin


def merge(df_first, df_second): # logic how cases should be merged
    all_labels = pd.DataFrame(columns=['file', 'file_number', 'start', 'end'])
    for index1, row1 in df_first.iterrows():
        for index2, row2 in df_second.iterrows():
            if row2['file'] == row1['file']:
                if row2['file_number'] > row1['file_number']:
                   continue # our files are in order
                if row1['file_number'] == row2['file_number']:
                    label1, label2 = row1['label'], row2['label']
                    start1, start2 = row1['start'], row2['start']
                    end1, end2 = row1['end'], row2['end']
                    if end1 < start2 or end2 < start1:
                        continue
                    start, end = None, None
                    if label1 == label2:
                    # equal labels
                        if label1 == 'Certain':
                            start = find_start(type_certain, start1, start2)
                            end = find_end(type_certain, end1, end2)
                        elif label1 == 'Begin AND End uncertain':
                            start = find_start(type_uncertain, start1, start2)
                            end = find_end(type_uncertain, end1, end2)
                        elif label1 == 'Begin uncertain':
                            start = find_start(type_uncertain, start1, start2)
                            end = find_end(type_certain, end1, end2)
                        elif label1 == 'End uncertain':
                            start = find_start(type_certain, start1, start2)
                            end = find_end(type_uncertain, end1, end2)
                    else:
                        if (label1 == 'Certain' and label2 == 'Begin AND End uncertain') or (label2 == 'Certain' and label1 == 'Begin AND End uncertain'):
                                start = find_un_cert_start(type_un_cert, start1, start2, label1)
                                end = find_un_cert_end(type_un_cert, end1, end2, label1)
                    #begin_uncertain
                        elif label1 == 'Begin uncertain' or label2 == 'Begin uncertain':
                            # handle cases with 'begin_uncertain'
                            if label2 == 'Certain' or label1 == 'Certain':
                                start = find_un_cert_start(type_un_cert, start1, start2, label1)
                                end = find_end(type_certain, end1, end2)
                            elif label2 == 'Begin AND End uncertain' or label1 == 'Begin AND End uncertain':
                                start = find_start(type_uncertain, start1, start2)
                                end = find_un_cert_end(type_un_cert, end1, end2, label1)
                            elif label2 == 'End uncertain' or label1 == 'End uncertain':
                                start = find_un_cert_start(type_un_cert, start1, start2, label1)
                                end = find_un_cert_end(type_un_cert, end1, end2, label1)
                        elif label1 == 'End uncertain' or label2 == 'End uncertain':
                            # handle cases with 'End uncertain'
                            #begin_uncertain End uncertain already handled
                            if label2 == 'Certain' or label1 == 'Certain':
                                start = find_start(type_certain, start1, start2)
                                end = find_un_cert_end(type_un_cert, end1, end2, label1)
                            elif label2 == 'Begin AND End uncertain' or label1 == 'Begin AND End uncertain':
                                start = find_un_cert_start(type_un_cert, start1, start2, label1)
                                end = find_end(type_uncertain, end1, end2)
                # append new label to df
                    if start and end:
                        new_row = {'file': row1['file'], 'file_number': row1['file_number'], 'start': start, 'end': end}
                        all_labels = pd.concat([all_labels, pd.DataFrame([new_row])], ignore_index=True)
    return all_labels


def find_start(merge_type, start1, start2):
    if merge_type == "intersection":
        return max(start1, start2)
    if merge_type == "union":
        return min(start1, start2)


def find_end(merge_type, end1, end2):
    if merge_type == "intersection":
        return min(end1, end2)
    if merge_type == "union":
        return max(end1, end2)


def find_un_cert_start(merge_type, start1, start2, label1):
    if merge_type == "intersection":
        return max(start1, start2)
    if merge_type == "ignore_uncertain":
        if label1 == "Certain" or label1 == "End uncertain": #CHANGED
            return start1
        else:
            return start2


def find_un_cert_end(merge_type, end1, end2, label1):
    if merge_type == "intersection":
        return min(end1, end2)
    if merge_type == "ignore_uncertain":
        if label1 == "Certain" or label1 == "Begin uncertain": #CHANGED
            return end1
        else:
            return end2

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
for file in os.listdir(target_path):
    if "OCDetect_"+str(run_subject) in file:
        os.remove(os.path.join(target_path,file)) #delete all

relabel(run_subject)


### Compulsive Testing

# new_row ={'file': 'OCDetect_03_recording_05_382535ec-9a0d-4359-b120-47f7605a22de.csv', 'file_number':12, 'start': '2022-04-05 17:41:40.480000', 'end': '2022-04-05 17:42:59.220000'}
# index = len(annotations_filtered) - 1
# annotations_filtered = pd.concat([annotations_filtered.iloc[:index], pd.DataFrame([new_row]), annotations_filtered.iloc[index:]], ignore_index=True)
# annotations_filtered['end'] = pd.to_datetime(annotations_filtered['end'])
# annotations_filtered['start'] = pd.to_datetime(annotations_filtered['start'])
# new_label = {'datetime': '2022-04-05 17:45:01.220000', 'compulsive': 0}
# label_dates = pd.concat([label_dates, pd.DataFrame([new_label])], ignore_index=True)
# label_dates['datetime'] = pd.to_datetime(label_dates['datetime'])

# Alternative Logic

# for user_index, user in label_dates.iterrows():
#     if pd.Timedelta('5 minutes') >= (user['datetime'] - annotation['end']) >= pd.Timedelta(0):
#         annotations_filtered.at[annotation_index, 'compulsive']=user['compulsive']
#         #print(annotation['end'], user['datetime'], user['compulsive'])
#         break


# Testing for merge logic
# label ='Certain'
# label_2 ='Certain'
# test_first = [['OCDetect_03_recording_06', 20, '2022-04-06 20:35:00.800000', '2022-04-06 20:39:00.740000', 'Begin AND End uncertain'],
#               ['OCDetect_03_recording_06', 19, '2022-04-06 20:01:00.800000', '2022-04-06 20:05:00.740000', label],
#               ['OCDetect_03_recording_06', 18, '2022-04-06 20:40:58.800000', '2022-04-06 20:45:00.740000', label],
#               ['OCDetect_03_recording_07', 1, '2022-04-06 20:35:00.800000', '2022-04-06 20:39:00.740000', label],
#               ['OCDetect_03_recording_08', 1, '2022-04-06 20:35:00.800000', '2022-04-06 20:39:00.740000', label],
#               ['OCDetect_03_recording_06', 17, '2022-04-06 21:30:00.800000', '2022-04-06 21:40:00.740000', label],
#               ['OCDetect_03_recording_06', 15, '2022-04-07 20:35:00.800000', '2022-04-07 20:39:00.740000', 'Certain']]
#
# test_second = [['OCDetect_03_recording_06', 20, '2022-04-06 20:36:00.800000', '2022-04-06 20:37:00.740000', 'Certain'],
#                ['OCDetect_03_recording_06', 19, '2022-04-06 20:00:00.800000', '2022-04-06 20:04:00.740000', label_2],
#                ['OCDetect_03_recording_06', 18, '2022-04-06 20:46:00.800000', '2022-04-06 20:48:00.740000', label_2],
#                ['OCDetect_03_recording_07', 2, '2022-04-06 20:35:00.800000', '2022-04-06 20:39:00.740000', label_2],
#                ['OCDetect_03_recording_09', 1, '2022-04-06 20:35:00.800000', '2022-04-06 20:39:00.740000', label_2],
#                ['OCDetect_03_recording_06', 17, '2022-04-06 21:29:00.800000', '2022-04-06 21:32:00.740000', label_2],
#                ['OCDetect_03_recording_06', 17, '2022-04-06 21:35:00.800000', '2022-04-06 21:41:00.740000', label_2],
#                ['OCDetect_03_recording_06', 15, '2022-04-07 20:36:00.800000', '2022-04-07 20:37:00.740000', 'Begin AND End uncertain']]
#
# # erste Einträge: test_first beinhaltet test_second
# # zweite Einträge: test_second beginnt, test_first beginnt, test_second endet, test_first endet
# # dritte einträge: überlappen sich nicht, test_second vor test_first
# # vierte Einträge: überlappen sich sind aber nicht auf dem gleichen File
# # fünfte Einträge: Überlappen sich sind aber nicht auf dem gleichen Subject
# # sechste und siebte Einträge: test_first ist langer Bereich wird von test_second in zwei verschiedenen Intervallen überlappt
# # letzte Spalte überlappen sich
#
# df_1 = pd.DataFrame(test_first, columns= ['file', 'file_number', 'start', 'end', 'label'])
# df_2 = pd.DataFrame(test_second, columns= ['file', 'file_number', 'start', 'end', 'label'])
# print("DF 1")
# print(df_1)
# print("DF 2")
# print(df_2)
#
# print("Merged DF")
# print(merge(df_1, df_2))