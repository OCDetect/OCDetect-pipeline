import getpass
import os
import pandas as pd
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from data_cleansing.helpers.definitions import Label, LabelMergeParameter, IgnoreReason, label_mapping

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

run_subject = "18"  # specify in files e.g. OCDetect_03 run_subject = "03", OCDetect_30 run_subject = "30"

relabeled_path = "/dhc/groups/ocdetect/relabeled_subjects"  # path to exported files from label studio, always name then "a{number of annotator}_subject_{subject_number}.csv subject_number as fpr run_subject
# lea: 1, lorenz: 2, robin: 3, kristina: 4
origin_path = "/dhc/groups/ocdetect/preprocessed"  # path to files getting relabeled
target_path = "/dhc/groups/ocdetect/preprocessed_relabeled"  # path to relabeled files

# Merging settings
d_type_certain = LabelMergeParameter.Intersection  # LabelMergeParameter.Intersection or LabelMergeParameter.Union
d_type_uncertain = LabelMergeParameter.Intersection  # LabelMergeParameter.Intersection or LabelMergeParameter.Union
d_type_un_cert = LabelMergeParameter.IgnoreUncertain  # LabelMergeParameter.Intersection or LabelMergeParameter.IgnoreUncertain


def relabel(subject):
    clean_target_directory(subject, target_path)
    relabeled = []
    for label_file in os.listdir(relabeled_path):
        if "_" + str(subject) + ".csv" in label_file:
            relabeled.append(label_file)
    if len(relabeled) < 2 or len(relabeled) > 2:
        print("Exactly 2 annotator files required")
        return

    df_first = convert_df(
        pd.read_csv(os.path.join(relabeled_path, relabeled[0])))  # convert jsons (labelstudio) into dfs
    df_second = convert_df(pd.read_csv(os.path.join(relabeled_path, relabeled[1])))

    annotations = merge(df_first, df_second, d_type_uncertain, d_type_certain, d_type_un_cert)  # apply merging logic

    args = (subject, df_first, df_second, annotations)

    # for preprocessed_file in os.listdir(origin_path): # for case that parallel processing not possible on atlas currently
    #     process_file(preprocessed_file, *args)

    with ProcessPoolExecutor(max_workers=5) as executor:  # let run files in parallel
        futures = [executor.submit(process_file, preprocessed_file, *args) for preprocessed_file in
                   os.listdir(origin_path)]

        for future in as_completed(futures):
            future.result()


def process_file(origin_file, subject, annotation_first, annotation_second, merged_annotation):
    if origin_file.endswith('.csv') and origin_file.split('_')[
        1] == subject:  # and origin_file == "OCDetect_03_recording_05_382535ec-9a0d-4359-b120-47f7605a22de.csv": #for  testing
        origin_df = pd.read_csv(os.path.join(origin_path, origin_file))
        origin_df['datetime'] = pd.to_datetime(origin_df['datetime'])

        # only files that belong to the annotation
        merged_annotation = merged_annotation.loc[merged_annotation['file'] == origin_file].copy()

        # get User labels from preprocessed origin_file
        label_dates = origin_df.loc[origin_df['user yes/no'] == 1.0, ['datetime', 'compulsive']]

        # Determine for labeled interval which User label it belongs to
        # Interval doesn't include User label -> first User label after end of interval
        # Interval does include User label -> last User label included in interval
        for annotation_index, annotation in merged_annotation.iterrows():
            label_crossings = label_dates[
                label_dates['datetime'].between(annotation['start'], annotation['end'], inclusive='neither')]
            if label_crossings.empty:
                next_labels = label_dates.loc[
                    (pd.Timedelta('5 minutes') >= (label_dates['datetime'] - annotation['end'])) & (
                                (label_dates['datetime'] - annotation['end']) >= pd.Timedelta(0)), ['datetime',
                                                                                                    'compulsive']]
                merged_annotation.loc[annotation_index, 'compulsive'] = next_labels.iloc[0]['compulsive']
                merged_annotation.loc[annotation_index, 'usr_label'] = next_labels.iloc[0]['datetime']
            else:
                merged_annotation.loc[annotation_index, 'compulsive'] = label_crossings.iloc[-1]['compulsive']
                merged_annotation.loc[annotation_index, 'usr_label'] = label_crossings.iloc[-1]['datetime']

        ## ALTERNATIVE: Determine for labeled interval which User label it belongs to, closest User label to end of interval (before or after end)
        # for annotation_index, annotation in merged_annotation.iterrows():
        #     min_distance_index = (label_dates['datetime'] - annotation['end']).abs().idxmin()
        #     merged_annotation.loc[annotation_index, 'compulsive'] = label_dates.loc[min_distance_index, 'compulsive']
        #     merged_annotation.loc[annotation_index, 'usr_label'] = label_dates.loc[min_distance_index, 'datetime']

        # write annotations first, annotations second and merged annotation into preprocessed file
        relabeled_df = add_annotations(origin_file, origin_df, annotation_first, annotation_second, merged_annotation)

        # set ignore column
        output_df = set_ignore(relabeled_df, merged_annotation)

        output_df.to_csv(os.path.join(target_path, origin_file), index=False)  # write relabeled files to target_path


def merge(df_first, df_second, type_uncertain, type_certain, type_un_cert):  # logic how cases should be merged
    all_labels = pd.DataFrame(columns=['file', 'file_number', 'start', 'end'])
    # iterate through all labels of annotator1 and annotator2
    for index1, row1 in df_first.iterrows():
        for index2, row2 in df_second.iterrows():
            if row2['file'] == row1['file']:
                if int(row2['file_number']) > int(row1['file_number']):
                    break  # our files are in order on the cluster
                if row1['file_number'] == row2['file_number']:
                    label1, label2 = row1['label'], row2['label']
                    start1, start2 = row1['start'], row2['start']
                    end1, end2 = row1['end'], row2['end']
                    # check for intersection between the annotations, else skip
                    if end1 < start2 or end2 < start1:
                        continue
                    start, end = None, None
                    if label1 == label2:
                        # handle equal labels with specified merge types
                        if label1 == Label.Certain:
                            start = find_start(type_certain, start1, start2)
                            end = find_end(type_certain, end1, end2)
                        elif label1 == Label.BeginEndUncertain:
                            start = find_start(type_uncertain, start1, start2)
                            end = find_end(type_uncertain, end1, end2)
                        elif label1 == Label.BeginUncertain:
                            start = find_start(type_uncertain, start1, start2)
                            end = find_end(type_certain, end1, end2)
                        elif label1 == Label.EndUncertain:
                            start = find_start(type_certain, start1, start2)
                            end = find_end(type_uncertain, end1, end2)
                    else:
                        # handle certain-uncertain cases
                        if ((label1 == Label.Certain and label2 == Label.BeginEndUncertain) or
                                (label2 == Label.Certain and label1 == Label.BeginEndUncertain)):
                            start = find_un_cert_start(type_un_cert, start1, start2, label1)
                            end = find_un_cert_end(type_un_cert, end1, end2, label1)
                        # handle label begin_uncertain
                        elif label1 == Label.BeginUncertain or label2 == Label.BeginUncertain:
                            if label2 == Label.Certain or label1 == Label.Certain:
                                start = find_un_cert_start(type_un_cert, start1, start2, label1)
                                end = find_end(type_certain, end1, end2)
                            elif label2 == Label.BeginEndUncertain or label1 == Label.BeginEndUncertain:
                                start = find_start(type_uncertain, start1, start2)
                                end = find_un_cert_end(type_un_cert, end1, end2, label1)
                            elif label2 == Label.EndUncertain or label1 == Label.EndUncertain:
                                start = find_un_cert_start(type_un_cert, start1, start2, label1)
                                end = find_un_cert_end(type_un_cert, end1, end2, label1)
                        # handle cases with end_uncertain
                        elif label1 == Label.EndUncertain or label2 == Label.EndUncertain:
                            # begin_uncertain End uncertain already handled
                            if label2 == Label.Certain or label1 == Label.Certain:
                                start = find_start(type_certain, start1, start2)
                                end = find_un_cert_end(type_un_cert, end1, end2, label1)
                            elif label2 == Label.BeginEndUncertain or label1 == Label.BeginEndUncertain:
                                start = find_un_cert_start(type_un_cert, start1, start2, label1)
                                end = find_end(type_uncertain, end1, end2)
                    # append new label to df
                    if start and end:
                        new_row = {'file': row1['file'], 'file_number': row1['file_number'], 'start': start, 'end': end}
                        all_labels = pd.concat([all_labels, pd.DataFrame([new_row])], ignore_index=True)
    return all_labels


def find_start(merge_type, start1, start2):
    if merge_type == LabelMergeParameter.Intersection:
        return max(start1, start2)
    if merge_type == LabelMergeParameter.Union:
        return min(start1, start2)


def find_end(merge_type, end1, end2):
    if merge_type == LabelMergeParameter.Intersection:
        return min(end1, end2)
    if merge_type == LabelMergeParameter.Union:
        return max(end1, end2)


def find_un_cert_start(merge_type, start1, start2, label1):
    if merge_type == LabelMergeParameter.Intersection:
        return max(start1, start2)
    if merge_type == LabelMergeParameter.IgnoreUncertain:
        if label1 == Label.Certain or label1 == Label.EndUncertain:
            return start1
        else:
            return start2


def find_un_cert_end(merge_type, end1, end2, label1):
    if merge_type == LabelMergeParameter.Intersection:
        return min(end1, end2)
    if merge_type == LabelMergeParameter.IgnoreUncertain:
        if label1 == Label.Certain or label1 == Label.BeginUncertain:
            return end1
        else:
            return end2


def add_annotations(file_name, origin, annotation_first, annotation_second, merged_annotation):
    origin[['annotator_1', 'annotator_2', 'merged_annotation', 'compulsive_relabeled']] = None
    annotators = [annotation_first, annotation_second]
    for i, annotator in enumerate(annotators):
        for index, row in annotator.iterrows():  # write single annotations to preprocessed_file
            if row['file'] == file_name:
                label = row['label']
                origin.loc[
                    (origin['datetime'] >= row['start']) & (origin['datetime'] <= row['end']), 'annotator_' + str(
                        i + 1)] = label.value

    for index, row in merged_annotation.iterrows():  # add merged annotation to corresponding row until User label
        if row['file'] == file_name:
            compulsive = int(row['compulsive'])
            origin.loc[(origin['datetime'] >= row['start']) & (origin['datetime'] <= row['end']) & (
                        origin['datetime'] <= row['usr_label']), 'merged_annotation'] = 1
            origin.loc[(origin['datetime'] >= row['start']) & (origin['datetime'] <= row['end']) & (
                        origin['datetime'] <= row['usr_label']), 'compulsive_relabeled'] = compulsive
    return origin


def set_ignore(relabeled,
               merged_annotation):  # set ignore value (7) for 5 minutes before relabeled handwashing timeframe
    for annotation_index, annotation in merged_annotation.iterrows():
        relabeled.loc[((annotation['start'] - relabeled['datetime']) > pd.Timedelta(minutes=0)) &
                      ((annotation['start'] - relabeled['datetime']) <= pd.Timedelta(minutes=5)) &
                      (relabeled['merged_annotation'] != 1), 'ignore'] = IgnoreReason.BeforeHandWash
    return relabeled


def convert_df(df):
    # create from label studio exported csv df as e.g.
    # 'file': OCDetect_03_recording_04_0a48395d-614f-497c-ab71-0d579d74ce27.csv, 'file_number': 1, 'start': 2022-04-05 10:17:45.080, 'end': 2022-04-05 10:18:23.100, 'label': Certain
    scheme = {'file': [], 'file_number': [], 'start': [], 'end': [], 'label': []}
    df_new = pd.DataFrame(scheme)
    for index, row in df.iterrows():
        file_base = os.path.basename(row['datetime']).split('-', 1)[1]
        file_name = file_base.rsplit('_', 1)[0] + ".csv"
        file_number = (file_base.rsplit('_', 1)[1]).rsplit('.', 1)[0]
        if pd.isna(row['label']):  # if no label was set, just ignore and jump to next row
            continue
        row_label = json.loads(row['label'])[0]
        start = row_label["start"][:23]
        end = row_label["end"][:23]
        label = label_mapping[row_label["timeserieslabels"][0]]

        new_row = {'file': file_name, 'file_number': file_number, 'start': start, 'end': end, 'label': label}
        df_new = pd.concat([df_new, pd.DataFrame([new_row])], ignore_index=True)

    df_new['start'] = pd.to_datetime(df_new['start'])  # convert to datetime for fast comparison
    df_new['end'] = pd.to_datetime(df_new['end'])
    return df_new


# delete all files of the subject in target directory (preprocessed_relabeled)
def clean_target_directory(subject_id, target_directory):
    for file in os.listdir(target_directory):
        if "OCDetect_" + str(subject_id) in file:
            os.remove(os.path.join(target_directory, file))


### RUN
relabel(run_subject)  # TODO run for all manually relabeled subjects

### Insert rows for testing of compulsive column
# new_row ={'file': 'OCDetect_03_recording_05_382535ec-9a0d-4359-b120-47f7605a22de.csv', 'file_number':12, 'start': '2022-04-05 17:40:40.480000', 'end': '2022-04-05 17:41:59.220000'}
#             index = len(merged_annotation) - 1
#             merged_annotation = pd.concat([merged_annotation.iloc[:index], pd.DataFrame([new_row]), merged_annotation.iloc[index:]], ignore_index=True)
#             merged_annotation['end'] = pd.to_datetime(merged_annotation['end'])
#             merged_annotation['start'] = pd.to_datetime(merged_annotation['start'])
# new_label = {'datetime': '2022-04-05 17:45:01.220000', 'compulsive': 0}
# label_dates = pd.concat([label_dates, pd.DataFrame([new_label])], ignore_index=True)
# label_dates['datetime'] = pd.to_datetime(label_dates['datetime'])
