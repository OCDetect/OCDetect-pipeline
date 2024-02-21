import getpass
import os
import pandas as pd
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from data_cleansing.helpers.definitions import Label, LabelMergeParameter, IgnoreReason, label_mapping
from merging import merge
from helpers import clean_merge_target_directory, convert_df

# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)

def relabel(subject, d_type_uncertain, d_type_certain, d_type_un_cert, relabeled_path, origin_path, target_path):
    clean_merge_target_directory(subject, target_path)
    relabeled = []
    for label_file in os.listdir(relabeled_path):
        if "_"+str(subject)+".csv" in label_file:
            relabeled.append(label_file)
    if len(relabeled) < 2 or len(relabeled) > 2:
        print ("Exactly 2 annotator files required")
        return

    df_first = convert_df(pd.read_csv(os.path.join(relabeled_path,relabeled[0]))) #convert jsons (labelstudio) into dfs
    df_second = convert_df(pd.read_csv(os.path.join(relabeled_path,relabeled[1])))

    annotations = merge(df_first, df_second, d_type_uncertain, d_type_certain, d_type_un_cert) # apply merging logic
    args = (subject, df_first, df_second, annotations, origin_path, target_path)

    # for preprocessed_file in os.listdir(origin_path): # for case that parallel processing not possible on atlas currently
    #     process_file(preprocessed_file, *args)

    with ProcessPoolExecutor(max_workers=3) as executor: #let run files in parallel
        futures = [executor.submit(process_file, preprocessed_file, *args) for preprocessed_file in os.listdir(origin_path)]

        for future in as_completed(futures):
            future.result()

def process_file(origin_file, subject, annotation_first, annotation_second, merged_annotation, origin_path, target_path):
        if origin_file.endswith('.csv') and origin_file.split('_')[1] == subject:# and origin_file == "OCDetect_03_recording_05_382535ec-9a0d-4359-b120-47f7605a22de.csv": #for  testing
            origin_df = pd.read_csv(os.path.join(origin_path, origin_file))
            origin_df['datetime'] = pd.to_datetime(origin_df['datetime'])

            # only files that belong to the annotation
            file_annotations = merged_annotation.loc[merged_annotation['file'] == origin_file].reset_index()
            indices_annotations = file_annotations.index

            label_dates = origin_df.loc[origin_df['user yes/no'] == 1.0, ['datetime', 'compulsive']]
            # get User labels from preprocessed origin_file

            # Determine for labeled interval which User label it belongs to
            # Interval doesn't include User label -> first User label after end of interval
            # Interval does include User label -> last User label included in interval
            for annotation_index in indices_annotations:
                label_crossings = label_dates[label_dates['datetime'].between(file_annotations.iloc[annotation_index, 3], file_annotations.iloc[annotation_index, 4], inclusive = 'neither')]
                if label_crossings.empty:
                    next_labels = label_dates.loc[(pd.Timedelta('5 minutes') >= (label_dates['datetime'] - file_annotations.iloc[annotation_index, 4])) &
                                                  ((label_dates['datetime'] - file_annotations.iloc[annotation_index, 4]) >= pd.Timedelta(0)),
                                                  ['datetime','compulsive']]
                    file_annotations.loc[annotation_index, 'compulsive'] = next_labels.iloc[0,1] # 4: compulsive 5: usr_label datetime
                    file_annotations.loc[annotation_index, 'usr_label'] = next_labels.iloc[0,0]
                else:
                    file_annotations.loc[annotation_index, 'compulsive'] = label_crossings.iloc[-1,1]
                    file_annotations.loc[annotation_index, 'usr_label'] = label_crossings.iloc[-1,0]

            # write annotations first, annotations second and merged annotation into preprocessed file
            relabeled_df = add_annotations(origin_file, origin_df, annotation_first, annotation_second, file_annotations)

            # set ignore column
            output_df = set_ignore(relabeled_df, merged_annotation)

            output_df.to_csv(os.path.join(target_path, origin_file), index = False) #write relabeled files to target_path
            print("file", origin_file, "done")

def add_annotations(file_name, origin, annotation_first, annotation_second, merged_annotation):
    origin[['annotator_1', 'annotator_2', 'merged_annotation', 'compulsive_relabeled']] = None
    annotators = [annotation_first, annotation_second]
    for i, annotator in enumerate(annotators):
        for index_annotator in annotator.index: # write single annotations to preprocessed_file
            if annotator.iloc[index_annotator, 0] == file_name:
                label = annotator.iloc[index_annotator]['label']
                origin.loc[(origin['datetime'] >= annotator.iloc[index_annotator]['start']) &
                           (origin['datetime'] <= annotator.iloc[index_annotator]['end']), 'annotator_'+str(i+1)] = label.value

    for index_merged in merged_annotation.index: # add merged annotation to corresponding row until User label
        if merged_annotation.iloc[index_merged]['file']==file_name:

            compulsive = int(merged_annotation.iloc[index_merged]['compulsive'])

            indices_merge_rows = origin.loc[(origin['datetime'] >= merged_annotation.iloc[index_merged]['start']) &
                       (origin['datetime'] <= merged_annotation.iloc[index_merged]['end']) &
                       (origin['datetime'] <= merged_annotation.iloc[index_merged]['usr_label'])].index

            origin.loc[indices_merge_rows, 'merged_annotation'] = 1
            origin.loc[indices_merge_rows, 'compulsive_relabeled'] = compulsive

    return origin

def set_ignore(relabeled, merged_annotation): # set ignore value (7) for 5 minutes before relabeled handwashing timeframe
    for index_merged in merged_annotation.index:
        relabeled.loc[((merged_annotation.iloc[index_merged]['start'] - relabeled['datetime']) > pd.Timedelta(minutes=0)) &
                      ((merged_annotation.iloc[index_merged]['start'] - relabeled['datetime']) <= pd.Timedelta(minutes=5)) &
                      (relabeled['merged_annotation'] != 1), 'ignore'] = IgnoreReason.BeforeHandWash
    return relabeled



### Insert rows for testing of compulsive column TO BE DELETED
# new_row ={'file': 'OCDetect_03_recording_05_382535ec-9a0d-4359-b120-47f7605a22de.csv', 'file_number':12, 'start': '2022-04-05 17:40:40.480000', 'end': '2022-04-05 17:41:59.220000'}
#             index = len(merged_annotation) - 1
#             merged_annotation = pd.concat([merged_annotation.iloc[:index], pd.DataFrame([new_row]), merged_annotation.iloc[index:]], ignore_index=True)
#             merged_annotation['end'] = pd.to_datetime(merged_annotation['end'])
#             merged_annotation['start'] = pd.to_datetime(merged_annotation['start'])
# new_label = {'datetime': '2022-04-05 17:45:01.220000', 'compulsive': 0}
# label_dates = pd.concat([label_dates, pd.DataFrame([new_label])], ignore_index=True)
# label_dates['datetime'] = pd.to_datetime(label_dates['datetime'])