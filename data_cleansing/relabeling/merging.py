import getpass
import os
import pandas as pd
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from data_cleansing.helpers.definitions import Label, LabelMergeParameter, IgnoreReason, label_mapping

def merge(df_first, df_second, type_uncertain, type_certain, type_un_cert): # logic how cases should be merged
    all_labels = pd.DataFrame(columns=['file', 'file_number', 'start', 'end'])
    # iterate through all labels of annotator1 and annotator2
    joined_df = pd.merge(df_first, df_second, on=['file', 'file_number'], how='inner')
    for index_joined in joined_df.index:
        label1, label2 = joined_df.iloc[index_joined]['label_x'], joined_df.iloc[index_joined]['label_y']
        start1, start2 = joined_df.iloc[index_joined]['start_x'], joined_df.iloc[index_joined]['start_y']
        end1, end2 = joined_df.iloc[index_joined]['end_x'], joined_df.iloc[index_joined]['end_y']
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
            new_row = {'file': joined_df.iloc[index_joined]['file'], 'file_number': joined_df.iloc[index_joined]['file_number'],
                    'start': start, 'end': end}
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