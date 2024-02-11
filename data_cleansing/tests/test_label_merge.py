from data_cleansing.helpers.definitions import Label, LabelMergeParameter
from data_cleansing.relabeling.label_merge import merge
import pandas as pd
import pytest

def test_merge():
    label = Label.Certain
    label_2 = Label.Certain
    # check to not handle cases: no overlap, same subject but different files, overlap but different subjects
    test_annotator1 = [
        ['OCDetect_03_recording_07', '1', '2022-04-06 20:35:00.800000', '2022-04-06 20:39:00.740000', label],
        ['OCDetect_03_recording_08', '1', '2022-04-06 20:35:00.800000', '2022-04-06 20:39:00.740000', label],
        ['OCDetect_03_recording_06', '18', '2022-04-06 20:40:58.800000', '2022-04-06 20:45:00.740000', label]]

    df_annotator1 = pd.DataFrame(test_annotator1, columns=['file', 'file_number', 'start', 'end', 'label'])
    test_annotator2 = [
        ['OCDetect_03_recording_09', '1', '2022-04-06 20:35:00.800000', '2022-04-06 20:39:00.740000', label_2],
        ['OCDetect_03_recording_07', '2', '2022-04-06 20:35:00.800000', '2022-04-06 20:39:00.740000', label_2],
        ['OCDetect_03_recording_06', '18', '2022-04-06 20:46:00.800000', '2022-04-06 20:48:00.740000', label_2]]

    df_annotator2 = pd.DataFrame(test_annotator2, columns=['file', 'file_number', 'start', 'end', 'label'])

    type_certain = LabelMergeParameter.Intersection  # LabelMergeParameter.Intersection or LabelMergeParameter.Union
    type_uncertain = LabelMergeParameter.Intersection  # LabelMergeParameter.Intersection or LabelMergeParameter.Union
    type_un_cert = LabelMergeParameter.IgnoreUncertain  # LabelMergeParameter.Intersection or LabelMergeParameter.IgnoreUncertain
    expected_output = pd.DataFrame({
            'file': [],
            'file_number': [],
            'start': [],
            'end': []})
    output = merge(df_annotator1, df_annotator2, type_uncertain, type_certain, type_un_cert)
    print(output == expected_output)
    # pd.testing.assert_frame_equal(output, expected_output)


def test_merge_certain_intersection():
    # all merge methods intersection
    test_annotator1 = [
        ['OCDetect_03_recording_06', 15, '2022-04-07 20:35:00.800000', '2022-04-07 20:39:00.740000', Label.BeginUncertain],
        ['OCDetect_03_recording_06', 17, '2022-04-06 21:30:00.800000', '2022-04-06 21:40:00.740000', Label.EndUncertain],
        ['OCDetect_03_recording_06', 19, '2022-04-06 20:01:00.800000', '2022-04-06 20:05:00.740000', Label.BeginEndUncertain],
        ['OCDetect_03_recording_06', 20, '2022-04-06 20:35:00.800000', '2022-04-06 20:39:00.740000', Label.Certain]]
    df_annotator1 = pd.DataFrame(test_annotator1, columns=['file', 'file_number', 'start', 'end', 'label'])

    test_annotator2 = [
        ['OCDetect_03_recording_06', 15, '2022-04-07 20:36:00.800000', '2022-04-07 20:37:00.740000', Label.EndUncertain],
        ['OCDetect_03_recording_06', 17, '2022-04-06 21:29:00.800000', '2022-04-06 21:32:00.740000', Label.BeginUncertain],
        ['OCDetect_03_recording_06', 17, '2022-04-06 21:35:00.800000', '2022-04-06 21:41:00.740000', Label.Certain],
        ['OCDetect_03_recording_06', 19, '2022-04-06 20:00:00.800000', '2022-04-06 20:04:00.740000', Label.BeginEndUncertain],
        ['OCDetect_03_recording_06', 20, '2022-04-06 20:36:00.800000', '2022-04-06 20:37:00.740000', Label.Certain]]
    df_annotator2 = pd.DataFrame(test_annotator2, columns=['file', 'file_number', 'start', 'end', 'label'])

    type_certain = LabelMergeParameter.Intersection  # LabelMergeParameter.Intersection or LabelMergeParameter.Union
    type_uncertain = LabelMergeParameter.Intersection  # LabelMergeParameter.Intersection or LabelMergeParameter.Union
    type_un_cert = LabelMergeParameter.Intersection  # LabelMergeParameter.Intersection or LabelMergeParameter.IgnoreUncertain
    expected = [
        ['OCDetect_03_recording_06', 15, '2022-04-07 20:36:00.800000', '2022-04-07 20:37:00.740000'],
        ['OCDetect_03_recording_06', 17, '2022-04-06 21:30:00.800000', '2022-04-06 21:32:00.740000'],
        ['OCDetect_03_recording_06', 17, '2022-04-06 21:35:00.800000', '2022-04-06 21:40:00.740000'],
        ['OCDetect_03_recording_06', 19, '2022-04-06 20:01:00.800000', '2022-04-06 20:04:00.740000'],
        ['OCDetect_03_recording_06', 20, '2022-04-06 20:36:00.800000', '2022-04-06 20:37:00.740000']]

    expected_output = pd.DataFrame(expected, columns=['file', 'file_number', 'start', 'end'])
    output = merge(df_annotator1, df_annotator2, type_uncertain, type_certain, type_un_cert)
    output.reset_index(drop=True, inplace=True)
    expected_output.reset_index(drop=True, inplace=True)
    print(output == expected_output)
    # pd.testing.assert_frame_equal(output, expected_output)



def test_merge_ignore_uncertain_union():
    # merge strategy: union, union, uncertain-certain: intersection
    test_annotator1 = [
        ['OCDetect_03_recording_06', 15, '2022-04-07 20:35:00.800000', '2022-04-07 20:39:00.740000', Label.BeginUncertain],
        ['OCDetect_03_recording_06', 17, '2022-04-06 21:30:00.800000', '2022-04-06 21:40:00.740000', Label.EndUncertain],
        ['OCDetect_03_recording_06', 19, '2022-04-06 20:01:00.800000', '2022-04-06 20:05:00.740000', Label.BeginEndUncertain],
        ['OCDetect_03_recording_06', 20, '2022-04-06 20:35:00.800000', '2022-04-06 20:39:00.740000', Label.Certain]]
    df_annotator1 = pd.DataFrame(test_annotator1, columns=['file', 'file_number', 'start', 'end', 'label'])

    test_annotator2 = [
        ['OCDetect_03_recording_06', 15, '2022-04-07 20:36:00.800000', '2022-04-07 20:37:00.740000', Label.EndUncertain],
        ['OCDetect_03_recording_06', 17, '2022-04-06 21:29:00.800000', '2022-04-06 21:32:00.740000', Label.BeginUncertain],
        ['OCDetect_03_recording_06', 17, '2022-04-06 21:35:00.800000', '2022-04-06 21:41:00.740000', Label.Certain],
        ['OCDetect_03_recording_06', 19, '2022-04-06 20:00:00.800000', '2022-04-06 20:04:00.740000', Label.BeginEndUncertain],
        ['OCDetect_03_recording_06', 20, '2022-04-06 20:36:00.800000', '2022-04-06 20:37:00.740000', Label.Certain]]
    df_annotator2 = pd.DataFrame(test_annotator2, columns=['file', 'file_number', 'start', 'end', 'label'])

    type_certain = LabelMergeParameter.Union  # LabelMergeParameter.Intersection or LabelMergeParameter.Union
    type_uncertain = LabelMergeParameter.Union  # LabelMergeParameter.Intersection or LabelMergeParameter.Union
    type_un_cert = LabelMergeParameter.Intersection  # LabelMergeParameter.Intersection or LabelMergeParameter.IgnoreUncertain
    expected = [
        ['OCDetect_03_recording_06', 15, '2022-04-07 20:36:00.800000', '2022-04-07 20:37:00.740000'],
        ['OCDetect_03_recording_06', 17, '2022-04-06 21:30:00.800000', '2022-04-06 21:32:00.740000'],
        ['OCDetect_03_recording_06', 17, '2022-04-06 21:30:00.800000', '2022-04-06 21:40:00.740000'],
        ['OCDetect_03_recording_06', 19, '2022-04-06 20:00:00.800000', '2022-04-06 20:05:00.740000'],
        ['OCDetect_03_recording_06', 20, '2022-04-06 20:35:00.800000', '2022-04-06 20:39:00.740000']]

    expected_output = pd.DataFrame(expected, columns=['file', 'file_number', 'start', 'end'])
    output = merge(df_annotator1, df_annotator2, type_uncertain, type_certain, type_un_cert)
    # print(output)
    # print(expected_output)
    print(output == expected_output)
    #pd.testing.assert_frame_equal(output3, expected_output)

def test_union_ignore_uncertain():
    # merge strategy union and uncertain-certain: ignore_uncertain
    test_annotator1 = [
        ['OCDetect_03_recording_06', 15, '2022-04-07 20:35:00.800000', '2022-04-07 20:39:00.740000', Label.BeginUncertain],
        ['OCDetect_03_recording_06', 17, '2022-04-06 21:30:00.800000', '2022-04-06 21:40:00.740000', Label.EndUncertain],
        ['OCDetect_03_recording_06', 19, '2022-04-06 20:01:00.800000', '2022-04-06 20:05:00.740000', Label.BeginEndUncertain],
        ['OCDetect_03_recording_06', 20, '2022-04-06 20:35:00.800000', '2022-04-06 20:39:00.740000', Label.Certain]]
    df_annotator1 = pd.DataFrame(test_annotator1, columns=['file', 'file_number', 'start', 'end', 'label'])

    test_annotator2 = [
        ['OCDetect_03_recording_06', 15, '2022-04-07 20:36:00.800000', '2022-04-07 20:37:00.740000', Label.EndUncertain],
        ['OCDetect_03_recording_06', 17, '2022-04-06 21:29:00.800000', '2022-04-06 21:32:00.740000', Label.BeginUncertain],
        ['OCDetect_03_recording_06', 17, '2022-04-06 21:35:00.800000', '2022-04-06 21:41:00.740000', Label.Certain],
        ['OCDetect_03_recording_06', 19, '2022-04-06 20:00:00.800000', '2022-04-06 20:04:00.740000', Label.BeginEndUncertain],
        ['OCDetect_03_recording_06', 20, '2022-04-06 20:36:00.800000', '2022-04-06 20:37:00.740000', Label.Certain]]
    df_annotator2 = pd.DataFrame(test_annotator2, columns=['file', 'file_number', 'start', 'end', 'label'])

    type_certain = LabelMergeParameter.Union  # LabelMergeParameter.Intersection or LabelMergeParameter.Union
    type_uncertain = LabelMergeParameter.Union  # LabelMergeParameter.Intersection or LabelMergeParameter.Union
    type_un_cert = LabelMergeParameter.IgnoreUncertain  # LabelMergeParameter.Intersection or LabelMergeParameter.IgnoreUncertain
    expected = [
        ['OCDetect_03_recording_06', 15, '2022-04-07 20:36:00.800000', '2022-04-07 20:39:00.740000'],
        ['OCDetect_03_recording_06', 17, '2022-04-06 21:30:00.800000', '2022-04-06 21:32:00.740000'],
        ['OCDetect_03_recording_06', 17, '2022-04-06 21:30:00.800000', '2022-04-06 21:41:00.740000'],
        ['OCDetect_03_recording_06', 19, '2022-04-06 20:00:00.800000', '2022-04-06 20:05:00.740000'],
        ['OCDetect_03_recording_06', 20, '2022-04-06 20:35:00.800000', '2022-04-06 20:39:00.740000']]

    expected_output = pd.DataFrame(expected, columns=['file', 'file_number', 'start', 'end'])
    output = merge(df_annotator1, df_annotator2, type_uncertain, type_certain, type_un_cert)
    print(output == expected_output)


def test_merge_ignore_uncertain_intersection():
    # intersection, union, ignore_uncertain
    test_annotator1 = [
        ['OCDetect_03_recording_06', 15, '2022-04-07 20:35:00.800000', '2022-04-07 20:39:00.740000', Label.BeginUncertain],
        ['OCDetect_03_recording_06', 17, '2022-04-06 21:30:00.800000', '2022-04-06 21:40:00.740000', Label.EndUncertain],
        ['OCDetect_03_recording_06', 19, '2022-04-06 20:01:00.800000', '2022-04-06 20:05:00.740000', Label.BeginEndUncertain],
        ['OCDetect_03_recording_06', 20, '2022-04-06 20:35:00.800000', '2022-04-06 20:39:00.740000', Label.Certain]]
    df_annotator1 = pd.DataFrame(test_annotator1, columns=['file', 'file_number', 'start', 'end', 'label'])

    test_annotator2 = [
        ['OCDetect_03_recording_06', 15, '2022-04-07 20:36:00.800000', '2022-04-07 20:37:00.740000', Label.EndUncertain],
        ['OCDetect_03_recording_06', 17, '2022-04-06 21:29:00.800000', '2022-04-06 21:32:00.740000', Label.BeginUncertain],
        ['OCDetect_03_recording_06', 17, '2022-04-06 21:35:00.800000', '2022-04-06 21:41:00.740000', Label.Certain],
        ['OCDetect_03_recording_06', 19, '2022-04-06 20:00:00.800000', '2022-04-06 20:04:00.740000', Label.BeginEndUncertain],
        ['OCDetect_03_recording_06', 20, '2022-04-06 20:36:00.800000', '2022-04-06 20:37:00.740000', Label.Certain]]
    df_annotator2 = pd.DataFrame(test_annotator2, columns=['file', 'file_number', 'start', 'end', 'label'])

    type_certain = LabelMergeParameter.Intersection  # LabelMergeParameter.Intersection or LabelMergeParameter.Union
    type_uncertain = LabelMergeParameter.Union  # LabelMergeParameter.Intersection or LabelMergeParameter.Union
    type_un_cert = LabelMergeParameter.IgnoreUncertain  # LabelMergeParameter.Intersection or LabelMergeParameter.IgnoreUncertain
    expected = [
        ['OCDetect_03_recording_06', 15, '2022-04-07 20:36:00.800000', '2022-04-07 20:39:00.740000'],
        ['OCDetect_03_recording_06', 17, '2022-04-06 21:30:00.800000', '2022-04-06 21:32:00.740000'],
        ['OCDetect_03_recording_06', 17, '2022-04-06 21:35:00.800000', '2022-04-06 21:41:00.740000'],
        ['OCDetect_03_recording_06', 19, '2022-04-06 20:00:00.800000', '2022-04-06 20:05:00.740000'],
        ['OCDetect_03_recording_06', 20, '2022-04-06 20:36:00.800000', '2022-04-06 20:37:00.740000']]
    expected_output = pd.DataFrame(expected, columns=['file', 'file_number', 'start', 'end'])
    output = merge(df_annotator1, df_annotator2, type_uncertain, type_certain, type_un_cert)
    print(expected_output == output)
    #pd.testing.assert_frame_equal(output, expected_output)

def test_begin_uncertain_merge():
    # merge label begin_uncertain, union, intersection, intersection
    label = Label.BeginUncertain
    label_2 = Label.BeginUncertain
    test_annotator1 = [
        ['OCDetect_03_recording_06', 15, '2022-04-07 20:35:00.800000', '2022-04-07 20:39:00.740000', label],
        ['OCDetect_03_recording_06', 17, '2022-04-06 21:30:00.800000', '2022-04-06 21:40:00.740000', label],
        ['OCDetect_03_recording_06', 19, '2022-04-06 20:01:00.800000', '2022-04-06 20:05:00.740000', label],
        ['OCDetect_03_recording_06', 20, '2022-04-06 20:35:00.800000', '2022-04-06 20:39:00.740000', label]]
    df_annotator1 = pd.DataFrame(test_annotator1, columns=['file', 'file_number', 'start', 'end', 'label'])

    test_annotator2 = [
        ['OCDetect_03_recording_06', 15, '2022-04-07 20:36:00.800000', '2022-04-07 20:37:00.740000', Label.BeginEndUncertain],
        ['OCDetect_03_recording_06', 17, '2022-04-06 21:29:00.800000', '2022-04-06 21:32:00.740000', Label.BeginEndUncertain],
        ['OCDetect_03_recording_06', 17, '2022-04-06 21:35:00.800000', '2022-04-06 21:41:00.740000', Label.EndUncertain],
        ['OCDetect_03_recording_06', 19, '2022-04-06 20:00:00.800000', '2022-04-06 20:04:00.740000', Label.Certain],
        ['OCDetect_03_recording_06', 20, '2022-04-06 20:36:00.800000', '2022-04-06 20:37:00.740000', label_2]]
    df_annotator2 = pd.DataFrame(test_annotator2, columns=['file', 'file_number', 'start', 'end', 'label'])

    type_certain = LabelMergeParameter.Union  # LabelMergeParameter.Intersection or LabelMergeParameter.Union
    type_uncertain = LabelMergeParameter.Intersection  # LabelMergeParameter.Intersection or LabelMergeParameter.Union
    type_un_cert = LabelMergeParameter.Intersection  # LabelMergeParameter.Intersection or LabelMergeParameter.IgnoreUncertain
    expected = [
        ['OCDetect_03_recording_06', 15, '2022-04-07 20:36:00.800000', '2022-04-07 20:37:00.740000'],
        ['OCDetect_03_recording_06', 17, '2022-04-06 21:30:00.800000', '2022-04-06 21:32:00.740000'],
        ['OCDetect_03_recording_06', 17, '2022-04-06 21:35:00.800000', '2022-04-06 21:40:00.740000'],
        ['OCDetect_03_recording_06', 19, '2022-04-06 20:01:00.800000', '2022-04-06 20:05:00.740000'],
        ['OCDetect_03_recording_06', 20, '2022-04-06 20:36:00.800000', '2022-04-06 20:39:00.740000']]

    expected_output = pd.DataFrame(expected, columns=['file', 'file_number', 'start', 'end'])
    output = merge(df_annotator1, df_annotator2, type_uncertain, type_certain, type_un_cert)
    print(output == expected_output)
    #pd.testing.assert_frame_equal(output, expected_output)


def test_end_uncertain_union():
    # merge label end_uncertain with union, union, intersection
    label = Label.EndUncertain
    label_2 = Label.EndUncertain
    test_annotator1 = [
        ['OCDetect_03_recording_06', 15, '2022-04-07 20:35:00.800000', '2022-04-07 20:39:00.740000', label],
        ['OCDetect_03_recording_06', 17, '2022-04-06 21:30:00.800000', '2022-04-06 21:40:00.740000', label],
        ['OCDetect_03_recording_06', 19, '2022-04-06 20:01:00.800000', '2022-04-06 20:05:00.740000', label],
        ['OCDetect_03_recording_06', 20, '2022-04-06 20:35:00.800000', '2022-04-06 20:39:00.740000', label]]
    df_annotator1 = pd.DataFrame(test_annotator1, columns=['file', 'file_number', 'start', 'end', 'label'])

    test_annotator2 = [
        ['OCDetect_03_recording_06', 15, '2022-04-07 20:36:00.800000', '2022-04-07 20:37:00.740000', Label.BeginEndUncertain],
        ['OCDetect_03_recording_06', 17, '2022-04-06 21:29:00.800000', '2022-04-06 21:32:00.740000', Label.BeginEndUncertain],
        ['OCDetect_03_recording_06', 17, '2022-04-06 21:35:00.800000', '2022-04-06 21:41:00.740000', Label.EndUncertain],
        ['OCDetect_03_recording_06', 19, '2022-04-06 20:00:00.800000', '2022-04-06 20:04:00.740000', Label.Certain],
        ['OCDetect_03_recording_06', 20, '2022-04-06 20:36:00.800000', '2022-04-06 20:37:00.740000', label_2]]
    df_annotator2 = pd.DataFrame(test_annotator2, columns=['file', 'file_number', 'start', 'end', 'label'])

    type_certain = LabelMergeParameter.Union  # LabelMergeParameter.Intersection or LabelMergeParameter.Union
    type_uncertain = LabelMergeParameter.Union  # LabelMergeParameter.Intersection or LabelMergeParameter.Union
    type_un_cert = LabelMergeParameter.Intersection  # LabelMergeParameter.Intersection or LabelMergeParameter.IgnoreUncertain
    expected = [
        ['OCDetect_03_recording_06', 15, '2022-04-07 20:36:00.800000', '2022-04-07 20:39:00.740000'],
        ['OCDetect_03_recording_06', 17, '2022-04-06 21:30:00.800000', '2022-04-06 21:40:00.740000'],
        ['OCDetect_03_recording_06', 17, '2022-04-06 21:30:00.800000', '2022-04-06 21:41:00.740000'],
        ['OCDetect_03_recording_06', 19, '2022-04-06 20:00:00.800000', '2022-04-06 20:04:00.740000'],
        ['OCDetect_03_recording_06', 20, '2022-04-06 20:35:00.800000', '2022-04-06 20:39:00.740000']
       ]

    expected_output = pd.DataFrame(expected, columns=['file', 'file_number', 'start', 'end'])
    output = merge(df_annotator1, df_annotator2, type_uncertain, type_certain, type_un_cert)
    print(output == expected_output)
    #pd.testing.assert_frame_equal(output, expected_output)