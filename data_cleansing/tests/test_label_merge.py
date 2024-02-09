from data_cleansing.helpers.definitions import Label, LabelMergeParameter, label_mapping, string_to_parameter
from data_cleansing.relabeling.label_merge import merge
import pandas as pd

# Die hier müssen auch variieren
# Merging settings
type_certain = LabelMergeParameter.Intersection # LabelMergeParameter.Intersection or LabelMergeParameter.Union
type_uncertain = LabelMergeParameter.Intersection # LabelMergeParameter.Intersection or LabelMergeParameter.Union
type_un_cert = LabelMergeParameter.IgnoreUncertain # LabelMergeParameter.Intersection or LabelMergeParameter.IgnoreUncertain


# label = Label.Certain
# label_2 = Label.Certain
# test_first = [['OCDetect_03_recording_06', 20, '2022-04-06 20:35:00.800000', '2022-04-06 20:39:00.740000', Label.BeginEndUncertain],
#               ['OCDetect_03_recording_06', 19, '2022-04-06 20:01:00.800000', '2022-04-06 20:05:00.740000', label],
#               ['OCDetect_03_recording_06', 18, '2022-04-06 20:40:58.800000', '2022-04-06 20:45:00.740000', label],
#               ['OCDetect_03_recording_07', 1, '2022-04-06 20:35:00.800000', '2022-04-06 20:39:00.740000', label],
#               ['OCDetect_03_recording_08', 1, '2022-04-06 20:35:00.800000', '2022-04-06 20:39:00.740000', label],
#               ['OCDetect_03_recording_06', 17, '2022-04-06 21:30:00.800000', '2022-04-06 21:40:00.740000', label],
#               ['OCDetect_03_recording_06', 15, '2022-04-07 20:35:00.800000', '2022-04-07 20:39:00.740000', label]]
#
# test_second = [['OCDetect_03_recording_06', 20, '2022-04-06 20:36:00.800000', '2022-04-06 20:37:00.740000', label_2],
#                ['OCDetect_03_recording_06', 19, '2022-04-06 20:00:00.800000', '2022-04-06 20:04:00.740000', label_2],
#                ['OCDetect_03_recording_06', 18, '2022-04-06 20:46:00.800000', '2022-04-06 20:48:00.740000', label_2],
#                ['OCDetect_03_recording_07', 2, '2022-04-06 20:35:00.800000', '2022-04-06 20:39:00.740000', label_2],
#                ['OCDetect_03_recording_09', 1, '2022-04-06 20:35:00.800000', '2022-04-06 20:39:00.740000', label_2],
#                ['OCDetect_03_recording_06', 17, '2022-04-06 21:29:00.800000', '2022-04-06 21:32:00.740000', label_2],
#                ['OCDetect_03_recording_06', 17, '2022-04-06 21:35:00.800000', '2022-04-06 21:41:00.740000', label_2],
#                ['OCDetect_03_recording_06', 15, '2022-04-07 20:36:00.800000', '2022-04-07 20:37:00.740000', Label.BeginEndUncertain]]
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