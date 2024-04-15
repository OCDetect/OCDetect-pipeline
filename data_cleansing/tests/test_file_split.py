import pandas as pd
from data_cleansing.relabeling.file_split import get_close_labels, get_file

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

def test_file_split():
    input_1 = [
        (0.0, '2022-04-05 09:17:10.000', -5.965994, -2.9985645, 8.47837, 0.7489205, -0.6059781, -0.68661225, None, None, None, None, None, None),
        (20000000.0, '2022-04-05 09:17:10.020', -4.608018, -3.2572267, 6.598279, 0.7489205, -0.6059781, -0.68661225, None, None, None, None, None, None),
        (30000000.0, '2022-04-05 09:22:10.000', -4.608018, -3.2572267, 6.598279, 0.7489205, -0.6059781, -0.68661225, 1.0, None, None, None, None, None),
        (40000000.0, '2022-04-05 09:23:10.020', -4.608018, -3.2572267, 6.598279, 0.7489205, -0.6059781, -0.68661225, None, None, None, None, None, None),
        (50000000.0, '2022-04-05 09:26:10.020', -4.608018, -3.2572267, 6.598279, 0.7489205, -0.6059781, -0.68661225, 1.0, None, None, None, None, None),
        (60000000.0, '2022-04-05 09:29:10.020', -4.608018, -3.2572267, 6.598279, 0.7489205, -0.6059781, -0.68661225, None, None, None, None, None, None),
        (70000000.0, '2022-04-05 09:30:10.020', -4.608018, -3.2572267, 6.598279, 0.7489205, -0.6059781, -0.68661225, 1.0, None, None, None, None, None),
        (80000000.0, '2022-04-05 09:31:10.020', -4.608018, -3.2572267, 6.598279, 0.7489205, -0.6059781, -0.68661225, None, None, None, None, None, None),
        (90000000.0, '2022-04-05 09:49:10.020', -4.608018, -3.2572267, 6.598279, 0.7489205, -0.6059781, -0.68661225, None, None, None, None, None, None),
        (100000000.0, '2022-04-05 09:52:10.020', -4.608018, -3.2572267, 6.598279, 0.7489205, -0.6059781, -0.68661225, 1.0, None, None, None, None, None)
    ]

    input_columns = ['timestamp', 'datetime', 'acc x', 'acc y', 'acc z', 'gyro x', 'gyro y', 'gyro z',
               'user yes/no', 'compulsive', 'urge', 'tense', 'ignore', 'relabeled']

    input_df = pd.DataFrame(input_1, columns=input_columns)
    input_df['datetime'] = pd.to_datetime(input_df['datetime'])

    output_1 = [
        (0,'2022-04-05 09:17:10.000000', -5.965994, -2.998565, 8.478370, 0.74892, -0.605978, -0.686612, None),
        (1,'2022-04-05 09:17:10.020000', -4.608018, -3.257227, 6.598279, 0.74892, -0.605978, -0.686612, None),
        (2,'2022-04-05 09:22:10.000000', -4.608018, -3.257227, 6.598279, 0.74892, -0.605978, -0.686612, 1.0),
        (3,'2022-04-05 09:23:10.020000', -4.608018, -3.257227, 6.598279, 0.74892, -0.605978, -0.686612, None),
        (4,'2022-04-05 09:26:10.020000', -4.608018, -3.257227, 6.598279, 0.74892, -0.605978, -0.686612, 1.0),
        (5,'2022-04-05 09:29:10.020000', -4.608018, -3.257227, 6.598279, 0.74892, -0.605978, -0.686612, None),
        (6,'2022-04-05 09:30:10.020000', -4.608018, -3.257227, 6.598279, 0.74892, -0.605978, -0.686612, 1.0),
    ]

    columns_output = ['index', 'datetime', 'acc x', 'acc y', 'acc z', 'gyro x', 'gyro y', 'gyro z', 'user yes/no']

    output_1_df = pd.DataFrame(output_1, columns=columns_output)
    output_1_df['datetime'] = pd.to_datetime(output_1_df['datetime'])
    output_1_df['datetime'] = output_1_df['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S.%f')

    output_2 = [
        (8,'2022-04-05 09:49:10.020000', -4.608018, -3.257227, 6.598279, 0.74892, -0.605978, -0.686612, None),
        (9,'2022-04-05 09:52:10.020000', -4.608018, -3.257227, 6.598279, 0.74892, -0.605978, -0.686612, 1.0),
    ]

    output_2_df = pd.DataFrame(output_2, columns=columns_output)
    output_2_df['datetime'] = pd.to_datetime(output_2_df['datetime'])
    output_2_df['datetime'] = output_2_df['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S.%f')


    outputs = [output_1_df, output_2_df]


    indices_close_labels = get_close_labels(input_df)
    for idx, close_label in enumerate(indices_close_labels):
        labeling_df = get_file(input_df, close_label).reset_index()
        pd.testing.assert_frame_equal(outputs[idx], labeling_df)
