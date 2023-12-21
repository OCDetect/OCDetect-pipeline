import getpass
import os
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

origin_path = "/dhc/home/"+getpass.getuser()+"/datasets/OCDetect/preprocessed"  # set path where files should be taken from
target_path = "/dhc/home/"+getpass.getuser()+"/datasets/OCDetect/relabel_split" # set path where files should be stored

if not os.path.exists(target_path):
    os.mkdir(target_path) # create directory to store files if not exist


def process_file(file):
    try:
        if file.endswith(".csv") and file.startswith("OCDetect_"): # only the recordings
            df = pd.read_csv(os.path.join(origin_path, file))
            df['datetime'] = pd.to_datetime(df['datetime'], format='%Y-%m-%d %H:%M:%S.%f')

            origin_name = os.path.splitext(os.path.basename(file))[0] # returns name of file without .csv
            dir_path = os.path.join(target_path, origin_name) # directory name where new files should be stored (target_path/origin_file_name without .csv)

            file_number = 1 # id for each subfile

            current = df.iloc[0]['datetime'] - pd.Timedelta(milliseconds=20) # set starting datetime minus a bit so the first row will be included

            for index, row in df.iterrows():
                # if the last user labeled row is not already in a file
                if row['datetime'] > current:
                    if row['user yes/no'] == 1:
                        close_label = True
                        current = row["datetime"]
                        start = (current - pd.Timedelta(minutes=5))

                        # expanding the end of file (current) until there are no more user labels in the next 5 minutes
                        while close_label:
                            forward_df = df[(df['datetime'] > current) & (df['datetime'] <= (current + pd.Timedelta(minutes=5)))]
                            close_label = ((forward_df['user yes/no'] == 1).any())
                            if close_label:
                                current = forward_df.loc[forward_df['user yes/no'] == 1, 'datetime'].max()

                        labeling_df = df[(df['datetime'] >= start) & (df['datetime'] <= current)]

                        labeling_df = labeling_df.loc[:, ['datetime', 'acc x', 'acc y', 'acc z', 'gyro x', 'gyro y', 'gyro z', 'user yes/no']] # reducing the file
                        labeling_df['datetime'] = labeling_df['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S.%f')

                        if not os.path.exists(dir_path):
                            os.mkdir(dir_path)
                        new_file_name = origin_name + "_" + str(file_number) + ".csv"

                        labeling_df.to_csv(os.path.join(dir_path, new_file_name), index=False)

                        file_number += 1

    except Exception as e:
        print(f"Error processing file {file}: {e}")


# Use ProcessPoolExecutor to parallelize file processing
with ProcessPoolExecutor(max_workers=63) as executor:
    futures = [executor.submit(process_file, file) for file in os.listdir(origin_path)]

    for future in as_completed(futures):
        future.result()

print("done")

# INPUT
# timestamp,datetime,acc x,acc y,acc z,gyro x,gyro y,gyro z,user yes/no,compulsive,urge,tense,ignore,relabeled
# 0.0,2022-04-05 09:17:10.000,-5.965994,-2.9985645,8.47837,0.7489205,-0.6059781,-0.68661225,,,,,0,0
# 20000000.0,2022-04-05 09:17:10.020,-4.608018,-3.2572267,6.598279,0.7489205,-0.6059781,-0.68661225,,,,,0,0
# 30000000.0,2022-04-05 09:22:10.000,-4.608018,-3.2572267,6.598279,0.7489205,-0.6059781,-0.68661225,1.0,,,,0,0
# 40000000.0,2022-04-05 09:23:10.020,-4.608018,-3.2572267,6.598279,0.7489205,-0.6059781,-0.68661225,,,,,0,0
# 50000000.0,2022-04-05 09:26:10.020,-4.608018,-3.2572267,6.598279,0.7489205,-0.6059781,-0.68661225,1.0,,,,0,0
# 60000000.0,2022-04-05 09:29:10.020,-4.608018,-3.2572267,6.598279,0.7489205,-0.6059781,-0.68661225,,,,,0,0
# 70000000.0,2022-04-05 09:30:10.020,-4.608018,-3.2572267,6.598279,0.7489205,-0.6059781,-0.68661225,1.0,,,,0,0
# 80000000.0,2022-04-05 09:31:10.020,-4.608018,-3.2572267,6.598279,0.7489205,-0.6059781,-0.68661225,,,,,0,0
# 90000000.0,2022-04-05 09:49:10.020,-4.608018,-3.2572267,6.598279,0.7489205,-0.6059781,-0.68661225,,,,,0,0
# 100000000.0,2022-04-05 09:52:10.020,-4.608018,-3.2572267,6.598279,0.7489205,-0.6059781,-0.68661225,1.0,,,,0,0
#
# OUTPUT
#                  datetime     acc x     acc y  ...    gyro y    gyro z  user yes/no
# 0 2022-04-05 09:17:10.000000 -5.965994 -2.998565  ... -0.605978 -0.686612          NaN
# 1 2022-04-05 09:17:10.020000 -4.608018 -3.257227  ... -0.605978 -0.686612          NaN
# 2 2022-04-05 09:22:10.000000 -4.608018 -3.257227  ... -0.605978 -0.686612          1.0
# 3 2022-04-05 09:23:10.020000 -4.608018 -3.257227  ... -0.605978 -0.686612          NaN
# 4 2022-04-05 09:26:10.020000 -4.608018 -3.257227  ... -0.605978 -0.686612          1.0
# 5 2022-04-05 09:29:10.020000 -4.608018 -3.257227  ... -0.605978 -0.686612          NaN
# 6 2022-04-05 09:30:10.020000 -4.608018 -3.257227  ... -0.605978 -0.686612          1.0
#
# [7 rows x 8 columns]
#                  datetime     acc x     acc y  ...    gyro y    gyro z  user yes/no
# 8 2022-04-05 09:49:10.020000 -4.608018 -3.257227  ... -0.605978 -0.686612          NaN
# 9 2022-04-05 09:52:10.020000 -4.608018 -3.257227  ... -0.605978 -0.686612          1.0
#
# [2 rows x 8 columns]
