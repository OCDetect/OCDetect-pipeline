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
        if file.endswith(".csv") and file.startswith("OCDetect_"): # and file == "OCDetect_30_recording_11_a2ba690a-5a14-45ba-a1ae-fd2872a80bd4.csv": # and file == "OCDetect_03_recording_05_382535ec-9a0d-4359-b120-47f7605a22de.csv": # only the recordings
            origin_data = pd.read_csv(os.path.join(origin_path, file)) # get data of the original file

            origin_data['datetime'] = pd.to_datetime(origin_data['datetime'], format='%Y-%m-%d %H:%M:%S.%f') # convert 'datetime' to type datetime for comparisons

            origin_name = os.path.splitext(os.path.basename(file))[0] # returns name of file without .csv
            dir_path = os.path.join(target_path, origin_name) # directory name where new files should be stored (target_path/origin_file_name without .csv)

            # get indices of user labels in file
            indices_user_label = list(enumerate(origin_data[origin_data['user yes/no'] == 1].index))

            # indices of labels seperated, which need to be in one file
            indices_close_labels = []

            last_idx = -1 # to include the first label as well
            for idx, label_index in indices_user_label:
                if idx > last_idx:
                    close_labels = [label_index]
                    forward = True
                    last_idx = idx
                    while forward and last_idx < len(indices_user_label)-1: # repeat as long as there are close user labels (5 min)
                        forward = (origin_data.loc[indices_user_label[last_idx+1][1]]['datetime'] -
                                   origin_data.loc[indices_user_label[last_idx][1]]['datetime'] <= pd.Timedelta(minutes = 5))
                        if forward:
                            close_labels.append(indices_user_label[last_idx+1][1])
                            last_idx +=1
                    indices_close_labels.append(close_labels) # append all user label indices for each file, that belong together

            for idx, close_label in enumerate(indices_close_labels):
                labeling_df = origin_data.loc[
                    (origin_data['datetime'] >= origin_data.loc[close_label[0], 'datetime'] - pd.Timedelta(minutes=5)) &
                    (origin_data['datetime'] <= origin_data.loc[close_label[-1], 'datetime']),
                    ['datetime', 'acc x', 'acc y', 'acc z', 'gyro x', 'gyro y', 'gyro z', 'user yes/no']] # create new dataframe for all files

                labeling_df['datetime'] = labeling_df['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S.%f') #convert to SS:MMMMMM for label studio

                if not os.path.exists(dir_path):
                    os.mkdir(dir_path) # create folder for the corresponding files

                new_file_name = origin_name + "_" + str(idx+1) + ".csv"
                labeling_df.to_csv(os.path.join(dir_path, new_file_name), index=False) #store each file with id

    except Exception as e:
        print(f"Error processing file {file}: {e}")

# for parallel processing
with ProcessPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(process_file, file) for file in os.listdir(origin_path)]

    for future in as_completed(futures):
        future.result()


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
