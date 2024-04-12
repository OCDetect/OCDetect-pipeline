import getpass
import os
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from data_cleansing.relabeling.helpers import clean_split_target_directory


def process_file(file, origin_path, target_path):
    try:
        if file.endswith(".csv") and file.startswith("OCDetect_"):# and file == "OCDetect_30_recording_11_a2ba690a-5a14-45ba-a1ae-fd2872a80bd4.csv": # and file == "OCDetect_03_recording_05_382535ec-9a0d-4359-b120-47f7605a22de.csv": # only the recordings

            origin_data = pd.read_csv(os.path.join(origin_path, file)) # get data of the original file
            origin_data['datetime'] = pd.to_datetime(origin_data['datetime'], format='%Y-%m-%d %H:%M:%S.%f') # convert 'datetime' to type datetime for comparisons

            origin_name = os.path.splitext(os.path.basename(file))[0] # returns name of file without .csv
            dir_path = os.path.join(target_path, origin_name) # directory name where new files should be stored (target_path/origin_file_name without .csv)

            indices_close_labels = get_close_labels(origin_data)

            for idx, close_label in enumerate(indices_close_labels):

                labeling_df = get_file(origin_data, close_label)

                if not os.path.exists(dir_path):
                    os.mkdir(dir_path) # create folder for the corresponding files

                new_file_name = origin_name + "_" + str(idx+1) + ".csv"
                labeling_df.to_csv(os.path.join(dir_path, new_file_name), index=False) #store each file with id

    except Exception as e:
        print(f"Error processing file {file}: {e}")


def get_close_labels(origin_data):
    # get indices of user labels in file
    indices_user_label = list(enumerate(origin_data[origin_data['user yes/no'] == 1].index))

    # indices of labels seperated, which need to be in one file
    indices_close_labels = []

    last_idx = -1  # to include the first label as well
    for idx, label_index in indices_user_label:
        if idx > last_idx:
            close_labels = [label_index]
            forward = True
            last_idx = idx
            while forward and last_idx < len(indices_user_label) - 1:
                # repeat as long as there are close user labels (5 min)
                forward = (origin_data.loc[indices_user_label[last_idx + 1][1]]['datetime'] -
                           origin_data.loc[indices_user_label[last_idx][1]]['datetime'] <= pd.Timedelta(minutes=5))
                if forward:
                    close_labels.append(indices_user_label[last_idx + 1][1])
                    last_idx += 1
            # append all user label indices for each file, that belong together
            indices_close_labels.append(close_labels)

    return indices_close_labels

def get_file(origin_data, close_label):

    labeling_df = origin_data.loc[
        (origin_data['datetime'] >= origin_data.loc[close_label[0], 'datetime'] - pd.Timedelta(minutes=5)) &
        (origin_data['datetime'] <= origin_data.loc[close_label[-1], 'datetime']),
        ['datetime', 'acc x', 'acc y', 'acc z', 'gyro x', 'gyro y', 'gyro z',
         'user yes/no']]  # create new dataframe for all files

    labeling_df['datetime'] = labeling_df['datetime'].dt.strftime(
        '%Y-%m-%d %H:%M:%S.%f')  # convert to SS:MMMMMM for label studio

    return labeling_df


def split(origin_path, target_path):
    if not os.path.exists(target_path):
        os.mkdir(target_path)  # create directory to store files if not exist

    clean_split_target_directory(target_path)

    args = (origin_path, target_path)
    # for parallel processing
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(process_file, file, *args) for file in os.listdir(origin_path)]

        for future in as_completed(futures):
            future.result()
