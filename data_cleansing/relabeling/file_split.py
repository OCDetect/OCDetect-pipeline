import yaml
import sys
import socket
from misc import logger
import getpass
import os
import pandas as pd


export_subfolder = "/dhc/home/"+getpass.getuser()+"/datasets/OCDetect/preprocessed/"
splits_subfolder = "/dhc/home/"+getpass.getuser()+"/datasets/OCDetect/relabel_split/"

file = export_subfolder+"OCDetect_09_recording_18_8c249737-cd58-444a-a22d-a2b49bef1a0c.csv"
noo = splits_subfolder+"OCDetect_09_recording_18_8c249737-cd58-444a-a22d-a2b49bef1a0c"
df = pd.read_csv(file)

#print(df["user yes/no"].unique())
df['datetime'] = pd.to_datetime(df['datetime'])
c = 1
for index, row in df.iterrows():
    if row["user yes/no"] == 1:
        print(row)
        time = row["datetime"]
        print(c)
        # new_file = file + "_" + str(c)
        # os.mkdir(splits_subfolder+"OCDetect_09_recording_18_8c249737-cd58-444a-a22d-a2b49bef1a0c")
        if not os.path.exists(file):
            os.makedirs(noo)
        actual = noo+"_"+str(c)+".csv"
        new_df = df[(df['datetime'] >= (time - pd.Timedelta(minutes=5))) & (df['datetime'] <= time)]
        df_reduziert = new_df.loc[:, ['datetime', 'acc x', 'acc y', 'acc z', 'gyro x', 'gyro y', 'gyro z', 'user yes/no']]
        print(df_reduziert)
        df_reduziert.to_csv(actual, index=False)
        c += 1






# file_names = os.listdir(export_subfolder)
#
# for file in file_names:
#     current_file = export_subfolder+"file"

