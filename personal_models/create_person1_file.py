import os
from glob import glob
import pandas as pd

def concatmethod():
    filelist_all = glob("../../datasets/OCDetect/preprocessed" + "*/*.csv")

    # directory to be scanned for csv
    path = '../../datasets/OCDetect/preprocessed'
    all_preprocessed = os.listdir(path)

    # filter out non-CSV files
    # keep files from person1
    csv_files = [f for f in all_preprocessed if f.startswith('OCDetect_01') and f.endswith('.csv')]
    df_list = []
    for csv in csv_files:
        file_path = os.path.join(path, csv)
        try:
            # Try reading the file using default UTF-8 encoding
            df = pd.read_csv(file_path)
            df_list.append(df)
        except UnicodeDecodeError:
            try:
                # If UTF-8 fails, try reading the file using UTF-16 encoding with tab separator
                df = pd.read_csv(file_path, sep='\t', encoding='utf-16')
                df_list.append(df)
            except Exception as e:
                print(f"Could not read file {csv} because of error: {e}")
        except Exception as e:
            print(f"Could not read file {csv} because of error: {e}")

    # Concatenate all data into one DataFrame
    person1_df = pd.concat(df_list, ignore_index=True)
    # Save the final result to a new CSV file
    person1_df.to_csv(os.path.join('../person1', 'person1_combined_file.csv'),
                      index=False)  # saving our new csv
