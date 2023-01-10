from pathlib import Path
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib

# read all csv files in the target folder and then merge into 1 dataframe and return
def readDataFrameFrom(dir_path):

    dataframe = None

    for path in os.listdir(dir_path):

        full_path = os.path.join(dir_path, path)

        if os.path.isfile(full_path):

            print(full_path)

            if (full_path.endswith(".csv")):

                df = pd.read_csv(full_path, parse_dates=["timestamp"])
                df = df.mask(df.eq('None')).dropna()

                df['timestamp'] = pd.to_datetime(df['timestamp'])

                if dataframe is None:
                    dataframe = df
                else:
                    dataframe = pd.concat([df, dataframe], ignore_index=True)

    dataframe = dataframe.sort_values(by=['timestamp'])

    return dataframe

# compare 2 grapg
def compareGraph(dark_df, light_df, preiod_accum = '30min'):

    dark_df = dark_df.sort_values(by=['timestamp'])
    dark_df = dark_df.set_index('timestamp')
    dark_df = dark_df.groupby(pd.Grouper(level='timestamp', freq=preiod_accum)).sum()

    light_df = light_df.sort_values(by=['timestamp'])
    light_df = light_df.set_index('timestamp')
    light_df = light_df.groupby(pd.Grouper(level='timestamp', freq=preiod_accum)).sum()

    matplotlib.rcParams['font.family'] = 'Times New Roman'
    matplotlib.rcParams['font.size'] = 12

    join_df =light_df.join(dark_df, lsuffix='_light', rsuffix='_dark')
    fig, ax = plt.subplots()

    join_df.plot(
        y=['consuming_area_light', 'consuming_area_dark'], kind='line', ax=ax, title='Diet Consuming frequency in one day (Group by 30 minutes)'
    )

    ax.legend(["Dark", "Bright"])
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Accumulation of found crickets")
    
    plt.show()
