from pathlib import Path
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib
from CosinorPy import file_parser, cosinor, cosinor1


# find the average of get into eating zone
def getIntoArea(df, preiod_accum = '30min'):
    print("--- get into area ---")
    print(df)

    df = df.sort_values(by=['timestamp'])
    df = df.set_index('timestamp')

    # todo: remove noise
    df_p = df.diff()

    print(df_p)

    df_p[df_p['consuming_area'] < 0] = 0
    df_p[df_p['drinking_area'] < 0] = 0
    df_p = df_p.groupby(pd.Grouper(level='timestamp', freq=preiod_accum)).sum()

    print(df_p)
    print(df_p['consuming_area'].mean())
    print(df_p['consuming_area'].sem())
    print(df_p['drinking_area'].mean())
    print(df_p['drinking_area'].sem())

    # find average

    df_p.plot(
        y=['consuming_area', 'drinking_area'], kind='line'
    )
    plt.show()

def getHowLong():
    print("--- find how long stay ---")
    

# reading csv and find the fit graph for it
def fitGraph(df, preiod_accum = '30min'):
    # tests = df.consuming_area.unique()
    # df2 = df[df['consuming_area'].isin(tests[:20])] # only for 20 genes

    df = df.sort_values(by=['timestamp'])
    df = df.set_index('timestamp')
    df = df.groupby(pd.Grouper(level='timestamp', freq=preiod_accum)).sum()

    # reset index (timestamp)
    df = df.reset_index()

    df['test'] = 'test1'
    max_consuming = df['consuming_area'].max()
    df['x'] = df['timestamp'].apply(lambda x: int(x.timestamp())-1666265400)
    df['y'] = df['consuming_area'] / max_consuming
    print(df)
    print(len(df.index))
    # cosinor.periodogram_df(df)
    r = cosinor.fit_group(df, n_components=1, period=24)
    print("fsdfdsfdsfsdrrr")
    print(r)

    # cosinor.periodogram_df(df)

    # df.plot(
    #     y=['y'], kind='line'
    # )
    # plt.show()

    return

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

def plotDF(df, title_prefix='Stage2 Mixed', preiod_accum = '30min'):

    df = df.sort_values(by=['timestamp'])
    df = df.set_index('timestamp')
    df = df.groupby(pd.Grouper(level='timestamp', freq=preiod_accum)).sum()

    print('plot')
    print(df)

    fig, ax = plt.subplots()

    df_title = "{} | Diet Consuming frequency in one day (Group by {})".format(title_prefix, preiod_accum)
    
    df.plot(y=['consuming_area'], kind='line', ax=ax, title=df_title)

    if title_prefix == 'Stage2 Mixed':
        ax.axvline("2022-10-21 09:00:00", color="red", linestyle="--")
        ax.axvline("2022-10-22 09:00:00", color="red", linestyle="--")
        ax.axvline("2022-10-23 09:00:00", color="red", linestyle="--")
        ax.axvline("2022-10-20 21:00:00", color="red", linestyle="--")
        ax.axvline("2022-10-21 21:00:00", color="red", linestyle="--")
        ax.axvline("2022-10-22 21:00:00", color="red", linestyle="--")
    else:
        ax.axvline("2022-11-03 09:00:00", color="red", linestyle="--")
        ax.axvline("2022-11-04 09:00:00", color="red", linestyle="--")
        ax.axvline("2022-11-05 09:00:00", color="red", linestyle="--")
        ax.axvline("2022-11-02 21:00:00", color="red", linestyle="--")
        ax.axvline("2022-11-03 21:00:00", color="red", linestyle="--")
        ax.axvline("2022-11-04 21:00:00", color="red", linestyle="--")

    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Accumulation of found crickets")

    plt.show()

# compare 2 grapg
def compareGraph(dark_df, light_df, title_prefix='Stage2', preiod_accum = '30min', darklight_df = None):

    dark_df = dark_df.sort_values(by=['timestamp'])
    dark_df = dark_df.set_index('timestamp')
    dark_df = dark_df.groupby(pd.Grouper(level='timestamp', freq=preiod_accum)).sum()

    light_df = light_df.sort_values(by=['timestamp'])
    light_df = light_df.set_index('timestamp')
    light_df = light_df.groupby(pd.Grouper(level='timestamp', freq=preiod_accum)).sum()

    if darklight_df is not None:
        darklight_df = darklight_df.sort_values(by=['timestamp'])
        darklight_df = darklight_df.set_index('timestamp')
        darklight_df = darklight_df.groupby(pd.Grouper(level='timestamp', freq=preiod_accum)).sum()

    matplotlib.rcParams['font.family'] = 'Times New Roman'
    matplotlib.rcParams['font.size'] = 12

    join_df = light_df.join(dark_df, lsuffix='_light', rsuffix='_dark')
    fig, ax = plt.subplots()

    df_title = "{} | Diet Consuming frequency in one day (Group by {})".format(title_prefix, preiod_accum)
    if darklight_df is not None:
        join_df = join_df.join(darklight_df, rsuffix='_darklight')

        join_df.plot(
            y=['consuming_area_light', 'consuming_area_dark', 'consuming_area'], kind='line', ax=ax, title=df_title
        )
        ax.legend(["Dark", "Bright", "Mixed"])
    else:
        join_df.plot(
            y=['consuming_area_light', 'consuming_area_dark'], kind='line', ax=ax, title=df_title
        )
        ax.legend(["Dark", "Bright"])

    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Accumulation of found crickets")
    
    plt.show()

def is_time_between(begin_time, end_time, check_time=None):
    # If check time is not given, default to current UTC time
    check_time = check_time or datetime.utcnow().time()
    if begin_time < end_time:
        return check_time >= begin_time and check_time <= end_time
    else: # crosses midnight
        return check_time >= begin_time or check_time <= end_time
