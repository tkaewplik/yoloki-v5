from utils.csv_utils import readDataFrameFrom, compareGraph, plotDF, fitGraph, getIntoArea
from pathlib import Path
import pandas as pd
import os

if __name__ == "__main__":

    # stage2
    light_csv_dir = "result/stage2/light"
    dark_csv_dir = "result/stage2/dark"
    darklight_csv_dir = "result/stage2/darklight"

    light_df = readDataFrameFrom(light_csv_dir)
    dark_df = readDataFrameFrom(dark_csv_dir)
    darklight_df = readDataFrameFrom(darklight_csv_dir)
    # compareGraph(dark_df,light_df, title_prefix='Stage2', darklight_df=darklight_df)
    # compareGraph(dark_df,light_df, title_prefix='Stage2')
    # plotDF(darklight_df, title_prefix='Stage2 Mixed')


    # fitGraph(darklight_df)
    getIntoArea(darklight_df)
    exit(0)

    # stage4
    light_csv_dir = "result/stage4/light"
    dark_csv_dir = "result/stage4/dark"
    s4_darklight_csv_dir = "result/stage4/darklight"

    light_df = readDataFrameFrom(light_csv_dir)
    dark_df = readDataFrameFrom(dark_csv_dir)
    s4_darklight_df = readDataFrameFrom(s4_darklight_csv_dir)
    compareGraph(dark_df, light_df, title_prefix='Stage4')
    plotDF(s4_darklight_df, title_prefix='Stage4 Mixed')
    

