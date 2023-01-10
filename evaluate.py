from utils.csv_utils import readDataFrameFrom, compareGraph
from pathlib import Path
import pandas as pd

if __name__ == "__main__":

    # stage2
    light_csv_dir = "result/stage2/light"
    dark_csv_dir = "result/stage2/dark"

    light_df = readDataFrameFrom(light_csv_dir)
    dark_df = readDataFrameFrom(dark_csv_dir)
    compareGraph(dark_df,light_df)

    # stage4
    light_csv_dir = "result/stage4/light"
    dark_csv_dir = "result/stage4/dark"

    light_df = readDataFrameFrom(light_csv_dir)
    dark_df = readDataFrameFrom(dark_csv_dir)
    compareGraph(dark_df,light_df)

