import pandas as pd
from helpers.definitions import Sensor
import matplotlib.pyplot as plt
from helpers.logger import logger
from helpers.misc import calc_magnitude
import numpy as np

def plot_3_axis(config: None, df: pd.DataFrame, sensor: Sensor, save_fig=False, fig_name="plot_3_axis",
                title="Line plot for the movement data over time", start_idx=None, end_idx=None) -> None:
    """
    Basic line plot that plots the 3-axis of either accelerometer or gyroscope in one plot.
    :param title: The title for the plot
    :param fig_name: The file name for the plot
    :param save_fig: True if the figure should be stored, False otherwise (default)
    :param config: The config with the path to the output folder
    :param end_idx: The end index for the dataframe if only a subset should be plotted
    :param start_idx: The start index for the dataframe if only a subset should be plotted
    :param df: The data to be plotted
    :param sensor: the sensor (accelerometer or gyroscope) to use for displaying the movement
    :return: None
    """

    if save_fig and config is None:
        logger.debug("Provide config with location for saving the figure, exiting...")
        return

    if (start_idx and end_idx) is not None:
        if start_idx >= end_idx:
            logger.debug("The provided start index is not smaller than the end index, exiting...")
            return

        if not (start_idx or end_idx) in df.index:
            logger.debug("The given indices are out of range, exiting...")
            return

    plt.subplots(figsize=(30, 10))

    if not (start_idx and end_idx) is None:
        df = df.iloc[start_idx:end_idx]

    plt.plot(df.index, df[f"{sensor.value} x"], color="red", label=f"{sensor.value} x")
    plt.plot(df.index, df[f"{sensor.value} y"], color="green", label=f"{sensor.value} y")
    plt.plot(df.index, df[f"{sensor.value} z"], color="blue", label=f"{sensor.value} z")

    plt.xlabel('Index')
    plt.ylabel(f"{sensor.value}")
    plt.title(title)
    plt.legend()

    if save_fig:
        plt.savefig(f"{config.get('output_folder')}{fig_name}.png", bbox_inches='tight')

    plt.show()


def plot_magnitude_around_label(config: None, df: pd.DataFrame, sensor: Sensor, index: int, sec_before: int,
                                sec_after: int, save_fig=False, fig_name="plot_magnitude_around_label",
                                title="Line plot for the magnitude movement data over time") -> None:
    """

    :param sec_after: the time in seconds to plot before the given index
    :param sec_before: the time in seconds to plot after the given index
    :param index: the index tp plot around
    :param config: the config with the path to the output folder
    :param df: The data to be plotted
    :param sensor: the sensor (accelerometer or gyroscope) to use for calculating the magnitude and displaying the movement
    :param fig_name: The file name for the plot
    :param save_fig: True if the figure should be stored, False otherwise (default)
    :param title: the title of the figure
    :return: None
    """

    if save_fig and config is None:
        logger.debug("Provide config with location for saving the figure, exiting...")
        return

    if index not in df.index:
        logger.debug("Index not found in provided DataFrame, exiting...")
        return

    sampling_freq = 50

    idx_steps_before = sec_before*sampling_freq
    idx_steps_after = sec_after*sampling_freq

    if (index-idx_steps_before or index+idx_steps_after) not in df.index:
        logger.debug("Before or after range is not valid, exiting...")
        return

    df = df.iloc[index-idx_steps_before:index+idx_steps_after]
    df = calc_magnitude(df, sensor)

    x_axis_recalculated = range(idx_steps_after+idx_steps_before)
    df["new axis"] = x_axis_recalculated

    plt.figure(figsize=(18, 8))
    plt.plot(x_axis_recalculated, df[f"mag {sensor.value}"], color="darkgrey")

    marker_pts_sub = df[df["user yes/no"] == 1.0]

    for index, row in marker_pts_sub.iterrows():
        labeltext = f"urge: {row['urge']}; tense: {row['tense']}; compulsive: {row['compulsive']}"
        plt.scatter(row["new axis"], row["mag acc"], marker="o", label=labeltext, c=np.random.rand(len(marker_pts_sub),), s=100,
                    zorder=10)

    labels = [*range(0, sec_before+sec_after, 10)]
    ticks = [x * sampling_freq for x in labels]

    plt.xticks(ticks=ticks, labels=labels)

    plt.xlabel('Time in seconds')
    plt.ylabel('Acceleration magnitude in $m/s^2$')
    plt.title(title)
    plt.legend()

    if save_fig:
        plt.savefig(f"{config.get('output_folder')}{fig_name}.png", bbox_inches='tight')
    else:
        plt.show()

