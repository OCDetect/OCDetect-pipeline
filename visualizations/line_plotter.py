import pandas as pd
from helpers.definitions import Sensor
import matplotlib.pyplot as plt
from helpers.logger import logger

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