import pandas as pd
from data_cleansing.helpers import calc_magnitude
from data_cleansing.helpers.definitions import Sensor
import pytest


@pytest.mark.skip(reason="tested for now, skip while writing more tests")
def test_calc_magnitude():
    # Sample input data
    data = pd.DataFrame({
        "acc x": [1, 2, 3],
        "acc y": [4, 5, 6],
        "acc z": [7, 8, 9]
    })
    expected_output = pd.DataFrame({
        "acc x": [1, 2, 3],
        "acc y": [4, 5, 6],
        "acc z": [7, 8, 9],
        "mag acc": [8.12403840463596, 9.643650760992955, 11.224972160321824]
    })

    output = calc_magnitude(data, Sensor.ACCELEROMETER)

    # Compare the output with the expected result
    # assert_frame_equal throws an AssertionError when two DataFrames aren't equal
    pd.testing.assert_frame_equal(output, expected_output)

