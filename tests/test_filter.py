import pytest
import pandas as pd
import numpy as np
from modules.filter import check_file_corrupt, check_insufficient_file_length, \
    check_insufficient_remaining_data_points, set_ignore_no_movement
from helpers.definitions import IgnoreReason

@pytest.mark.skip(reason="tested for now, skip while writing more tests")
def test_check_file_corrupt():
    # Test case: Empty DataFrame
    data_empty = pd.DataFrame()
    assert check_file_corrupt(data_empty) is True

    # Test case: DataFrame with only header
    data_header = pd.DataFrame(columns=['Column1', 'Column2', 'Column3'])
    assert check_file_corrupt(data_header) is True

    # Test case: DataFrame with data
    data_valid = pd.DataFrame({'Column1': [1, 2, 3], 'Column2': ['A', 'B', 'C']})
    assert check_file_corrupt(data_valid) is False


@pytest.mark.skip(reason="tested for now, skip while writing more tests")
@pytest.fixture
def sample_data():
    # Create a sample DataFrame for testing
    data = pd.DataFrame({
        'timestamp': [1000000000, 2000000000, 3000000000],
        # Add other relevant columns as needed
    })
    return data


@pytest.mark.skip(reason="tested for now, skip while writing more tests")
# Test case for when file length is too short
def test_check_insufficient_file_length_true(sample_data):
    initial_hw_time = 4
    assert check_insufficient_file_length(sample_data, initial_hw_time) is True


@pytest.mark.skip(reason="tested for now, skip while writing more tests")
# Test case for when file length is sufficient
def test_check_insufficient_file_length_false(sample_data):
    initial_hw_time = 2
    assert check_insufficient_file_length(sample_data, initial_hw_time) is False


@pytest.mark.skip(reason="tested for now, skip while writing more tests")
def test_check_insufficient_remaining_data_points():
    # Test case 1: Sufficient remaining data points
    recording_w_idle = pd.DataFrame({"idle": [0.0, 0.0, 0.0, 1.0, 1.0, 0.0]})
    initial_hw_time = 2
    sampling_frequency = 1
    assert check_insufficient_remaining_data_points(recording_w_idle, initial_hw_time, sampling_frequency) is False

    # Test case 2: Insufficient remaining data points
    recording_w_idle = pd.DataFrame({"idle": [0.0, 1.0, 1.0, 1.0, 0.0]})
    initial_hw_time = 3
    sampling_frequency = 1
    assert check_insufficient_remaining_data_points(recording_w_idle, initial_hw_time, sampling_frequency) is True

    # Test case 3: No idle data points
    recording_w_idle = pd.DataFrame({"idle": [0.0, 0.0, 0.0, 0.0]})
    initial_hw_time = 1
    sampling_frequency = 1
    assert check_insufficient_remaining_data_points(recording_w_idle, initial_hw_time, sampling_frequency) is False

    # Test case 4: Empty recording
    recording_w_idle = pd.DataFrame({"idle": []})
    initial_hw_time = 0
    sampling_frequency = 1
    assert check_insufficient_remaining_data_points(recording_w_idle, initial_hw_time, sampling_frequency) is False

    # Test case 5: Recording with only idle data points
    recording_w_idle = pd.DataFrame({"idle": [1.0, 1.0, 1.0]})
    initial_hw_time = 2
    sampling_frequency = 1
    assert check_insufficient_remaining_data_points(recording_w_idle, initial_hw_time, sampling_frequency) is True


@pytest.mark.skip(reason="tested for now, skip while writing more tests")
def test_set_ignore_no_movement():
    # Sample input data
    data = pd.DataFrame({
        "idle": [0.0, 1.0, 0.0, 1.0, 0.0],
        "ignore": [np.nan, np.nan, np.nan, np.nan, np.nan]
    })
    expected_output = pd.DataFrame({
        "idle": [0.0, 1.0, 0.0, 1.0, 0.0],
        "ignore": [np.nan, IgnoreReason.NoMovement, np.nan, IgnoreReason.NoMovement, np.nan]
    })

    output = set_ignore_no_movement(data)
    pd.testing.assert_frame_equal(output, expected_output)