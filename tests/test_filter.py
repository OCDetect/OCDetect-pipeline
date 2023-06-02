import pandas as pd
from modules.filter import check_file_corrupt


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

