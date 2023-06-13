import pytest
from modules.csv_loader import initial_handwash_time
import yaml
import socket


@pytest.mark.skip(reason="tested for now, skip while writing more tests")
@pytest.fixture()
def load_config():
    config_file_name = "config/config.yaml"
    with open(config_file_name, "r") as config_stream:
        configs = yaml.safe_load(config_stream)
        active_config = [list(entry.values())[0] for entry in configs if
                         list(entry.values())[0].get("hostname", "") == socket.gethostname()][0]
    return active_config

@pytest.mark.skip(reason="tested for now, skip while writing more tests")
@pytest.mark.parametrize("subject, expected_time", [
    ("01", 39), ("02", 39), ("03", 20), ("04", 18), ("05", 47), ("07", 45), ("09", 22), ("10", 39), ("11", 54),
    ("12", 33), ("13", 39), ("15", 43), ("18", 23), ("19", 61), ("20", 40), ("22", 26), ("24", 60), ("25", 42),
    ("27", 31), ("29", 55), ("30", 44)
])
def test_initial_handwash_time(load_config, subject, expected_time):
    assert initial_handwash_time(subject, load_config) == expected_time


