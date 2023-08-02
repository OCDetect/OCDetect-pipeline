import pandas as pd
import numpy as np
from data_cleansing.modules import relabel, RelabelStrategy
from data_cleansing.helpers.definitions import IgnoreReason, HandWashingType


def test_relabel():
    # (dfs: List[pd.DataFrame], config: dict, settings: dict, subject: str, use_ignore: bool = True,
    # strategy: RelabelStrategy = RelabelStrategy.TimeSpan) -> List[pd.DataFrame]:
    df = pd.DataFrame({"timestamp": list(np.linspace(0, 100000 * 1e7, 50001)),
                       "user yes/no": 0.0,
                       "ignore": IgnoreReason.DontIgnore,
                       "compulsive": 0.0})
    df.loc[5000, "user yes/no"] = 1.0
    df.loc[10000, "user yes/no"] = 1.0
    df.loc[10000, "compulsive"] = 1.0

    expected_df = df.copy()
    expected_df["relabeled"] = HandWashingType.NoHandWash
    expected_df.loc[5000 - 250 - 38*50: 5000-250, "relabeled"] = HandWashingType.Routine
    expected_df.loc[10000 - 250 - 38 * 50: 10000 - 250, "relabeled"] = HandWashingType.Compulsive
    dfs = [df]
    config = {}
    settings = {"relabel_offset": 5, "relabel_duration": 38}
    subject = "00"
    use_ignore = True
    strategy = RelabelStrategy.TimeSpan
    relabeled_df = relabel(dfs, config, settings, subject, use_ignore=True, strategy=strategy)[0]
    pd.testing.assert_frame_equal(relabeled_df, expected_df)

