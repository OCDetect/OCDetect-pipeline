import pandas as pd
from sklearn.preprocessing import StandardScaler
from helpers.logger import logger

def std_scaling_data(features_list, settings):
    logger.info("Scaling Features")

    scaler = settings.get("scaler")
    if scaler == "standard":
        std_scaler = StandardScaler()
        return pd.DataFrame(std_scaler.fit_transform(features_list))

    return None

