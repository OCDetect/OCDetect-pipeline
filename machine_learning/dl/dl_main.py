# ------------------------------------------------------------------------
# Run the three baseline deep learning models on the dataset using loso-CV
# ------------------------------------------------------------------------
# Robin Burchard
# Email: robin.burchard(at)uni-siegen.de
# ------------------------------------------------------------------------

from misc.logger import logger
import numpy as np
import pandas as pd
import torch.cuda
from sklearn.model_selection import LeaveOneOut, StratifiedKFold

from .train import *
from torch.utils.data import DataLoader


def dl_main(config: dict, settings: dict, users, subset="all"):
    # ---------------------------------------------------------------------------------------------------------
    # 1. Prepare the Dataset according to the config
    if type(users) == pd.DataFrame:
        users = users["user"]
    users = users.unique()
    if subset != "all":
        if subset == "ocd_diagnosed_only":
            users = np.intersect1d(users, settings.get("ocd_diagnosed_subjects"))
        elif subset == "trustworthy_only":
            users = np.intersect1d(users, settings.get("trustworthy_subjects"))
            logger.info(f"Training on subset trustworthy: {users}")
    loo = LeaveOneOut()
    split = loo.split(users)

    # ---------------------------------------------------------------------------------------------------------
    # 2. Cross-Validation loop and results-gathering.
    dl_config["devices"] = ["cuda"] if torch.cuda.is_available() else ["cpu"]
    ckpt_folder = config.get("ml_results_folder") + "dl_checkpoints/"
    try:
        results_df = pd.read_csv(os.path.join(ckpt_folder, 'processed_results', "results.csv"))
    except FileNotFoundError:
        results_df = None
    for train_idx, test_idx in split:
        train_subs = users[train_idx]
        test_subs = users[test_idx]
        for model_name in ["ShallowDeepConvLSTM", "tinyhar"]:
            dl_config["name"] = model_name
            dl_config["model"] = dl_config[model_name]
            split_name = f"train_{str(train_subs)}_test_{str(test_subs)}"
            if results_df is not None:
                if len(results_df[results_df.splitname==split_name]) == 2:
                    logger.info(f"Split {split_name} already finished, skipping...")
                    continue
            #train_dataset = OCDetectDataset(train_subs, dl_config['dataset']['window_size'], model=dl_config['name'])
            #test_dataset = OCDetectDataset(test_subs, dl_config['dataset']['window_size'], model=dl_config['name'])

            train_dataset = OCDetectDataset(train_subs, dl_config['dataset']['window_size'], model=dl_config['name'])
            test_dataset = OCDetectDataset(test_subs, dl_config['dataset']['window_size'], model=dl_config['name'])

            t_losses, v_losses, v_preds, v_gt = run_inertial_network(train_dataset, test_dataset, dl_config,
                                 ckpt_folder, 10, resume=False, split_name=split_name)

    # ---------------------------------------------------------------------------------------------------------
    # 3. Generation of relevant plots and tables in tex, according to plan
    # TODO


dl_config = {
    "name" : "ShallowDeepConvLSTM",
    "ShallowDeepConvLSTM": {
        "conv_kernels": 64,
        "conv_kernel_size": 9,
        "lstm_units": 128,
        "lstm_layers": 1,
        "dropout": 0.5
    },
    "attendanddiscriminate": {
        "hidden_dim": 128,
        "conv_kernels": 64,
        "conv_kernel_size": 9,
        "enc_layers": 2,
        "enc_is_bidirectional": False,
        "dropout": 0.5,
        "dropout_rnn": 0.5,
        "dropout_cls": 0.5,
        "activation": 'ReLU',
        "sa_div": 1
    },
    "tinyhar": {
        "conv_kernels": 20,
        "conv_layers": 4,
        "conv_kernel_size": 9,
        "dropout": 0.5
    },
    "loader": {
        "batch_size": 4096,
    },
    "train_cfg": {
        "lr": 0.0001,
        "lr_decay": 0.8,
        "lr_step": 5,
        "epochs": 30,
        "weight_decay": 0.000001,
        "weight_init": 'xavier_uniform',
        "weighted_loss": True,
        "beta": 1,  # for attentanddiscriminate
        "lr_cent": 1.0  # for attentanddiscrimate
    },
    "dataset": {"window_size": 250  # TODO: parameterise
    }
}  # Config and hyperparameters by Marius Bock (marius.bock(at)uni-siegen.de)

