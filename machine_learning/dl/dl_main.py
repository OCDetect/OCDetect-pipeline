# ------------------------------------------------------------------------
# Run the three baseline deep learning models on the dataset using loso-CV
# ------------------------------------------------------------------------
# Robin Burchard
# Email: robin.burchard(at)uni-siegen.de
# ------------------------------------------------------------------------


import numpy as np
import pandas as pd
import torch.cuda

from .train import *
from torch.utils.data import DataLoader

def dl_main(config: dict):
    # ---------------------------------------------------------------------------------------------------------
    # 1. Prepare the Dataset according to the config

    train_test_split = [[[1,2], [5]], [[5, 2], [1]]]

    # ---------------------------------------------------------------------------------------------------------
    # 2. Cross-Validation loop and results-gathering.
    dl_config["devices"] = ["cuda"] if torch.cuda.is_available() else ["cpu"]
    for train_subs, test_subs in train_test_split:
        for model_name in ["ShallowDeepConvLSTM", "attendanddiscriminate", "tinyhar"]:
            dl_config["name"] = model_name
            dl_config["model"] = dl_config[model_name]
            t_losses, v_losses, v_preds, v_gt = run_inertial_network(train_subs, test_subs, dl_config,
                                 config.get("ml_results_folder") + "dl_checkpoints/", 10, resume=False)


    # ---------------------------------------------------------------------------------------------------------
    # 3. Generation of relevant plots and tables in tex, according to plan

    pass

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
    "loader" : {
        "batch_size":1024,

    },
    "train_cfg": {
        "lr": 0.0001,
        "lr_decay": 0.9,
        "lr_step": 10,
        "epochs": 100,
        "weight_decay": 0.000001,
        "weight_init": 'xavier_normal',
        "weighted_loss": True,
        "beta": 1,  # for tinyhar
        "lr_cent": 1.0  # for tinyhar
    },
    "dataset": {"window_size": 150
    }
}  # Config and hyperparameters by Marius Bock (marius.bock(at)uni-siegen.de)

