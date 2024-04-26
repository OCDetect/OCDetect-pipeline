# ------------------------------------------------------------------------
# Methods used for training inertial-based models
# ------------------------------------------------------------------------
# Author: Marius Bock
# E-mail: marius.bock(at)uni-siegen.de
# Adapted by .... TODO
# ------------------------------------------------------------------------
import os
import time
import numpy as np
import pandas as pd

from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
from imblearn.metrics import specificity_score
from sklearn.utils import compute_class_weight
import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from .AttendAndDiscriminate import AttendAndDiscriminate, DimAttendAndDiscriminate, compute_center_loss, get_center_delta
from .DeepConvLSTM import DeepConvLSTM
from .TinyHAR import TinyHAR, DimTinyHAR
from .OCDetectDataset import OCDetectDataset


def init_weights(net, method):
    def init_layer(m):
        if type(m) in [nn.Linear, nn.Conv2d, nn.Conv1d]:
            if method == "xavier_normal":
                torch.nn.init.xavier_normal_(m.weight)
            elif method == "xavier_uniform":
                torch.nn.init.xavier_uniform_(m.weight)
    net.apply(init_layer)
    return net  # TODO: ask marius for original code


def save_checkpoint(save_states, _, file_folder, file_name):
    torch.save(save_states, os.path.join(file_folder, file_name))


def worker_init_reset_seed(worker_id):
    seed = 769584165
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_inertial_network(train_dataset, test_dataset, cfg, ckpt_folder, ckpt_freq, resume, run=None, split_name="split_name", net=None):
    # define dataloaders
    train_loader = DataLoader(train_dataset, cfg['loader']['batch_size'], shuffle=True, num_workers=4, worker_init_fn=worker_init_reset_seed, persistent_workers=True)
    test_loader = DataLoader(test_dataset, cfg['loader']['batch_size'], shuffle=False, num_workers=4, worker_init_fn=worker_init_reset_seed, persistent_workers=True)
    
    # define network
    if net is None:
        if cfg['name'] == 'ShallowDeepConvLSTM':
            net = DeepConvLSTM(
                train_dataset.channels, train_dataset.classes, train_dataset.window_size,
                cfg['model']['conv_kernels'], cfg['model']['conv_kernel_size'],
                cfg['model']['lstm_units'], cfg['model']['lstm_layers'], cfg['model']['dropout']
                )
            print("Number of learnable parameters for DeepConvLSTM: {}".format(sum(p.numel() for p in net.parameters() if p.requires_grad)))
            criterion = nn.CrossEntropyLoss()
        elif cfg['name'] == 'attendanddiscriminate':
            net = AttendAndDiscriminate(
                train_dataset.channels, train_dataset.classes, cfg['model']['hidden_dim'], cfg['model']['conv_kernels'], cfg['model']['conv_kernel_size'], cfg['model']['enc_layers'], cfg['model']['enc_is_bidirectional'], cfg['model']['dropout'], cfg['model']['dropout_rnn'], cfg['model']['dropout_cls'], cfg['model']['activation'], cfg['model']['sa_div']
                )
            print("Number of learnable parameters for A-and-D: {}".format(sum(p.numel() for p in net.parameters() if p.requires_grad)))
            criterion = nn.CrossEntropyLoss(reduction="mean")
        elif cfg['name'] == 'tinyhar':
            net = TinyHAR((100, 1, train_dataset.features.shape[1], train_dataset.channels), train_dataset.classes, cfg['model']['conv_kernels'], cfg['model']['conv_layers'], cfg['model']['conv_kernel_size'], dropout=cfg['model']['dropout'])
            print("Number of learnable parameters for TinyHAR: {}".format(sum(p.numel() for p in net.parameters() if p.requires_grad)))
            criterion = nn.CrossEntropyLoss()
    # define criterion and optimizer
    else:  # Net is not none, define criterion only:
        if cfg['name'] == 'ShallowDeepConvLSTM' or cfg['name'] == 'tinyhar':
            criterion = nn.CrossEntropyLoss()
        elif cfg['name'] == 'attendanddiscriminate':
            criterion = nn.CrossEntropyLoss(reduction="mean")

    opt = torch.optim.Adam(net.parameters(), lr=cfg['train_cfg']['lr'], weight_decay=cfg['train_cfg']['weight_decay'])

    # use lr schedule if selected
    if cfg['train_cfg']['lr_step'] > 0:
        scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=cfg['train_cfg']['lr_step'], gamma=cfg['train_cfg']['lr_decay'])
    
    # use weighted loss if selected
    if cfg['train_cfg']['weighted_loss']:
        class_weights = compute_class_weight('balanced', classes=np.unique(train_dataset.labels), y=train_dataset.labels)
        if len(class_weights) < 2:
            class_weights = list(class_weights) + [1.0]
        criterion.weight = torch.tensor(class_weights).float().to(cfg['devices'][0])

    if resume:
        resume = os.path.join(ckpt_folder, 'ckpts', resume)
        if os.path.isfile(resume):
            if cfg['devices'][0] != "cpu":
                checkpoint = torch.load(resume, map_location = lambda storage, loc: storage.cuda(cfg['devices'][0]))
            else:
                checkpoint = torch.load(resume)
            start_epoch = checkpoint['epoch']
            net.load_state_dict(checkpoint['state_dict'])
            opt.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{:s}' (epoch {:d})".format(resume, checkpoint['epoch']))
            del checkpoint
        else:
            print("=> no checkpoint found at '{}'".format(resume))
            return
    else:
        net = init_weights(net, cfg['train_cfg']['weight_init'])
        start_epoch = 0

    net.to(cfg['devices'][0])
    for epoch in range(start_epoch, cfg['train_cfg']['epochs']):
        start_time = time.time()
        # training
        if cfg['name'] == 'attendanddiscriminate':
            net, t_losses, t_preds, t_gt = train_one_epoch(train_loader, net, opt, criterion, cfg['devices'][0], cfg['train_cfg']['beta'], cfg['train_cfg']['lr_cent'], cfg['name'])
        else:
            net, t_losses, t_preds, t_gt, = train_one_epoch(train_loader, net, opt, criterion, cfg['devices'][0])
        print("--- %s seconds ---" % (time.time() - start_time))
        # save ckpt once in a while
        if (((ckpt_freq > 0) and ((epoch + 1) % ckpt_freq == 0))):
            save_states = {
                'epoch': epoch + 1,
                'state_dict': net.state_dict(),
                'optimizer': opt.state_dict(),
            }

            file_name = 'epoch_{:03d}_{}.pth.tar'.format(epoch + 1, split_name)
            os.makedirs(os.path.join(ckpt_folder, 'ckpts'), exist_ok=True)
            save_checkpoint(save_states, False, file_folder=os.path.join(ckpt_folder, 'ckpts'), file_name=file_name)

        # validation
        if cfg['name'] == 'attendanddiscriminate':
            v_losses, v_preds, v_preds_raw, v_gt = validate_one_epoch(test_loader, net, criterion, cfg['devices'][0], cfg['name'])
        else:
            v_losses, v_preds, v_preds_raw, v_gt = validate_one_epoch(test_loader, net, criterion, cfg['devices'][0])

        if cfg['train_cfg']['lr_step'] > 0:
            scheduler.step()


        # calculate validation metrics
        v_acc = accuracy_score(v_gt, v_preds)
        v_prec = precision_score(v_gt, v_preds)
        v_rec = recall_score(v_gt, v_preds)
        v_f1 = f1_score(v_gt, v_preds)
        v_spec = specificity_score(v_gt, v_preds)

        if epoch == 0:
            v_train_gt = []
            for i, (inputs, targets) in enumerate(train_loader):
                batch_gt = targets.cpu().numpy().flatten()
                v_train_gt = np.concatenate((v_train_gt, batch_gt))
            os.makedirs(os.path.join(ckpt_folder, 'unprocessed_results'), exist_ok=True)
            np.save(os.path.join(ckpt_folder, 'unprocessed_results', 'v_train_' + cfg['name'] + "_" + split_name), v_train_gt)

        if epoch == (start_epoch + cfg['train_cfg']['epochs']) - 1:
            # save raw results (for later postprocessing)
            os.makedirs(os.path.join(ckpt_folder, 'unprocessed_results'), exist_ok=True)
            np.save(os.path.join(ckpt_folder, 'unprocessed_results', 'v_preds_' + cfg['name'] + "_" + split_name), v_preds)
            np.save(os.path.join(ckpt_folder, 'unprocessed_results', 'v_gt_' + cfg['name'] + "_" + split_name), v_gt)
            np.save(os.path.join(ckpt_folder, 'unprocessed_results', 'v_raw_' + cfg['name'] + "_" + split_name), v_preds_raw)
            np.save(os.path.join(ckpt_folder, 'unprocessed_results', 't_preds_' + cfg['name'] + "_" + split_name), t_preds)
            np.save(os.path.join(ckpt_folder, 'unprocessed_results', 't_gt_' + cfg['name'] + "_" + split_name),
                    t_gt)
            os.makedirs(os.path.join(ckpt_folder, 'processed_results'), exist_ok=True)

            results_filename = os.path.join(ckpt_folder, 'processed_results', "results.csv")
            write_headline = not os.path.isfile(results_filename)
            with open(results_filename, "a") as f:
                if write_headline:
                    f.write("model_name,epoch,acc,prec,recall,f1,spec,splitname\n")
                f.write(f"{cfg['name']}, {epoch + 1}, {v_acc}, {v_prec}, {v_rec}, {v_f1}, {v_spec}, {split_name} \n")

        # print results to terminal
        block1 = 'Epoch: [{:03d}/{:03d}]'.format(epoch, cfg['train_cfg']['epochs'])
        block2 = 'TRAINING:\tavg. loss {:.4f}'.format(np.nanmean(t_losses))
        block3 = 'VALIDATION:\tavg. loss {:.4f}'.format(np.nanmean(v_losses))
        block4 = ''
        block4  += '\n\t\tAcc {:>4.4f} (%)'.format(v_acc * 100)
        block4  += ' Prec {:>4.4f} (%)'.format(v_prec * 100)
        block4  += ' Rec/Sens {:>4.4f} (%)'.format(v_rec * 100)
        block4 += ' Spec {:>4.4f} (%)'.format(v_spec * 100)
        block4  += ' F1 {:>4.4f} (%)'.format(v_f1 * 100)

        print('\n'.join([block1, block2, block3, block4]))

        if run is not None:
            run[split_name].append({"train_loss": np.nanmean(t_losses), "val_loss": np.nanmean(v_losses), "accuracy": v_acc, "precision": np.nanmean(v_prec), "recall": np.nanmean(v_rec), 'f1': np.nanmean(v_f1)})

    return t_losses, v_losses, v_preds, v_preds_raw, v_gt, net


def train_one_epoch(loader, network, opt, criterion, gpu=None, beta=0.0003, lr_cent=0.001, network_name='deepconvlstm'):
    losses, preds, gt = [], [], []
    network.train()
    for i, (inputs, targets) in enumerate(loader):
        if network_name == 'attendanddiscriminate':
            if gpu is not None:
                inputs, targets = inputs.to(gpu), targets.view(-1).to(gpu)
            centers = network.centers
            z, output = network(inputs)
            center_loss = compute_center_loss(z, centers, targets)
            batch_loss = criterion(output, targets) + beta * center_loss
        else:
            if gpu is not None:
                inputs, targets = inputs.to(gpu), targets.to(gpu)
            output = network(inputs)
            batch_loss = criterion(output, targets)
        opt.zero_grad()
        batch_loss.backward()
        opt.step()
        if network_name == 'attendandiscriminate':
            center_deltas = get_center_delta(z.data, centers, targets, lr_cent)
            network.centers = centers - center_deltas
        # append train loss to list
        losses.append(batch_loss.item())

        # create predictions and append them to final list
        batch_preds = np.argmax(output.cpu().detach().numpy(), axis=-1)
        batch_gt = targets.cpu().numpy().flatten()
        preds = np.concatenate((preds, batch_preds))
        gt = np.concatenate((gt, batch_gt))
    
    return network, losses, preds, gt


def validate_one_epoch(loader, network, criterion, gpu=None, network_name='deepconvlstm'):
    losses, preds, preds_raw, gt = [], [], [], []

    network.eval()
    with torch.no_grad():
        # iterate over validation dataset
        for i, (inputs, targets) in enumerate(loader):
            # send inputs through network to get predictions, loss and calculate softmax probabilities
            if network_name == 'attendanddiscriminate':
                # send x and y to GPU
                if gpu is not None:
                    inputs, targets = inputs.to(gpu), targets.view(-1).to(gpu)
                z, output = network(inputs)
            else:
                # send x and y to GPU
                if gpu is not None:
                    inputs, targets = inputs.to(gpu), targets.to(gpu)
                output = network(inputs)
            batch_loss = criterion(output, targets.long())
            losses.append(batch_loss.item())

            # create predictions and append them to final list
            batch_preds_raw = output.cpu().detach().numpy()
            batch_preds = np.argmax(batch_preds_raw, axis=-1)
            batch_gt = targets.cpu().numpy().flatten()
            preds = np.concatenate((preds, batch_preds))
            if len(preds_raw) == 0:
                preds_raw = batch_preds_raw
            else:
                preds_raw = np.concatenate((preds_raw, batch_preds_raw))
            gt = np.concatenate((gt, batch_gt))
    return losses, preds, preds_raw, gt

