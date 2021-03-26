"""3D SPECT image PD classification experiment
"""
import random
from collections import defaultdict
from copy import deepcopy
from inspect import signature

import numpy as np
import pandas as pd
import posix_ipc
import signac
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as tdata
from flow import FlowProject, directives
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix
from sklearn.utils import check_random_state

import metrics
from beta_quartile import bisect3
from graph_oversampling import oversample_class
from network import BrainNet, OrdinalNet, NominalNet, ordinal_distance_loss

project = signac.get_project()
project_data_semaphore_name = f'/{project.id}_project-data'


def project_dataset(ds: tdata.Dataset, net: nn.Module, batch_size=32):
    """
    Project the original dataset into a lower dimensionality using the
    convolutional part of the model.

    Parameters
    ----------
    ds : tdata.Dataset
        Original dataset of shape :math:`(N, C_{in}, D_{in}, H_{in}, W_{in})`
    net : nn.Module
        Convolutional part of the CNN (:math:`f: R^(C_{in}, D_{in}, H_{in}, W_{in}) -> R^d`)
    batch_size : int
        Batch size to perform the projection

    Returns
    -------
    tdata.TensorDataset
        The projected dataset of shape :math:`(N, d)`
    """
    dataloader = tdata.DataLoader(ds, batch_size, shuffle=False)
    net.eval()
    x = list()
    y = list()
    with torch.no_grad():
        for batch in dataloader:
            bx, by = batch
            projected = net(bx)
            x.append(projected.detach())
            y.append(by.detach())
    x = torch.cat(x)
    y = torch.cat(y)
    return tdata.TensorDataset(x, y)


class HorizontalBrainFlipTransform:
    def __init__(self, p=0.5, random_state=None):
        self.p = p
        self.random_state = check_random_state(random_state)

    def __call__(self, sample):
        image, label = sample
        if self.random_state.rand() < self.p:
            image = image.flip(1)
        return image, label


class TransformedDataset(tdata.Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        if self.transform:
            return self.transform(self.dataset[item])
        else:
            return self.dataset[item]


def train(net, criterion, optimizer, train_ds, val_ds, batch_size, max_epochs, patience, verbose=False):
    """
    Train the CNN.

    Parameters
    ----------
    net
        Model to train
    criterion
        Optimization criterion
    optimizer
        Optimizer method
    train_ds
        Training dataset
    val_ds
        Validation dataset
    batch_size
        Batch size for training
    max_epochs
        Maximum number of epochs to train
    patience
        Training patience, after this many iterations without improvement training halts
    verbose
        Wether to print training information during execution

    Returns
    -------
    dict of str: ndarray
        A dictionary containing the training history (train and validation loss)
    """
    net.train()
    validation_losses = list()
    train_losses = list()
    best_validation_loss = np.inf
    epochs_without_improvement = 0
    best_parameters = deepcopy(net.state_dict())

    train_dl = tdata.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = tdata.DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    for epoch in range(max_epochs):
        net.train()
        # Run epoch for every batch from the train DataLoader
        train_loss = 0.0
        for i_batch, batch in enumerate(train_dl):
            inputs, labels = batch

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
        train_loss = train_loss / len(train_ds)
        train_losses.append(train_loss)

        # Validation step
        with torch.no_grad():
            net.eval()
            val_loss = 0.0
            for batch in val_dl:
                inputs, labels = batch
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
            val_loss = val_loss / len(val_ds)
            validation_losses.append(val_loss)
        if verbose:
            print(f'[Epoch {epoch + 1}]\tval loss: {val_loss:.4f}, train loss: {train_loss:.4f}')

        # Check for early stopping
        if val_loss < best_validation_loss:
            best_validation_loss = val_loss
            best_parameters = deepcopy(net.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            if verbose:
                print(f'{patience} epochs without improvement, restoring best '
                      f'parameters and stopping training')
            net.train(False)
            net.load_state_dict(best_parameters)
            break
    return {'val_loss': validation_losses,
            'train_loss': train_losses}


def training_components(image_shape, n_channels, kernel_size, stride, hidden_size,
                        n_classes, dropout_rate, learning_rate, classifier_type: str, device):
    """
    Obtain training components from the configuration.

    Parameters
    ----------
    image_shape
        Shape of the input images (depth, height, width)
    n_channels
        Number of channels (feature maps) of each stage of convolution
    kernel_size
        Size of convolution kernels
    stride
        Stride for the convolution operation
    hidden_size
        Size of the hidden fully connected layer
    n_classes
        Number of classes to classify into
    dropout_rate
        Rate for the dropout layers
    learning_rate
        Learning rate
    classifier_type
        Type of classifier (``'nominal'`` or ``'ordinal'``)
    device
        Device use for training (C{'cpu:x'} or C{'cuda:x'})

    Returns
    -------
    net: BrainNet
        CNN model to train
    criterion: nn.Module
        Optimization criterion
    optimizer: optim.Optimizer
        Optimizer method
    """
    if classifier_type == 'nominal':
        net = NominalNet(image_shape, n_channels, kernel_size, stride,
                         hidden_size, n_classes, dropout_rate)
        criterion = nn.CrossEntropyLoss(reduction='sum')
    elif classifier_type == 'ordinal':
        net = OrdinalNet(image_shape, n_channels, kernel_size, stride,
                         hidden_size, n_classes, dropout_rate)
        criterion = ordinal_distance_loss(n_classes, device)
    else:
        raise NotImplementedError
    net = net.to(device)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    return net, criterion, optimizer


def predictions(net: BrainNet, test_ds, batch_size=32):
    """
    Use the whole CNN to estimate the class labels from a test dataset.

    Parameters
    ----------
    net : nn.Module
        CNN model
    test_ds : tdata.Dataset
        Test dataset
    batch_size : int
        Batch size to use during testing

    Returns
    -------
    true_labels: np.ndarray
        True labels
    pred_labels: np.ndarray
        Predicted labels
    pred_probas: np.ndarray
        Predicted class probabilities
    """
    net.eval()
    test_dl = tdata.DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        true_labels = list()
        pred_labels = list()
        pred_probas = list()
        for batch in test_dl:
            inputs, tl = batch
            pl = net.predict(inputs)
            true_labels.append(tl.cpu())
            pred_labels.append(pl)
            
            pp = net.scores(inputs)
            pred_probas.append(pp)
    true_labels = np.concatenate(true_labels)
    pred_labels = np.concatenate(pred_labels)
    pred_probas = np.concatenate(pred_probas)
    return true_labels, pred_labels, pred_probas


def evaluation_metrics(ytrue, ypred, probas):
    """
    Compute all the evaluation metrics into a dictionary.
    ytrue
        True labels
    ypred
        Predicted labels
    probas
        Predicted class probabilities

    Returns
    -------
    x:
        Dictionary containing all computed metrics (as listed in C{metrics.metric_list})
    """
    ms = dict()
    for m in metrics.metric_list:
        metric_value = m(ytrue, ypred)
        ms[m.__name__] = metric_value
    for m in metrics.score_metric_list:
        metric_value = m(ytrue, probas)
        ms[m.__name__] = metric_value
    return ms


def augment_dataset(ds, sp, device, random_state):
    """
    Performs data augmentation on the dataset, according to the configuration, so that
    all classes have the same representation in the augmented dataset.

    ds
        Orignal projected dataset with shape :math:`(N, d)`
    sp
        Job statepoint (experiment configuration)
    device
        Device in which the data should be placed
    random_state
        Random state for syntetic sample generation

    Returns
    -------
    tdata.TensorDataset
        Augmented dataset with shape :math:`(N', d)`
    """
    dsx, dsy = ds.tensors[0].cpu().numpy(), ds.tensors[1].cpu().numpy()
    classes, n_per_class = np.unique(dsy, return_counts=True)
    if sp.classifier_type == 'ordinal':
        if 'ordinal_augment_beta_params' in sp:
            bp = sp.ordinal_augment_beta_params
            beta_a, beta_b = bisect3(bp.xql, bp.xqu, bp.ql, bp.qu, 1e-10)
            frontier_random_gen = lambda: random_state.beta(beta_a, beta_b)
        elif 'ordinal_augment_gamma_params' in sp:
            gp = sp.ordinal_augment_gamma_params
            frontier_random_gen = lambda: random_state.gamma(gp.shape, gp.scale)
        else:
            raise NotImplementedError
        n_to_augment = n_per_class.max() - n_per_class
        augds = list()
        for c, n in zip(classes, n_to_augment):
            if n > 0:
                augx, augy = oversample_class(dsx, dsy, c, n,
                                              sp.neighbourhood_size,
                                              frontier_random_gen)
                augds.append(tdata.TensorDataset(torch.tensor(augx, device=device),
                                                 torch.tensor(augy, device=device)))
        newds = tdata.ConcatDataset([ds] + augds)
    elif sp.classifier_type == 'nominal':
        sampling_strategy = {c: n_per_class.max() for c in classes}
        smote = SMOTE(sampling_strategy=sampling_strategy, k_neighbors=sp.neighbourhood_size,
                      random_state=random_state)
        augx, augy = smote.fit_resample(dsx, dsy)
        newds = tdata.TensorDataset(torch.tensor(augx, device=device),
                                    torch.tensor(augy, device=device))
    else:
        raise NotImplementedError(f'classifier_type="{sp.classifier_type}"')

    return newds


def seed_from_str(s: str) -> int:
    """
    Obtains an integer seed from a string using the hash function
    """
    return hash(s) % (2 ** 32)


def determinism(seed):
    """
    Uses a given seed to ensure determinism when launching a new experiment
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)


def train_parameter_combination(train_ds: tdata.Dataset, val_ds: tdata.Dataset, job, split_name: str,
                                device: torch.device, random_state):
    """
    Trains a new model according to the specified parameter combination.

    Parameters
    ----------
    train_ds
        Train dataset
    val_ds
        Validation dataset
    job
        signac job with the configuration
    split_name
        Name of the current validation split
    device
        Device in which to put the data
    random_state
        Random state for data augmentation

    Returns
    -------
    BrainNet
        The final trained network
    """
    net, criterion, optimizer = training_components(**{p: v for p, v in job.sp.items()
                                                       if p in signature(training_components).parameters},
                                                    image_shape=job.doc.image_shape, n_classes=job.doc.n_classes,
                                                    device=device)

    if job.isfile(f'trained_state_{split_name}.pt'):
        net.load_state_dict(torch.load(job.fn(f'trained_state_{split_name}.pt'), map_location=device))
        return net

    # Train full network
    history1 = train(net, criterion, optimizer, train_ds, val_ds,
                     **{p: v for p, v in job.sp.items()
                        if p in signature(train).parameters})

    with job.stores.training_data as d:
        d[f'{split_name}/train_loss/pre_augment'] = np.array(history1['train_loss'])
        d[f'{split_name}/val_loss/pre_augment'] = np.array(history1['val_loss'])

    # Project data using convolutional part
    proj_train_ds = project_dataset(train_ds, net.convnet)
    proj_val_ds = project_dataset(val_ds, net.convnet)

    # Augment data
    aug_proj_train_ds = augment_dataset(proj_train_ds, job.sp, random_state=random_state, device=device)

    # Train only fully connected part with augmented data
    optimizer.state = defaultdict(dict)  # reset optimizer state
    history2 = train(net.densenet, criterion, optimizer, aug_proj_train_ds, proj_val_ds,
                     patience=job.sp.patience_dense,
                     **{p: v for p, v in job.sp.items()
                        if p in signature(train).parameters and
                        p != 'patience'})
    with job.stores.training_data as d:
        d[f'{split_name}/train_loss/post_augment'] = np.array(history2['train_loss'])
        d[f'{split_name}/val_loss/post_augment'] = np.array(history2['val_loss'])

    torch.save(net.state_dict(), job.fn(f'trained_state_{split_name}.pt'))

    return net


def load_data(sp, split_name, device, random_state):
    """
    Load experiment data from job stores.

    Parameters
    __________
    sp
        Job statepoint
    split_name
        Name of the validation split
    device
        Device in which to place the data
    random_state
        Random state for flip augmentation

    Returns
    -------
    train_ds : tdata.Dataset
        Train dataset
    val_ds : tdata.Dataset
        Validation dataset
    test_ds : tdata.Dataset
        Test dataset
    """
    random_state = check_random_state(random_state)

    with posix_ipc.Semaphore(project_data_semaphore_name, flags=posix_ipc.O_CREAT, initial_value=1):
        with project.data:
            data = project.data
            train_index = data[f'seed{sp.seed}/fold{sp.fold}/{sp.phase}/{split_name}/train'][:]
            val_index = data[f'seed{sp.seed}/fold{sp.fold}/{sp.phase}/{split_name}/validation'][:]

            if sp.phase == 'validation':
                test_index = data[f'seed{sp.seed}/fold{sp.fold}/validation/{split_name}/test'][:]
            elif sp.phase == 'evaluation':
                test_index = data[f'seed{sp.seed}/fold{sp.fold}/evaluation/test'][:]
            else:
                raise RuntimeError

            samples = data['samples']
            targets = data['targets']
            class_mapping = dict(sp.class_mapping)
            targets = np.array([class_mapping[t] for t in targets])

            train_samples = samples[sorted(train_index)]
            train_targets = targets[sorted(train_index)]
            train_ds = tdata.TensorDataset(torch.tensor(train_samples, device=device),
                                           torch.tensor(train_targets, device=device))

            val_samples = samples[sorted(val_index)]
            val_targets = targets[sorted(val_index)]
            val_ds = tdata.TensorDataset(torch.tensor(val_samples, device=device),
                                         torch.tensor(val_targets, device=device))

            test_samples = samples[sorted(test_index)]
            test_targets = targets[sorted(test_index)]
            test_ds = tdata.TensorDataset(torch.tensor(test_samples, device=device),
                                          torch.tensor(test_targets, device=device))

            train_ds = TransformedDataset(train_ds,
                                          transform=HorizontalBrainFlipTransform(random_state=random_state)
                                          if sp.augment_flip else None)

    return train_ds, val_ds, test_ds


@FlowProject.label
def results_saved(job):
    with posix_ipc.Semaphore(project_data_semaphore_name, flags=posix_ipc.O_CREAT, initial_value=1):
        splits = set(project.data[f'seed{job.sp.seed}/fold{job.sp.fold}/{job.sp.phase}'].keys()) - {'test'}
    return all(job.isfile(f'trained_state_{split_name}.pt') for split_name in splits) and \
           job.isfile('evaluation_metrics.csv') and \
           all(f'confusion_matrix_{split_name}' in job.doc.keys() for split_name in splits) and \
           all(f'result_metrics_{split_name}' in job.doc.keys() for split_name in splits)


def get_split_names(sp):
    with posix_ipc.Semaphore(project_data_semaphore_name, flags=posix_ipc.O_CREAT, initial_value=1):
        with project.data:
            names = [n for n in project.data[f'seed{sp.seed}/fold{sp.fold}/{sp.phase}'].keys() if n.startswith('split')]
    return names


@FlowProject.operation
@directives(ngpu=1)
@FlowProject.post.isfile('evaluation_metrics.csv')
@FlowProject.post(results_saved)
def train_combination(job):
    split_names = get_split_names(job.sp)
    eval_metrics = list()
    for split_name in split_names:
        split_seed = seed_from_str(f'{job.sp.seed}_{split_name}')
        determinism(split_seed)
        random_state = check_random_state(split_seed)
        device = torch.device('cuda:0')

        train_ds, val_ds, test_ds = load_data(job.sp, split_name, device, random_state)
        net = train_parameter_combination(train_ds, val_ds, job, split_name, device, random_state)
        ytrue, ypred, probas = predictions(net, test_ds)
        cm = confusion_matrix(ytrue, ypred)
        em = evaluation_metrics(ytrue, ypred, probas)
        eval_metrics.append(em)

        job.doc[f'confusion_matrix_{split_name}'] = cm.tolist()
        job.doc[f'result_metrics_{split_name}'] = em

    eval_metrics = pd.DataFrame(eval_metrics)
    mean = eval_metrics.mean()
    stdv = eval_metrics.std()
    eval_metrics.loc['mean'] = mean
    eval_metrics.loc['stdv'] = stdv
    eval_metrics.to_csv(job.fn('evaluation_metrics.csv'))


def main():
    FlowProject().main()


if __name__ == '__main__':
    main()
