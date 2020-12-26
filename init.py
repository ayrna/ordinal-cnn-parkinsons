"""Initialize the signac project

This script loads the project data and initializes a signac project in the CWD.

You must specify the location of the source data and it will store everything in
the signac project-wide `data` attribute (H5Store using the HDF5 format for
efficient use).

It then initializes all the necessary jobs for the cross validation phase.
5 folds x 32 configurations (2*2*2*2*2) = 160 jobs for each different methodology.

4 different methodologies are tested: nominal, ordinal using a gamma distribution for
OGO-SP (shape=2, scale=0.15), and 3 different ordinal ones using a beta distribution
with the following quantile restrictions:

* P(X < 0.5) = 0.75 and P(X < 0.65) = 0.9
* P(X < 0.5) = 0.75 and P(X < 0.75) = 0.9
* P(X < 0.5) = 0.75 and P(X < 0.85) = 0.9
"""

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import signac
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, ParameterGrid, train_test_split
from sklearn.utils import check_random_state
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('base_data_path', type=Path,
                    help="Base path where the data is located. It should contain a 'labels.csv' file " \
                         "containing filenames and their associated target label, as well as a 'mni' " \
                         "directory containing images in MNI152 2mm standard space in NIfTI-1 format " \
                         "compressed in gzip or any other format supported by nibabel.")
args = parser.parse_args()

if not args.base_data_path.is_dir():
    parser.error(f'"{str(args.base_data_path)}" is not a valid directory')

data_root_path = args.base_data_path / 'mni'
labels_path = args.base_data_path / 'labels.csv'
n_folds = 5
n_validation_splits = 3
n_evaluation_splits = 10
validation_ratio = 0.1
seed = 0
random_state = check_random_state(seed)

project = signac.init_project('parkinson-cross-validation')

labels_df = pd.read_csv(str(labels_path))

if ('samples' not in project.data) or ('targets' not in project.data):
    samples_array = np.empty((len(labels_df), 1, 91, 109, 91), dtype=np.float32)
    targets_array = np.empty((len(labels_df),), dtype=int)

    print('Reading data for project into memory')
    for i, row in tqdm(enumerate(labels_df.itertuples()), total=len(labels_df)):
        nii: nib.Nifti1Image = nib.load(str(data_root_path / row.filename))
        samples_array[i, 0, :, :, :] = nii.get_fdata()
        targets_array[i] = row.target

    with project.data:
        print('Saving data into project data')
        project.data['samples'] = samples_array
        project.data['targets'] = targets_array


with project.data:
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    for fold, (train_index, test_index) in enumerate(skf.split(np.zeros((len(labels_df), 1)), labels_df.target)):
        project.data[f'seed{seed}/fold{fold}/evaluation/test'] = test_index

        random_state = check_random_state(hash(f'{seed}_{fold}_evaluation') % (2**32))
        sss = StratifiedShuffleSplit(n_splits=n_evaluation_splits, test_size=validation_ratio,
                                    random_state=random_state)
        for i, (split_train_index, split_val_index) in enumerate(sss.split(np.zeros((len(train_index), 1)),
                                                                        labels_df.target.loc[train_index])):
            subtrain_index = train_index[split_train_index]
            val_index = train_index[split_val_index]
            project.data[f'seed{seed}/fold{fold}/evaluation/split{i}/train'] = subtrain_index
            project.data[f'seed{seed}/fold{fold}/evaluation/split{i}/validation'] = val_index

        random_state = check_random_state(hash(f'{seed}_{fold}_validation') % (2**32))
        sss = StratifiedShuffleSplit(n_splits=n_validation_splits,
                                    test_size=validation_ratio, random_state=random_state)
        for i, (val_train_index, val_test_index) in enumerate(sss.split(np.zeros((len(train_index), 1)),
                                                                        labels_df.target.loc[train_index])):
            subtrain_index = train_index[val_train_index]
            subtest_index = train_index[val_test_index]
            subtrain_index, val_index = train_test_split(subtrain_index,
                                                        test_size=validation_ratio, random_state=random_state,
                                                        stratify=labels_df.target.loc[subtrain_index])
            project.data[f'seed{seed}/fold{fold}/validation/split{i}/train'] = subtrain_index
            project.data[f'seed{seed}/fold{fold}/validation/split{i}/validation'] = val_index
            project.data[f'seed{seed}/fold{fold}/validation/split{i}/test'] = subtest_index


cm4 = [(0, 0), (1, 1), (2, 2), (3, 3)]
grid = {
    'fold': list(range(n_folds)),
    'classifier_type': ['nominal', 'ordinal'],
    'learning_rate': [1e-4, 1e-3],
    'hidden_size': [2048, 4096],
    'dropout_rate': [0.0, 0.3],
    'kernel_size': [3, 5],
    'neighbourhood_size': [3, 5],
    'class_mapping': [cm4],

    # Fixed params
    'augment_flip': [True],
    'n_channels': [[1, 20, 30, 40, 50]],
    'stride': [2],
    'n_inner_splits': [3],
    'batch_size': [32],
    'validation_ratio': [0.1],
    'max_epochs': [1000],
    'patience': [50],
    'patience_dense': [50],
    'seed': [seed],
    'phase': ['validation'],
}

print('Adding jobs to project')
for p in tqdm(ParameterGrid(grid)):

    def add_job(sp):
        job = project.open_job(sp).init()
        job.doc['n_classes'] = len(np.unique(list(zip(*job.sp.class_mapping))[1]))
        job.doc['image_shape'] = (91, 109, 91)


    if p['classifier_type'] == 'ordinal':
        ordinal_gamma_configuration = {
            'ordinal_augment_gamma_params': {
                'shape': 2, 'scale': 0.15,
            }
        }
        add_job({**p, **ordinal_gamma_configuration})
        
        ordinal_beta1_configuration = {
            'ordinal_augment_beta_params': {
                'xql': 0.5, 'ql': 0.75,
                'xqu': 0.65, 'qu': 0.9,
            }
        }
        add_job({**p, **ordinal_beta1_configuration})

        ordinal_beta2_configuration = {
            'ordinal_augment_beta_params': {
                'xql': 0.5, 'ql': 0.75,
                'xqu': 0.75, 'qu': 0.9,
            }
        }
        add_job({**p, **ordinal_beta2_configuration})

        ordinal_beta3_configuration = {
            'ordinal_augment_beta_params': {
                'xql': 0.5, 'ql': 0.75,
                'xqu': 0.85, 'qu': 0.9,
            }
        }
        add_job({**p, **ordinal_beta3_configuration})
    else:
        add_job(p)
