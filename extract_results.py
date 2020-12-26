"""Extract results from jobs

Extracts the computed metrics for the finished jobs and, if all folds/splits have
been completed, generates a summary for each configuration
"""

import argparse
import textwrap

import pandas as pd
import signac

from common import get_frontier_distribution, add_filters_to_parser, get_filters_from_args
from project import results_saved

project = signac.get_project()
wrapper = textwrap.TextWrapper(initial_indent='    - ', subsequent_indent='    ')


parser = argparse.ArgumentParser()
add_filters_to_parser(parser)
args = parser.parse_args()

filter_dict = get_filters_from_args(args.filter)
assert 'phase' not in filter_dict
filter_dict['phase'] = 'evaluation'
doc_filter_dict = get_filters_from_args(args.doc_filter)

filtered_jobs = project.find_jobs(filter_dict, doc_filter_dict)
grouped_jobs = filtered_jobs.groupby(lambda job: (
    job.sp.seed,
    job.sp.classifier_type,
    job.doc.n_classes,
    get_frontier_distribution(job),
))

n_evaluation_splits = 10
for (seed, classifier_type, n_classes, frontier_distribution), group in grouped_jobs:
    print(f'Processing case: {classifier_type} classifier'
          f'{" ("+frontier_distribution+")" if frontier_distribution is not None else ""}'
          f', {n_classes} classes'
          f' (seed {seed})')
    
    jobs = sorted(list(group), key=lambda j: j.sp.fold)
    unfinished_folds = set(j.sp.fold for j in jobs if not results_saved(j))
    if unfinished_folds:
        print(wrapper.fill(f'The following folds have not been computed: {unfinished_folds}'))
        continue
    
    folds = [pd.DataFrame([j.doc[f'result_metrics_split{i}'] for i in range(n_evaluation_splits)],
                          index=pd.Index(list(range(n_evaluation_splits)), name='validation_split'))
             for j in jobs]
    full = pd.concat(folds, keys=range(5), names=['fold']).drop(columns=['mze'])
    full = full[['ccr', 'amae', 'gm', 'mae', 'mmae', 'ms', 'tkendall', 'wkappa', 'spearman']]
    print(wrapper.fill(f'Saving to xlsx file'))
    full.to_excel(f'{n_classes}classes_{classifier_type}'
                  f'{"_"+frontier_distribution if classifier_type == "ordinal" else ""}'
                  f'_seed{seed}'
                  f'_evaluation.xlsx')
