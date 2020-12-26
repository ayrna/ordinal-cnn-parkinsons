"""Adding evaluation jobs to project

Once all configurations have been tested, next the best ranked configuration is selected for
the evaluation phase.

For each fold, the best configuration is selected and it is tested with 10 different
cross-validation splits.

A filter can be added to only add the evaluation jobs for certain methodologies, using the
same parameters as the ones used in signac CLI interface.
"""

import argparse
import textwrap

import numpy as np
import signac
from common import get_frontier_distribution, add_filters_to_parser, get_filters_from_args

from project import results_saved

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--pretend', action='store_true', dest='pretend')
add_filters_to_parser(parser)
args = parser.parse_args()

project = signac.get_project()
ranking_metric, ranking_type = 'mae', 'minimize'


filter_dict = get_filters_from_args(args.filter)
assert 'phase' not in filter_dict
filter_dict['phase'] = 'validation'
doc_filter_dict = get_filters_from_args(args.doc_filter)

filtered_jobs = project.find_jobs(filter_dict, doc_filter_dict)
grouped_jobs = filtered_jobs.groupby(lambda job: (
    job.sp.seed,
    job.sp.classifier_type,
    job.doc.n_classes,
    job.sp.fold,
    get_frontier_distribution(job)
))

for (seed, classifier_type, n_classes, fold, frontier_distribution), group in grouped_jobs:
    print(f'Querying combination: {classifier_type} classifier'
          f'{"("+frontier_distribution+")" if frontier_distribution else ""}'
          f', {n_classes} classes, fold {fold}')
    wrapper = textwrap.TextWrapper(initial_indent='    - ', subsequent_indent='    ')
    jobs = list(group)
    saved = [results_saved(j) for j in jobs]
    if all(saved):
        print(wrapper.fill('All jobs completed, determining best configuration...'))

        def mean_metric(job):
            results = [v[ranking_metric] for k, v in job.doc.items() if k.startswith('result_metrics_')]
            return np.mean(results)

        ranked_jobs = sorted(list(zip(jobs, map(mean_metric, jobs))),
                             key=lambda e: e[1], reverse=(ranking_type == 'maximize'))
        best_configuration_job = ranked_jobs[0][0]
        best_configuration = dict(best_configuration_job.sp).copy()
        best_configuration['phase'] = 'evaluation'
        print(wrapper.fill(f'Best configuration is: {best_configuration}'))
        if best_configuration not in (j.sp for j in jobs):
            print(wrapper.fill('Adding configuration to project jobs'))
            if not args.pretend:
                new_job = project.open_job(best_configuration).init()
                new_job.doc['n_classes'] = best_configuration_job.doc.n_classes
                new_job.doc['image_shape'] = best_configuration_job.doc.image_shape
            else:
                print(wrapper.fill('(pretending)'))

        else:
            print(wrapper.fill('Configuration already added'))
            evaluated = all(results_saved(j) for j in project.find_jobs(best_configuration))
            if evaluated:
                print(wrapper.fill('All evaluation completed'))
            else:
                print(wrapper.fill('Evaluation NOT completed'))
    else:
        print(wrapper.fill(f'Skipping, missing {len(saved) - sum(saved)} out of {len(saved)} jobs'))
