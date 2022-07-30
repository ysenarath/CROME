from collections import defaultdict
from datetime import datetime

import numpy as np

from crome.preprocessing.regions import GridArray


def to_reports(y_true, y_pred, t_index, columns, builder, y_score=None):
    """converts predictions and true labels to reports.

    :param y_true:
    :param y_pred:
    :param t_index:
    :param columns:
    :param builder:
    :param y_score:
    :return:
    """
    pred_incidents_loc = np.argwhere(y_pred > 0)
    incidents_spat = defaultdict(set)  # spacial indexed
    incident_reports = defaultdict(set)  # temporal indexed
    # output incident time is total minutes from 2019/08/01
    t2 = datetime(2019, 8, 1)
    for acc_idx, loc_idx, (loc_x, loc_y) in \
            zip(pred_incidents_loc[:, 0], pred_incidents_loc[:, 1], columns[pred_incidents_loc[:, 1]]):
        t1 = t_index.reset_index(drop=True)[acc_idx]
        p = 1.0
        if y_score is not None:
            p = float(y_score[acc_idx][loc_idx])
        t_diff = int((t1 - t2).total_seconds() / 60.0)
        incidents_spat[loc_x, loc_y].add((t_diff, 'pred'))
        incident_reports[t_diff].add((builder.grids.reverse_search(loc_x, loc_y), 'pred', p))
    for t1, y_xys, y_lng, y_lat in y_true:
        t_diff = int((t1 - t2).total_seconds() / 60.0)
        for loc_x, loc_y in y_xys:
            incidents_spat[loc_x, loc_y].add((t_diff, 'true'))
        for loc_x, loc_y in zip(y_lng, y_lat):
            incident_reports[t_diff].add(((loc_x, loc_y), 'true', 1.0))
    return incident_reports


def report_scorer(reports, threshold_t=(-30, 30), threshold_s=5000):
    """Calculate the scores table provided reports predictions.

    :param reports:
    :param threshold_t:
    :param threshold_s:
    :return:
    """
    scores_test_dist = {
        # pred
        'pred.true_positives': 0,
        'pred.early_pred': 0,
        'pred.early_pred.dist': 0,  # avg dist
        'pred.early_pred.time': 0,  # avg time
        'pred.late_pred': 0,
        'pred.false_positives': 0,
        # true incident focused matrices
        'true.true_positives': 0,
        'true.early_pred.dist': 0,  # avg dist
        'true.early_pred.time': 0,  # avg time
        'true.early_pred': 0,
        'true.late_pred': 0,
        'true.false_negative': 0,
    }
    for report_time in reports:
        records = []
        pred_reports = []
        for t in reports.keys():
            if (t + threshold_t[0] <= report_time) and (report_time <= t + threshold_t[1]):
                for report_loc, report_type, _ in reports[t]:
                    if report_type == 'true':
                        records.append((t, report_loc))
                    else:
                        pred_reports.append((t, report_loc))
        for report_loc, report_type, _ in reports[report_time]:
            if report_type == 'pred':
                has_incident = False
                is_early = False
                early_delta_d, early_delta_t = [], []
                for true_time, true_loc in records:
                    pred_proj_x, pred_proj_y = GridArray.proj.transform(*report_loc)
                    true_proj_x, true_proj_y = GridArray.proj.transform(*true_loc)
                    dist = ((pred_proj_x - true_proj_x) ** 2 + (pred_proj_y - true_proj_y) ** 2) ** 0.5
                    if dist <= threshold_s:
                        has_incident = True
                        if true_time > report_time:
                            is_early = True
                            early_delta_d.append(dist)
                            early_delta_t.append(true_time - report_time)
                if len(early_delta_d) != 0:
                    scores_test_dist['pred.early_pred.dist'] += sum(early_delta_d) / len(early_delta_d)
                if len(early_delta_t) != 0:
                    scores_test_dist['pred.early_pred.time'] += sum(early_delta_t) / len(early_delta_t)
                if has_incident:
                    scores_test_dist['pred.true_positives'] += 1
                    if is_early:
                        scores_test_dist['pred.early_pred'] += 1
                    else:
                        scores_test_dist['pred.late_pred'] += 1
                else:
                    scores_test_dist['pred.false_positives'] += 1
            else:
                has_pred_report = False
                is_early = False
                early_delta_d, early_delta_t = [], []
                for pred_time, pred_loc in pred_reports:
                    pred_proj_x, pred_proj_y = GridArray.proj.transform(*pred_loc)
                    true_proj_x, true_proj_y = GridArray.proj.transform(*report_loc)
                    dist = ((pred_proj_x - true_proj_x) ** 2 + (pred_proj_y - true_proj_y) ** 2) ** 0.5
                    if dist <= threshold_s:
                        has_pred_report = True
                        if report_time > pred_time:
                            is_early = True
                            early_delta_d.append(dist)
                            early_delta_t.append(report_time - pred_time)
                if len(early_delta_d) != 0:
                    scores_test_dist['true.early_pred.dist'] += sum(early_delta_d) / len(early_delta_d)
                if len(early_delta_t) != 0:
                    scores_test_dist['true.early_pred.time'] += sum(early_delta_t) / len(early_delta_t)
                if has_pred_report:
                    scores_test_dist['true.true_positives'] += 1
                    if is_early:
                        scores_test_dist['true.early_pred'] += 1
                    else:
                        scores_test_dist['true.late_pred'] += 1
                else:
                    scores_test_dist['true.false_negative'] += 1
    return scores_test_dist
