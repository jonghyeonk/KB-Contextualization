"""
This script prepares data in the format for the testing
algorithms to run

The script is expanded to the resource attribute
"""

from __future__ import division

import copy
import csv
import pdb
import re
import time
from queue import PriorityQueue
from datetime import datetime

import numpy as np
from pathlib import Path
import shared_variables
from evaluation.server_replayer import verify_model_compliantness
from shared_variables import get_unicode_from_int


def prepare_testing_data(eventlog):
    csvfile = open(shared_variables.data_folder / f"{eventlog}.csv", 'r')
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    next(spamreader, None)  # skip the headers

    lastcase = ''
    line = ''
    line_group = ''
    first_line = True
    lines_id = []
    lines = []
    lines_group = []
    timeseqs = []  # relative time since previous event
    timeseqs2 = []  # relative time since case start
    timeseqs3 = []  # absolute time of previous event
    timeseqs4 = []  # absolute time of event as a string
    times = []
    times2 = []
    times3 = []
    times4 = []
    numlines = 0
    casestarttime = None
    lasteventtime = None

    for row in spamreader:
        t = time.strptime(row[2], "%Y-%m-%d %H:%M:%S")
        if row[0] != lastcase:
            casestarttime = t
            lasteventtime = t
            lastcase = row[0]
            if not first_line:
                lines.append(line)
                lines_group.append(line_group)
                timeseqs.append(times)
                timeseqs2.append(times2)
                timeseqs3.append(times3)
                timeseqs4.append(times4)
            lines_id.append(lastcase)
            line = ''
            line_group = ''
            times = []
            times2 = []
            times3 = []
            times4 = []
            numlines += 1
        line += get_unicode_from_int(row[1])
        line_group += get_unicode_from_int(row[3])
        timesincelastevent = datetime.fromtimestamp(time.mktime(t)) - datetime.fromtimestamp(time.mktime(lasteventtime))
        timesincecasestart = datetime.fromtimestamp(time.mktime(t)) - datetime.fromtimestamp(time.mktime(casestarttime))
        timediff = 86400 * timesincelastevent.days + timesincelastevent.seconds
        timediff2 = 86400 * timesincecasestart.days + timesincecasestart.seconds
        times.append(timediff)
        times2.append(timediff2)
        times3.append(datetime.fromtimestamp(time.mktime(t)))
        times4.append(row[2])
        lasteventtime = t
        first_line = False

    # add last case
    lines.append(line)
    lines_group.append(line_group)
    timeseqs.append(times)
    timeseqs2.append(times2)
    timeseqs3.append(times3)
    timeseqs4.append(times4)
    numlines += 1

    divisor = np.max([item for sublist in timeseqs for item in sublist])
    divisor2 = np.max([item for sublist in timeseqs2 for item in sublist])
    # divisor3 = np.max(map(lambda x: np.max(map(lambda y: x[len(x) - 1] - y, x)), timeseqs2))
    divisor3 = np.max([np.max([x[len(x) - 1] - y for y in x]) for x in timeseqs2])

    elems_per_fold = int(round(numlines / 3))

    fold1and2lines = lines[:2 * elems_per_fold]
    # fold1and2lines = map(lambda x: x + '!', fold1and2lines)
    # maxlen = max(map(lambda x: len(x), fold1and2lines))
    fold1and2lines = [x + '!' for x in fold1and2lines]
    maxlen = max([len(x) for x in fold1and2lines])
    chars = list(map(lambda x: set(x), fold1and2lines))
    chars = list(set().union(*chars))
    chars.sort()
    target_chars = copy.copy(chars)
    if '!' in chars:
        chars.remove('!')
    char_indices = dict((c, i) for i, c in enumerate(chars))
    target_char_indices = dict((c, i) for i, c in enumerate(target_chars))
    target_indices_char = dict((i, c) for i, c in enumerate(target_chars))

    fold1and2lines_group = lines_group[:2 * elems_per_fold]
    # fold1and2lines_group = map(lambda x: x + '!', fold1and2lines_group)
    chars_group = list(map(lambda x: set(x), fold1and2lines_group))
    chars_group = list(set().union(*chars_group))
    chars_group.sort()
    target_chars_group = copy.copy(chars_group)
    # chars_group.remove('!')
    char_indices_group = dict((c, i) for i, c in enumerate(chars_group))
    target_char_indices_group = dict((c, i) for i, c in enumerate(target_chars_group))
    target_indices_char_group = dict((i, c) for i, c in enumerate(target_chars_group))

    # we only need the third fold, because first two were used for training
    fold3 = lines[2 * elems_per_fold:]
    fold3_id = lines_id[2 * elems_per_fold:]
    fold3_group = lines_group[2 * elems_per_fold:]
    fold3_t = timeseqs[2 * elems_per_fold:]
    fold3_t2 = timeseqs2[2 * elems_per_fold:]
    fold3_t3 = timeseqs3[2 * elems_per_fold:]
    fold3_t4 = timeseqs4[2 * elems_per_fold:]

    lines = fold3
    lines_id = fold3_id
    lines_group = fold3_group
    lines_t = fold3_t
    lines_t2 = fold3_t2
    lines_t3 = fold3_t3
    lines_t4 = fold3_t4

    # set parameters
    predict_size = maxlen

    return lines, lines_id, lines_group, lines_t, lines_t2, lines_t3, lines_t4, maxlen, chars, chars_group, \
           char_indices, char_indices_group, divisor, divisor2, divisor3, predict_size, target_indices_char, \
           target_indices_char_group, target_char_indices, target_char_indices_group


# selects traces verified by a declare model
def select_declare_verified_traces(lines, lines_id, lines_group, lines_t, lines_t2, lines_t3, lines_t4,
                                   path_to_declare_model_file):
    # select only lines with formula verified
    lines_v = []
    lines_id_v = []
    lines_group_v = []
    lines_t_v = []
    lines_t2_v = []
    lines_t3_v = []
    lines_t4_v = []
    for line, line_id, line_group, times, times2, times3, times4 in zip(lines, lines_id, lines_group, lines_t, lines_t2,
                                                                        lines_t3, lines_t4):

        if verify_model_compliantness(path_to_declare_model_file, line_id, line, line_group, times4):
            lines_v.append(line)
            lines_id_v.append(line_id)
            lines_group_v.append(line_group)
            lines_t_v.append(times)
            lines_t2_v.append(times2)
            lines_t3_v.append(times3)
            lines_t4_v.append(times4)

    return lines_v, lines_id_v, lines_group_v, lines_t_v, lines_t2_v, lines_t3_v, lines_t4_v

"""
# selects traces verified by LTL formula
def select_formula_verified_traces(log_name, lines, lines_id, lines_group, lines_t, lines_t2, lines_t3, lines_t4,
                                   formula, prefix=0):
    # select only lines with formula verified
    lines_v = []
    lines_id_v = []
    lines_group_v = []
    lines_t_v = []
    lines_t2_v = []
    lines_t3_v = []
    lines_t4_v = []

    for line, line_id, line_group, times, times2, times3, times4 in zip(lines, lines_id, lines_group, lines_t, lines_t2,
                                                                        lines_t3, lines_t4):
        if verify_formula_as_compliant(line_id, line, log_name, prefix):
            lines_v.append(line)
            lines_id_v.append(line_id)
            lines_group_v.append(line_group)
            lines_t_v.append(times)
            lines_t2_v.append(times2)
            lines_t3_v.append(times3)
            lines_t4_v.append(times4)

    return lines_v, lines_id_v, lines_group_v, lines_t_v, lines_t2_v, lines_t3_v, lines_t4_v
"""

# define helper functions
# this one encodes the current sentence into the onehot encoding
def encode(sentence, sentence_group, times, times3, maxlen, chars, chars_group,
           char_indices, char_indices_group, divisor, divisor2):
    num_features = len(chars) + len(chars_group) + 5
    x = np.zeros((1, maxlen, num_features), dtype=np.float32)
    leftpad = maxlen - len(sentence)
    times2 = np.cumsum(times)
    for t, char in enumerate(sentence):
        midnight = times3[t].replace(hour=0, minute=0, second=0, microsecond=0)
        timesincemidnight = times3[t] - midnight
        for c in chars:
            if c == char:
                x[0, t + leftpad, char_indices[c]] = 1
        for g in chars_group:
            if g == sentence_group[t]:
                x[0, t + leftpad, len(char_indices) + char_indices_group[g]] = 1
        x[0, t + leftpad, len(chars) + len(chars_group)] = t + 1
        x[0, t + leftpad, len(chars) + len(chars_group) + 1] = times[t] / divisor
        x[0, t + leftpad, len(chars) + len(chars_group) + 2] = times2[t] / divisor2
        x[0, t + leftpad, len(chars) + len(chars_group) + 3] = timesincemidnight.seconds / 86400
        x[0, t + leftpad, len(chars) + len(chars_group) + 4] = times3[t].weekday() / 7
    return x


# modify to be able to get second best prediction
def get_symbol_ampl(predictions, target_indices_char, target_char_indices, start_of_the_cycle_symbol,
                    stop_symbol_probability_amplifier_current, ith_best=0):
    a_pred = list(predictions)
    if start_of_the_cycle_symbol in target_char_indices:
        place_of_starting_symbol = target_char_indices[start_of_the_cycle_symbol]
        a_pred[place_of_starting_symbol] = a_pred[place_of_starting_symbol] / stop_symbol_probability_amplifier_current
    i = np.argsort(a_pred)[len(a_pred) - ith_best - 1]
    return target_indices_char[i]


# modify to be able to get second best prediction
def adjust_probabilities(predictions, target_char_indices, start_of_the_cycle_symbol,
                         stop_symbol_probability_amplifier_current):
    a_pred = list(predictions)
    if start_of_the_cycle_symbol in target_char_indices:
        place_of_starting_symbol = target_char_indices[start_of_the_cycle_symbol]
        a_pred[place_of_starting_symbol] = a_pred[place_of_starting_symbol] / stop_symbol_probability_amplifier_current
    return a_pred


# find repetitions
def repetitions(s):
    r = re.compile(r"(.+?)\1+")
    for match in r.finditer(s):
        yield (match.group(1), len(match.group(0)) / len(match.group(1)))


def amplify(s):
    list_of_rep = list(repetitions(s))
    if list_of_rep:
        str_rep = list_of_rep[-1][0]
        if s.endswith(str_rep):
            return np.math.exp(list_of_rep[-1][-1]), list_of_rep[-1][0][0]
        else:
            return 1, list_of_rep[-1][0][0]
    return 1, " "


def create_queue(activites, resources):
    queue = PriorityQueue()
    # resources_standardized = standardize_list(activites, resources)
    for activity_index in range(len(activites)):
        for resource_index in range(len(resources)):
            queue.put((-(np.log(activites[activity_index]) + np.log(resources[resource_index])),
                       [activity_index, resource_index]))
    return queue


def standardize_list(list1, list2):
    len1 = float(len(list1))
    len2 = float(len(list2))
    weight = len2 / len1
    # standardized_list = map(lambda x: weight * x, list2)
    standardized_list = [weight * x for x in list2]
    return standardized_list
