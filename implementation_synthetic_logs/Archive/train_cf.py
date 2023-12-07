"""
this script trains an LSTM model on one of the data files in the data folder of
this repository. the input file can be changed to another file from the data folder
by changing its name in line 46.

it is recommended to run this script on GPU, as recurrent networks are quite
computationally intensive.

Author: Niek Tax
"""

from __future__ import print_function, division
import copy
import csv
import os
import time
from datetime import datetime
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import LSTM, Dense, Input, BatchNormalization, LeakyReLU, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Nadam, Adam
import shared_variables
from shared_variables import get_unicode_from_int, folds, epochs, validation_split
from training.train_common import create_checkpoints_path, plot_loss


class TrainCF:
    def __init__(self):
        pass

    @staticmethod
    def _build_model(max_len, num_features, target_chars, use_old_model):
        print('Build model...')

        main_input = Input(shape=(max_len, num_features), name='main_input')
        processed = main_input

        if use_old_model:
            processed = LSTM(50, return_sequences=True, dropout=0.2)(processed)
            processed = BatchNormalization()(processed)

            activity_output = LSTM(50, return_sequences=False, dropout=0.2)(processed)
            activity_output = BatchNormalization()(activity_output)

            time_output = LSTM(50, return_sequences=False, dropout=0.2)(processed)
            time_output = BatchNormalization()(time_output)

            activity_output = Dense(len(target_chars), activation='softmax', name='act_output')(activity_output)
            time_output = Dense(1, name='time_output')(time_output)

            opt = Nadam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004, clipvalue=3)
        else:
            processed = Dense(32//2)(processed)
            processed = BatchNormalization()(processed)
            processed = LeakyReLU()(processed)
            processed = Dropout(0.5)(processed)

            processed = LSTM(64//2, return_sequences=False, recurrent_dropout=0.5)(processed)

            processed = Dense(32//2)(processed)
            processed = BatchNormalization()(processed)
            processed = LeakyReLU()(processed)
            processed = Dropout(0.5)(processed)

            activity_output = Dense(len(target_chars), activation='softmax', name='act_output')(processed)
            time_output = Dense(1, name='time_output')(processed)

            opt = Adam()

        model = Model(main_input, [activity_output, time_output])
        model.compile(loss={'act_output': 'categorical_crossentropy', 'time_output': 'mae'}, optimizer=opt)
        return model

    @staticmethod
    def _train_model(model, checkpoint_name, X, y_a, y_t):
        model_checkpoint = ModelCheckpoint(checkpoint_name, save_best_only=True)
        early_stopping = EarlyStopping(monitor='val_loss', patience=7)
        lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=0, mode='auto',
                                       min_delta=0.0001, cooldown=0, min_lr=0)
        history = model.fit(X, {'act_output': y_a, 'time_output': y_t}, validation_split=validation_split, batch_size=32,
                            verbose=2, callbacks=[early_stopping, model_checkpoint, lr_reducer], epochs=epochs, shuffle=True)
        plot_loss(history, os.path.dirname(checkpoint_name))

    @staticmethod
    def _load_dataset(log_name):
        dataset = []
        current_case = []
        current_case_id = None
        current_case_start_time = None
        last_event_time = None

        csvfile = open(shared_variables.data_folder / f"{log_name}.csv", 'r')
        csv_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        next(csv_reader, None)

        for row in csv_reader:  # CaseID, ActivityID, Timestamp, ResourceID
            case_id = row[0]
            activity_id = int(row[1])
            timestamp = datetime.strptime(row[2], '%Y-%m-%d %H:%M:%S')

            if case_id != current_case_id:
                if len(current_case) > 0:
                    dataset.append(current_case)
                current_case = []
                current_case_id = case_id
                current_case_start_time = timestamp
                last_event_time = timestamp

            time_since_case_start = int((timestamp - current_case_start_time).total_seconds())
            time_since_last_event = int((timestamp - last_event_time).total_seconds())
            time_since_midnight = int(
                (timestamp - timestamp.replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds())
            day_of_week = timestamp.weekday()

            current_case.append(
                (activity_id, time_since_case_start, time_since_last_event, time_since_midnight, day_of_week))

        print(len(dataset))

    @staticmethod
    def train(log_name, models_folder, use_old_model):
        # TrainCF._load_dataset(log_name)

        lines = []  # list of all the activity sequences
        timeseqs = []  # time sequences (differences between two events)
        timeseqs2 = []  # time sequences (differences between the current and first)

        # helper variables
        last_case = ''
        line = ''  # sequence of activities for one case
        first_line = True
        times = []
        times2 = []
        num_lines = 0
        case_start_time = None
        last_event_time = None

        path = shared_variables.data_folder / f"{log_name}.csv"
        print(f"Path to log: {path}")
        csvfile = open(path, 'r')
        csv_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        next(csv_reader, None)  # skip the headers

        for row in csv_reader:  # the rows are "CaseID,ActivityID,CompleteTimestamp"
            t = time.strptime(row[2], "%Y-%m-%d %H:%M:%S")  # creates a datetime object from row[2]
            if row[0] != last_case:  # 'last_case' is to save the last executed case for the loop
                case_start_time = t
                last_event_time = t
                last_case = row[0]
                if not first_line:  # here we actually add the sequences to the lists
                    lines.append(line)
                    timeseqs.append(times)
                    timeseqs2.append(times2)
                line = ''
                times = []
                times2 = []
                num_lines += 1
            line += get_unicode_from_int(row[1])
            time_since_last_event = datetime.fromtimestamp(time.mktime(t)) - datetime.fromtimestamp(
                time.mktime(last_event_time))
            time_since_case_start = datetime.fromtimestamp(time.mktime(t)) - datetime.fromtimestamp(
                time.mktime(case_start_time))
            time_diff = 86400 * time_since_last_event.days + time_since_last_event.seconds
            time_diff2 = 86400 * time_since_case_start.days + time_since_case_start.seconds
            times.append(time_diff)
            times2.append(time_diff2)
            last_event_time = t
            first_line = False

        # add last case
        lines.append(line)
        timeseqs.append(times)
        timeseqs2.append(times2)
        num_lines += 1

        divisor = 1.0 * np.max([item for sublist in timeseqs for item in sublist])  # average time between events
        print('divisor: {}'.format(divisor))
        divisor2 = 1.0 * np.max([item for sublist in timeseqs2 for item in sublist])  # average time between current and
        # first events
        print('divisor2: {}'.format(divisor2))

        # separate training data into 2(out of 3) parts
        elements_per_fold = int(round(num_lines / 3))

        many = 0
        for i in range(len(lines)):
            many = many + len(lines[i])

        print("average length of the trace: ", many / len(lines))
        print("number of traces: ", len(lines))

        fold1 = lines[:elements_per_fold]
        fold2 = lines[elements_per_fold:2 * elements_per_fold]
        lines = fold1 + fold2

        # lines = map(lambda x: x + '!', lines)  # put delimiter symbol
        # maxlen = max(map(lambda x: len(x), lines))  # find maximum line size
        lines = [x + '!' for x in lines]
        maxlen = max([len(x) for x in lines])

        # next lines here to get all possible characters for events and annotate them with numbers
        chars = list(map(lambda x: set(x), lines))  # remove duplicate activities from each separate case
        chars = list(set().union(*chars))  # creates a list of all the unique activities in the data set
        chars.sort()  # sorts the chars in alphabetical order
        target_chars = copy.copy(chars)
        # if '!' in chars:
        chars.remove('!')
        print('total chars: {}, target chars: {}'.format(len(chars), len(target_chars)))
        char_indices = dict((c, i) for i, c in enumerate(chars))
        target_char_indices = dict((c, i) for i, c in enumerate(target_chars))

        csvfile = open(shared_variables.data_folder / f"{log_name}.csv", 'r')
        csv_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        next(csv_reader, None)  # skip the headers
        last_case = ''
        line = ''
        first_line = True
        lines = []
        timeseqs = []
        timeseqs2 = []
        timeseqs3 = []
        timeseqs4 = []
        times = []
        times2 = []
        times3 = []
        times4 = []
        num_lines = 0
        case_start_time = None
        last_event_time = None
        for row in csv_reader:
            t = time.strptime(row[2], "%Y-%m-%d %H:%M:%S")
            # new case starts
            if row[0] != last_case:
                case_start_time = t
                last_event_time = t
                last_case = row[0]
                if not first_line:
                    lines.append(line)
                    timeseqs.append(times)
                    timeseqs2.append(times2)
                    timeseqs3.append(times3)
                    timeseqs4.append(times4)
                line = ''
                times = []
                times2 = []
                times3 = []
                times4 = []
                num_lines += 1
            line += get_unicode_from_int(row[1])
            time_since_last_event = datetime.fromtimestamp(time.mktime(t)) - datetime.fromtimestamp(
                time.mktime(last_event_time))
            time_since_case_start = datetime.fromtimestamp(time.mktime(t)) - datetime.fromtimestamp(
                time.mktime(case_start_time))
            midnight = datetime.fromtimestamp(time.mktime(t)).replace(hour=0, minute=0, second=0, microsecond=0)
            timesincemidnight = datetime.fromtimestamp(time.mktime(t)) - midnight
            time_diff = 86400 * time_since_last_event.days + time_since_last_event.seconds
            time_diff2 = 86400 * time_since_case_start.days + time_since_case_start.seconds
            timediff3 = timesincemidnight.seconds  # this leaves only time even occurred after midnight
            timediff4 = datetime.fromtimestamp(time.mktime(t)).weekday()  # day of the week
            times.append(time_diff)
            times2.append(time_diff2)
            times3.append(timediff3)
            times4.append(timediff4)
            last_event_time = t
            first_line = False

        # add last case
        lines.append(line)
        timeseqs.append(times)
        timeseqs2.append(times2)
        timeseqs3.append(times3)
        timeseqs4.append(times4)
        num_lines += 1

        elements_per_fold = int(round(num_lines / 3))

        lines = lines[:-elements_per_fold]
        lines_t = timeseqs[:-elements_per_fold]
        lines_t2 = timeseqs2[:-elements_per_fold]
        lines_t3 = timeseqs3[:-elements_per_fold]
        lines_t4 = timeseqs4[:-elements_per_fold]

        step = 1
        sentences = []
        softness = 0
        next_chars = []
        # lines = map(lambda x: x + '!', lines)
        lines = [x + '!' for x in lines]

        sentences_t = []
        sentences_t2 = []
        sentences_t3 = []
        sentences_t4 = []
        next_chars_t = []
        for line, line_t, line_t2, line_t3, line_t4 in zip(lines, lines_t, lines_t2, lines_t3, lines_t4):
            for i in range(0, len(line), step):
                if i == 0:
                    continue

                # we add iteratively, first symbol of the line, then two first, three...
                sentences.append(line[0: i])
                sentences_t.append(line_t[0:i])
                sentences_t2.append(line_t2[0:i])
                sentences_t3.append(line_t3[0:i])
                sentences_t4.append(line_t4[0:i])
                next_chars.append(line[i])
                if i == len(line) - 1:  # special case to deal time of end character
                    next_chars_t.append(0)
                else:
                    next_chars_t.append(line_t[i])
        print('nb sequences:', len(sentences))
        print('Vectorization...')
        num_features = len(chars) + 5
        print('num features: {}'.format(num_features))
        X = np.zeros((len(sentences), maxlen, num_features), dtype=np.float32)
        y_a = np.zeros((len(sentences), len(target_chars)), dtype=np.float32)
        y_t = np.zeros((len(sentences)), dtype=np.float32)
        for i, sentence in enumerate(sentences):
            leftpad = maxlen - len(sentence)
            next_t = next_chars_t[i]
            sentence_t = sentences_t[i]
            sentence_t2 = sentences_t2[i]
            sentence_t3 = sentences_t3[i]
            sentence_t4 = sentences_t4[i]
            for t, char in enumerate(sentence):
                # multiset_abstraction = Counter(sentence[:t+1])
                for c in chars:
                    if c == char:  # this will encode present events to the right places
                        X[i, t + leftpad, char_indices[c]] = 1
                X[i, t + leftpad, len(chars)] = t + 1
                X[i, t + leftpad, len(chars) + 1] = sentence_t[t] / divisor
                X[i, t + leftpad, len(chars) + 2] = sentence_t2[t] / divisor2
                X[i, t + leftpad, len(chars) + 3] = sentence_t3[t] / 86400
                X[i, t + leftpad, len(chars) + 4] = sentence_t4[t] / 7
            for c in target_chars:
                if c == next_chars[i]:
                    y_a[i, target_char_indices[c]] = 1 - softness
                else:
                    y_a[i, target_char_indices[c]] = softness / (len(target_chars) - 1)
            y_t[i] = next_t / divisor

        for fold in range(folds):
            # model = build_model(max_length, num_features, max_activity_id)
            model = TrainCF._build_model(maxlen, num_features, target_chars, use_old_model)
            checkpoint_name = create_checkpoints_path(log_name, models_folder, fold, 'CF')
            TrainCF._train_model(model, checkpoint_name, X, y_a, y_t)
