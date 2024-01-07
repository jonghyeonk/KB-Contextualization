"""
This script takes as input the workflow, timestamps and an event attribute "resource"
It makes predictions on the workflow & timestamps and the event attribute "resource"
this script trains an LSTM model on one of the data files in the data folder of
this repository. the input file can be changed to another file from the data folder
by changing its name in line 46.
it is recommended to run this script on GPU, as recurrent networks are quite
computationally intensive.
Author: Niek Tax
"""

from __future__ import print_function, division
import copy
import os
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import LSTM, Dense, Input, Dropout, BatchNormalization, GlobalMaxPooling1D
from keras.models import Model
from keras.optimizers import Nadam, Adam
from keras_nlp.layers import TransformerEncoder

from src.commons import shared_variables as shared
from src.commons.log_utils import LogData
from src.commons.utils import extract_trace_sequences
from src.training.train_common import create_checkpoints_path, plot_loss, CustomTransformer


def _build_model(max_len, num_features, target_chars, target_chars_group, models_folder, resource, outcome):
    print('Build model...')

    main_input = Input(shape=(max_len, num_features), name='main_input')

    if models_folder == "LSTM":
        processed = LSTM(50, return_sequences=True, dropout=0.2)(main_input)
        processed = BatchNormalization()(processed)

        activity_output = LSTM(50, return_sequences=False, dropout=0.2)(processed)
        activity_output = BatchNormalization()(activity_output)
        activity_output = Dense(len(target_chars), activation='softmax', name='act_output')(activity_output)

        if resource:
            group_output = LSTM(50, return_sequences=False, dropout=0.2)(processed)
            group_output = BatchNormalization()(group_output)
            group_output = Dense(len(target_chars_group), activation='softmax', name='group_output')(group_output)
        if outcome:
            outcome_output = LSTM(50, return_sequences=False, dropout=0.2)(processed)
            outcome_output = BatchNormalization()(outcome_output)
            outcome_output = Dense(1, activation='sigmoid', name='outcome_output')(outcome_output)

        opt = Nadam(learning_rate=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004, clipvalue=3)

    elif models_folder == "keras_trans":

        processed = TransformerEncoder(intermediate_dim=64, num_heads=8)(main_input)

        processed = GlobalMaxPooling1D()(processed)
        processed = Dropout(0.5)(processed)

        activity_output = Dense(len(target_chars), activation='softmax', name='act_output')(processed)
        if resource:
            group_output = Dense(len(target_chars_group), activation='softmax', name='group_output')(processed)
        if outcome:
            outcome_output = Dense(1, activation='sigmoid', name='outcome_output')(processed)

        opt = Adam()

    else:
        raise RuntimeError(f'The "{models_folder}" network is not defined!')

    if ~resource and ~outcome:
        model = Model(main_input, [activity_output])
        model.compile(loss={'act_output': 'categorical_crossentropy'}, optimizer=opt)

    elif resource and ~outcome:
        model = Model(main_input, [activity_output, group_output])
        model.compile(loss={'act_output': 'categorical_crossentropy', 'group_output': 'categorical_crossentropy'}, optimizer=opt)
        
    elif resource and outcome:
        model = Model(main_input, [activity_output, group_output, outcome_output])
        model.compile(loss={'act_output': 'categorical_crossentropy', 'group_output': 'categorical_crossentropy',
                            'outcome_output': 'binary_crossentropy'}, optimizer=opt)
        
    return model


def _train_model(model, checkpoint_name, x, y_a, y_o, y_g):
    model_checkpoint = ModelCheckpoint(checkpoint_name, save_best_only=True)
    lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=0, mode='auto',
                                   min_delta=0.0001, cooldown=0, min_lr=0)

    early_stopping = EarlyStopping(monitor='val_loss', patience=7)

    if (y_g is None) and (y_o is None) :
        history = model.fit(x, {'act_output': y_a }, validation_split=shared.validation_split,
                            batch_size=32, verbose=2, callbacks=[early_stopping, model_checkpoint, lr_reducer],
                            epochs=shared.epochs)
    elif (y_g is None) and (y_o is not None) :
        history = model.fit(x, {'act_output': y_a, 'outcome_output': y_o},
                            validation_split=shared.validation_split, verbose=2, batch_size=32,
                            callbacks=[early_stopping, model_checkpoint, lr_reducer], epochs=shared.epochs)
    elif (y_g is not None) and (y_o is not None) :
        history = model.fit(x, {'act_output': y_a, 'outcome_output': y_o, 'group_output': y_g},
                            validation_split=shared.validation_split, verbose=2, batch_size=32,
                            callbacks=[early_stopping, model_checkpoint, lr_reducer], epochs=shared.epochs)
        
    plot_loss(history, os.path.dirname(checkpoint_name))


def train(log_data: LogData, models_folder: str, resource: bool, outcome: bool):
        
    training_lines, training_lines_group, training_outcomes = extract_trace_sequences(log_data, log_data.training_trace_ids, resource, outcome)

    # Adding '!' to identify end of trace
    training_lines = [x + '!' for x in training_lines]
    if resource:
        training_lines_group = [x + '!' for x in training_lines_group]
    maxlen = max([len(x) for x in training_lines])

    # Next lines here to get all possible characters for events and annotate them with numbers
    chars = list(map(lambda x: set(x), training_lines))  # Remove duplicate activities from each separate case
    chars = list(set().union(*chars))  # Creates a list of all the unique activities in the data set
    chars.sort()  # Sorts the chars in alphabetical order

    # check_new_act = log_data.log[log_data.act_name_key].unique().tolist()
    # if "\xad" in check_new_act:
    #     check_new_act.remove("\xad")
    
    # if len(check_new_act) > len(chars):
    # chars = chars + [na for na in check_new_act if na not in chars]

    target_chars = copy.copy(chars)
    chars.remove('!')
    target_chars.sort()
    
    print(chars)
    print(f'Total chars: {len(chars)} - Target chars: {len(target_chars)}')

    char_indices = dict((c, i) for i, c in enumerate(chars))
    target_char_indices = dict((c, i) for i, c in enumerate(target_chars))

    if resource:
        chars_group = list(map(lambda x: set(x), training_lines_group))
        chars_group = list(set().union(*chars_group))
        chars_group.sort()
        
        check_new_res = log_data.log[log_data.res_name_key].unique()
        if len(check_new_res) > len(chars_group):
            chars_group = chars_group + [nr for nr in check_new_res if nr not in chars_group]
            
        target_chars_group = copy.copy(chars_group)
        chars_group.remove('!')
        print(f'Total groups: {len(chars_group)} - Target groups: {len(target_chars_group)}')

        char_indices_group = dict((c, i) for i, c in enumerate(chars_group))
        target_char_indices_group = dict((c, i) for i, c in enumerate(target_chars_group))

    step = 1
    softness = 0

    sentences = []
    sentences_group = []
    sentences_o = []
    next_chars = []
    next_chars_group = []

    if ~resource and ~outcome:
        for line in training_lines:
            for i in range(0, len(line), step):
                if i == 0:
                    continue
                # We add iteratively, first symbol of the line, then two first, three...
                sentences.append(line[0: i])
                next_chars.append(line[i])

        print('Num. of training sequences:', len(sentences))
        print('Vectorization...')
        num_features = len(chars) + 1
        print(f'Num. of features: {num_features}')

        x = np.zeros((len(sentences), maxlen, num_features), dtype=np.float32)
        y_a = np.zeros((len(sentences), len(target_chars)), dtype=np.float32)
        y_g = None
        y_o = None 
        
        for i, sentence in enumerate(sentences):
            leftpad = maxlen - len(sentence)

            for t, char in enumerate(sentence):
                for c in chars:
                    if c == char:
                        x[i, t + leftpad, char_indices[c]] = 1
                x[i, t + leftpad, len(chars)] = t + 1

            for c in target_chars:
                if c == next_chars[i]:
                    y_a[i, target_char_indices[c]] = 1 - softness
                else:
                    y_a[i, target_char_indices[c]] = softness / (len(target_chars) - 1)

        for fold in range(shared.folds):
            model = _build_model(maxlen, num_features, target_chars, None, models_folder, resource, outcome)
            checkpoint_name = create_checkpoints_path(log_data.log_name.value, models_folder, fold, 'CF')
            _train_model(model, checkpoint_name, x, y_a, y_o, y_g)

    if ~resource and outcome:
        for line, outcome in zip(training_lines, training_outcomes):
            for i in range(0, len(line), step):
                if i == 0:
                    continue
                # We add iteratively, first symbol of the line, then two first, three...
                sentences.append(line[0: i])
                sentences_o.append(outcome)

                next_chars.append(line[i])

        print('Num. of training sequences:', len(sentences))
        print('Vectorization...')
        num_features = len(chars) + 1
        print(f'Num. of features: {num_features}')

        x = np.zeros((len(sentences), maxlen, num_features), dtype=np.float32)
        y_a = np.zeros((len(sentences), len(target_chars)), dtype=np.float32)
        y_g = None
        y_o = np.zeros((len(sentences)), dtype=np.float32)

        for i, sentence in enumerate(sentences):
            leftpad = maxlen - len(sentence)

            sentence_group = sentences_group[i]
            for t, char in enumerate(sentence):
                for c in chars:
                    if c == char:
                        x[i, t + leftpad, char_indices[c]] = 1
                x[i, t + leftpad, len(chars)] = t + 1

            for c in target_chars:
                if c == next_chars[i]:
                    y_a[i, target_char_indices[c]] = 1 - softness
                else:
                    y_a[i, target_char_indices[c]] = softness / (len(target_chars) - 1)

            y_o[i] = sentences_o[i]

        for fold in range(shared.folds):
            model = _build_model(maxlen, num_features, target_chars, None, models_folder, resource, outcome)
            checkpoint_name = create_checkpoints_path(log_data.log_name.value, models_folder, fold, 'CFO')
            _train_model(model, checkpoint_name, x, y_a, y_o, y_g)

    if resource and outcome:
        for line, line_group, outcome in zip(training_lines, training_lines_group, training_outcomes):
            for i in range(0, len(line), step):
                if i == 0:
                    continue
                # We add iteratively, first symbol of the line, then two first, three...
                sentences.append(line[0: i])
                sentences_group.append(line_group[0: i])
                sentences_o.append(outcome)

                next_chars.append(line[i])
                next_chars_group.append(line_group[i])

        print('Num. of training sequences:', len(sentences))
        print('Vectorization...')
        num_features = len(chars) + len(chars_group) + 1
        print(f'Num. of features: {num_features}')

        x = np.zeros((len(sentences), maxlen, num_features), dtype=np.float32)
        y_a = np.zeros((len(sentences), len(target_chars)), dtype=np.float32)
        y_g = np.zeros((len(sentences), len(target_chars_group)), dtype=np.float32)
        y_o = np.zeros((len(sentences)), dtype=np.float32)

        for i, sentence in enumerate(sentences):
            leftpad = maxlen - len(sentence)

            sentence_group = sentences_group[i]
            for t, char in enumerate(sentence):
                for c in chars:
                    if c == char:
                        x[i, t + leftpad, char_indices[c]] = 1
                for g in chars_group:
                    if g == sentence_group[t]:
                        x[i, t + leftpad, len(chars) + char_indices_group[g]] = 1
                x[i, t + leftpad, len(chars) + len(chars_group)] = t + 1

            for c in target_chars:
                if c == next_chars[i]:
                    y_a[i, target_char_indices[c]] = 1 - softness
                else:
                    y_a[i, target_char_indices[c]] = softness / (len(target_chars) - 1)

            for g in target_chars_group:
                if g == next_chars_group[i]:
                    y_g[i, target_char_indices_group[g]] = 1 - softness
                else:
                    y_g[i, target_char_indices_group[g]] = softness / (len(target_chars_group) - 1)

            y_o[i] = sentences_o[i]

        for fold in range(shared.folds):
            model = _build_model(maxlen, num_features, target_chars, target_chars_group, models_folder, resource, outcome)
            checkpoint_name = create_checkpoints_path(log_data.log_name.value, models_folder, fold, 'CFRO')
            _train_model(model, checkpoint_name, x, y_a, y_o, y_g)
