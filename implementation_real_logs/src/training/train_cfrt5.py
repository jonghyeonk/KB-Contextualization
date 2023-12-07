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


def _build_model(max_len, num_features, target_chars, target_chars_group, models_folder):
    print('Build model...')

    main_input = Input(shape=(max_len, num_features), name='main_input')

    if models_folder == "LSTM":
        processed = LSTM(50, return_sequences=True, dropout=0.2)(main_input)
        processed = BatchNormalization()(processed)

        activity_output = LSTM(50, return_sequences=False, dropout=0.2)(processed)
        activity_output = BatchNormalization()(activity_output)

        group_output = LSTM(50, return_sequences=False, dropout=0.2)(processed)
        group_output = BatchNormalization()(group_output)

        time_output = LSTM(50, return_sequences=False, dropout=0.2)(processed)
        time_output = BatchNormalization()(time_output)

        time2_output = LSTM(50, return_sequences=False, dropout=0.2)(processed)
        time2_output = BatchNormalization()(time2_output)

        outcome_output = LSTM(50, return_sequences=False, dropout=0.2)(processed)
        outcome_output = BatchNormalization()(outcome_output)

        activity_output = Dense(len(target_chars), activation='softmax', name='act_output')(activity_output)
        time_output = Dense(1, activation='ReLU', name='time_output')(time_output)     
        time2_output = Dense(1, activation='ReLU', name='time_output')(time2_output)    
        group_output = Dense(len(target_chars_group), activation='softmax', name='group_output')(group_output)
        outcome_output = Dense(1, activation='sigmoid', name='outcome_output')(outcome_output)
        print("###############################")
        opt = Nadam(learning_rate=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004, clipvalue=3)

    elif models_folder == "custom_trans":

        embed_dim = 256
        embeddings1 = Dense(embed_dim, activation="ReLU")(main_input)
        embeddings2 = Dense(embed_dim, activation="ReLU")(embeddings1)
        embeddings3 = Dense(embed_dim, activation="ReLU")(embeddings2)

        processed = CustomTransformer(embed_dim=embed_dim)(embeddings3)

        processed = GlobalMaxPooling1D()(processed)
        processed = Dropout(0.5)(processed)

        activity_output = Dense(len(target_chars), activation='softmax', name='act_output')(processed)
        time_output = Dense(1, activation='ReLU', name='time_output')(processed)  
        time2_output = Dense(1, activation='ReLU', name='time2_output')(processed)        
        group_output = Dense(len(target_chars_group), activation='softmax', name='group_output')(processed)
        outcome_output = Dense(1, activation='sigmoid', name='outcome_output')(processed)

        opt = Adam()

    elif models_folder == "keras_trans":

        processed = TransformerEncoder(intermediate_dim=64, num_heads=8)(main_input)

        processed = GlobalMaxPooling1D()(processed)
        processed = Dropout(0.5)(processed)

        activity_output = Dense(len(target_chars), activation='softmax', name='act_output')(processed)
        time_output = Dense(1, activation='ReLU', name='time_output')(processed)     
        time2_output = Dense(1, activation='ReLU', name='time2_output')(processed)      
        group_output = Dense(len(target_chars_group), activation='softmax', name='group_output')(processed)
        outcome_output = Dense(1, activation='sigmoid', name='outcome_output')(processed)

        opt = Adam()

    else:
        raise RuntimeError(f'The "{models_folder}" network is not defined!')

    model = Model(main_input, [activity_output, group_output, time_output, time2_output, outcome_output])
    model.compile(loss={'act_output': 'categorical_crossentropy', 'group_output': 'categorical_crossentropy',
                        'time_output': 'mean_squared_error', 'time2_output': 'mean_squared_error', 'outcome_output': 'binary_crossentropy'}, optimizer=opt)
    return model


def _train_model(model, checkpoint_name, x, y_a, y_o, y_g, y_t, y_t2, y_t3, y_t4):
    model_checkpoint = ModelCheckpoint(checkpoint_name, save_best_only=True)
    lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=0, mode='auto',
                                   min_delta=0.0001, cooldown=0, min_lr=0)

    early_stopping = EarlyStopping(monitor='val_loss', patience=7)

    history = model.fit(x, {'act_output': y_a, 'outcome_output': y_o, 'group_output': y_g, 'time_output':y_t, 'time2_output':y_t2 },
                        validation_split=shared.validation_split, verbose=2, batch_size=32,
                        callbacks=[early_stopping, model_checkpoint, lr_reducer], epochs=shared.epochs)
    plot_loss(history, os.path.dirname(checkpoint_name))


def train(log_data: LogData, models_folder: str):
    training_lines, training_lines_group, training_time_group,  training_time_group2, training_time_group3, training_time_group4, training_outcomes \
        = extract_trace_sequences(log_data, log_data.training_trace_ids)

    # Adding '!' to identify end of trace
    training_lines = [x + '!' for x in training_lines]
    training_lines_group = [x + '!' for x in training_lines_group]

    training_lines_time = training_time_group
    training_lines_time2 = training_time_group2
    training_lines_time3 = training_time_group3
    training_lines_time4 = training_time_group4
    maxlen = max([len(x) for x in training_lines])

    # Next lines here to get all possible characters for events and annotate them with numbers
    chars = list(map(lambda x: set(x), training_lines))  # Remove duplicate activities from each separate case
    chars = list(set().union(*chars))  # Creates a list of all the unique activities in the data set
    chars.sort()  # Sorts the chars in alphabetical order

    target_chars = copy.copy(chars)
    chars.remove('!')
    print(f'Total chars: {len(chars)} - Target chars: {len(target_chars)}')

    char_indices = dict((c, i) for i, c in enumerate(chars))
    target_char_indices = dict((c, i) for i, c in enumerate(target_chars))

    chars_group = list(map(lambda x: set(x), training_lines_group))
    chars_group = list(set().union(*chars_group))
    chars_group.sort()

    target_chars_group = copy.copy(chars_group)
    chars_group.remove('!')
    print(f'Total groups: {len(chars_group)} - Target groups: {len(target_chars_group)}')

    char_indices_group = dict((c, i) for i, c in enumerate(chars_group))
    target_char_indices_group = dict((c, i) for i, c in enumerate(target_chars_group))

    step = 1
    softness = 0

    sentences = []
    sentences_group = []
    sentences_time = []  # time sequences (differences between two events)
    sentences_time2 = []  
    sentences_time3 = []  
    sentences_time4 = []  
    # timeseqs2 = []  # time sequences (differences between the current and first)
    sentences_o = []
    next_chars = []
    next_chars_group = []
    next_nums_time = []
    next_nums_time2 = []
    next_nums_time3 = []
    next_nums_time4 = []

    for line, line_group, line_time, line_time2, line_time3, line_time4, outcome in zip(training_lines, training_lines_group,\
        training_lines_time, training_lines_time2, training_lines_time3, training_lines_time4, training_outcomes):
        for i in range(0, len(line), step):
            if i == 0:
                continue
            # We add iteratively, first symbol of the line, then two first, three...
            sentences.append(line[0: i])
            sentences_group.append(line_group[0:i])
            sentences_time.append(line_time[0:i])
            sentences_time2.append(line_time2[0:i])
            sentences_time3.append(line_time3[0:i])
            sentences_time4.append(line_time4[0:i])
            sentences_o.append(outcome)

            next_chars.append(line[i])
            next_chars_group.append(line_group[i])
            next_nums_time.append(line_time[i])
            next_nums_time2.append(line_time2[i])
            next_nums_time3.append(line_time3[i])
            next_nums_time4.append(line_time4[i])
            
            
    print('Num. of training sequences:', len(sentences))
    print('Vectorization...')
    num_features = len(chars) + len(chars_group) + 3
    print(f'Num. of features: {num_features}')

    x = np.zeros((len(sentences), maxlen, num_features), dtype=np.float32)
    y_a = np.zeros((len(sentences), len(target_chars)), dtype=np.float32)
    y_g = np.zeros((len(sentences), len(target_chars_group)), dtype=np.float32)
    y_t = np.zeros((len(sentences)), dtype=np.float32)
    y_t2 = np.zeros((len(sentences)), dtype=np.float32)
    y_t3 = np.zeros((len(sentences)), dtype=np.float32)
    y_t4 = np.zeros((len(sentences)), dtype=np.float32)
    y_o = np.zeros((len(sentences)), dtype=np.float32)

    for i, sentence in enumerate(sentences):
        leftpad = maxlen - len(sentence)

        sentence_group = sentences_group[i]
        sentence_time = sentences_time[i]
        sentence_time2 = sentences_time2[i]
        for t, char in enumerate(sentence):
            for c in chars:
                if c == char:
                    x[i, t + leftpad, char_indices[c]] = 1
            for g in chars_group:
                if g == sentence_group[t]:
                    x[i, t + leftpad, len(chars) + char_indices_group[g]] = 1
            for ti in sentence_time:
                x[i, t + leftpad, len(chars) + len(chars_group) ] = ti
            for ti in sentence_time2:
                x[i, t + leftpad, len(chars) + len(chars_group)+1 ] = ti
            x[i, t + leftpad, len(chars) + len(chars_group)+2] = t + 1

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

        y_t[i] = next_nums_time[i]
        y_t2[i] = next_nums_time2[i]
        y_t3[i] = next_nums_time3[i]
        y_t4[i] = next_nums_time4[i]
        y_o[i] = sentences_o[i]

    for fold in range(shared.folds):
        model = _build_model(maxlen, num_features, target_chars, target_chars_group, models_folder)
        checkpoint_name = create_checkpoints_path(log_data.log_name.value, models_folder, fold, 'CFRT5')
        _train_model(model, checkpoint_name, x, y_a, y_o, y_g, y_t, y_t2, y_t3, y_t4)
