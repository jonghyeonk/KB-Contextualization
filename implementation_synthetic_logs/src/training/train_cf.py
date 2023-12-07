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
import os
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import LSTM, Dense, Input, BatchNormalization, Dropout, GlobalMaxPooling1D
from keras.models import Model
from keras.optimizers import Nadam, Adam
from keras_nlp.layers import TransformerEncoder

from src.commons import shared_variables as shared
from src.commons.log_utils import LogData
from src.commons.utils import extract_trace_sequences
from src.training.train_common import create_checkpoints_path, plot_loss, CustomTransformer


def _build_model(max_len, num_features, target_chars, models_folder):
    print('Build model...')

    main_input = Input(shape=(max_len, num_features), name='main_input')

    if models_folder == "LSTM":
        processed = LSTM(50, return_sequences=True, dropout=0.2)(main_input)
        processed = BatchNormalization()(processed)

        activity_output = LSTM(50, return_sequences=False, dropout=0.2)(processed)
        activity_output = BatchNormalization()(activity_output)

        # time_output = LSTM(50, return_sequences=False, dropout=0.2)(processed)
        # time_output = BatchNormalization()(time_output)

        # outcome_output = LSTM(50, return_sequences=False, dropout=0.2)(processed)
        # outcome_output = BatchNormalization()(outcome_output)

        activity_output = Dense(len(target_chars), activation='softmax', name='act_output')(activity_output)
        # time_output = Dense(1, activation='ReLU', name='time_output')(time_output)
        # outcome_output = Dense(1, activation='sigmoid', name='outcome_output')(outcome_output)

        opt = Nadam(learning_rate=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004,
                    clipvalue=3)

    elif models_folder == "custom_trans":

        embed_dim = 256
        embeddings1 = Dense(embed_dim, activation="ReLU")(main_input)
        embeddings2 = Dense(embed_dim, activation="ReLU")(embeddings1)
        embeddings3 = Dense(embed_dim, activation="ReLU")(embeddings2)

        processed = CustomTransformer(embed_dim=embed_dim)(embeddings3)

        processed = GlobalMaxPooling1D()(processed)
        processed = Dropout(0.5)(processed)

        activity_output = Dense(len(target_chars), activation='softmax', name='act_output')(processed)
        # time_output = Dense(1, activation='ReLU', name='time_output')(time_output)
        # outcome_output = Dense(1, activation='sigmoid', name='outcome_output')(processed)

        opt = Adam()

    elif models_folder == "keras_trans":

        processed = TransformerEncoder(intermediate_dim=64, num_heads=8)(main_input)

        processed = GlobalMaxPooling1D()(processed)
        processed = Dropout(0.5)(processed)

        activity_output = Dense(len(target_chars), activation='softmax', name='act_output')(processed)
        # time_output = Dense(1, activation='ReLU', name='time_output')(time_output)
        # outcome_output = Dense(1, activation='sigmoid', name='outcome_output')(processed)

        opt = Adam()

    else:
        raise RuntimeError(f'The "{models_folder}" network is not defined!')

    model = Model(main_input, [activity_output])
    model.compile(loss={'act_output': 'categorical_crossentropy'},
                  optimizer=opt)
    return model


def _train_model(model, checkpoint_name, x, y_a):
    model_checkpoint = ModelCheckpoint(checkpoint_name, save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_loss', patience=7)
    lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=0, mode='auto',
                                   min_delta=0.0001, cooldown=0, min_lr=0)
    history = model.fit(x, {'act_output': y_a}, validation_split=shared.validation_split,
                        batch_size=32, verbose=2, callbacks=[early_stopping, model_checkpoint, lr_reducer],
                        epochs=shared.epochs)
    plot_loss(history, os.path.dirname(checkpoint_name))


def train(log_data: LogData, models_folder: str):
    training_lines= extract_trace_sequences(log_data, log_data.training_trace_ids)

    # Adding '!' to identify end of trace
    training_lines = [x + '!' for x in training_lines]
    maxlen = max([len(x) for x in training_lines])

    # Next lines here to get all possible characters for events and annotate them with numbers
    chars = list(map(lambda x: set(x), training_lines))     # Remove duplicate activities from each separate case
    chars = list(set().union(*chars))   # Creates a list of all the unique activities in the data set
    chars.sort()    # Sorts the chars in alphabetical order

    target_chars = copy.copy(chars)
    chars.remove('!')
    target_chars.sort()
    
    print(f'Total chars: {len(chars)} - Target chars: {len(target_chars)}')

    char_indices = dict((c, i) for i, c in enumerate(chars))
    target_char_indices = dict((c, i) for i, c in enumerate(target_chars))

    step = 1
    softness = 0

    sentences = []
    next_chars = []
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

    for i, sentence in enumerate(sentences):
        leftpad = maxlen - len(sentence)
        for t, char in enumerate(sentence):
            for c in chars:
                if c == char:  # this will encode present events to the right places
                    x[i, t + leftpad, char_indices[c]] = 1
            x[i, t + leftpad, len(chars)] = t + 1

        for c in target_chars:
            if c == next_chars[i]:
                y_a[i, target_char_indices[c]] = 1 - softness
            else:
                y_a[i, target_char_indices[c]] = softness / (len(target_chars) - 1)


    for fold in range(shared.folds):
        model = _build_model(maxlen, num_features, target_chars, models_folder)
        checkpoint_name = create_checkpoints_path(log_data.log_name.value, models_folder, fold, 'CF')
        _train_model(model, checkpoint_name, x, y_a)
