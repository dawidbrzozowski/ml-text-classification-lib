from typing import List

from keras import layers

ACTIVATIONS = ['relu', 'softmax', 'tanh']

LAYERS = {
    'dense': layers.Dense,
    # 'conv1d': layers.Conv1D,
    'maxpooling1d': layers.MaxPooling1D,
    'globalmaxpooling1d': layers.GlobalMaxPooling1D,
    'averagepooling1d': layers.AveragePooling1D,
    'globalaveragepooling1d': layers.GlobalAveragePooling1D,
    'lstm': layers.LSTM,
    'gru': layers.GRU,
    'bidirectional': layers.Bidirectional,
    'dropout': layers.Dropout,
    'spatialdropout1d': layers.SpatialDropout1D,
    'flatten': layers.Flatten,
}

ACTIVATION_NAMES = '\n'.join([activation_name for activation_name in ACTIVATIONS])


class ActivationNotFoundError(Exception):
    pass


def _build_maxpooling1d(descr: List[str]):
    return layers.MaxPooling1D()


def _build_averagepooling1d(descr: List[str]):
    return layers.AveragePooling1D()


def _build_globalmaxpooling1d(descr: List[str]):
    return layers.GlobalMaxPooling1D()


def _build_globalaveragepooling1d(descr: List[str]):
    return layers.GlobalMaxPooling1D()


def _build_lstm(descr: List[str]):
    if descr[1] not in ACTIVATIONS:
        raise ActivationNotFoundError(
            f'Activation with name: {descr[1]} not found. Please try one of the following:\n{ACTIVATION_NAMES}')
    units = int(descr[2])
    return_sequences = True if len(descr) == 4 and descr[3] == 'return_sequences' else False
    return layers.LSTM(activation=descr[1], units=units, return_sequences=return_sequences)


def _build_gru(descr: List[str]):
    return _build_core_layer(descr)


def _build_dense(descr: List[str]):
    return _build_core_layer(descr)


def _build_bidirectional(descr: List[str], next_layer_func, next_layer_descr):
    next_layer = next_layer_func(next_layer_descr)
    layer = LAYERS[descr[0]](next_layer)
    return layer


def _build_dropout(descr: List[str]):
    return layers.Dropout(rate=descr[1])


def _build_spatialdropout1d(descr: List[str]):
    return layers.SpatialDropout1D(rate=descr[1])


def _build_flatten(descr: List[str]):
    return layers.Flatten()


def _build_core_layer(descr: List[str]):
    if descr[1] not in ACTIVATIONS:
        raise ActivationNotFoundError(
            f'Activation with name: {descr[1]} not found. Please try one of the following:\n{ACTIVATION_NAMES}')
    units = int(descr[2])
    return LAYERS[descr[0]](activation=descr[1], units=units)
