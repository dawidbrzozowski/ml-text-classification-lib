from typing import List

from text_clsf_lib.models.layers_builder_utils import LAYERS, _build_dense, _build_averagepooling1d, \
    _build_globalaveragepooling1d, _build_globalmaxpooling1d, _build_maxpooling1d, _build_lstm, _build_gru, \
    _build_bidirectional, _build_dropout, _build_spatialdropout1d, _build_flatten

LAYER_NAMES = '\n'.join([layer_name for layer_name in LAYERS])


class LayerNotFoundError(Exception):
    pass


LAYER_BUILDERS = {
    'dense': _build_dense,
    # 'conv1d': layers.Conv1D,
    'maxpooling1d': _build_maxpooling1d,
    'globalmaxpooling1d': _build_globalmaxpooling1d,
    'averagepooling1d': _build_averagepooling1d,
    'globalaveragepooling1d': _build_globalaveragepooling1d,
    'lstm': _build_lstm,
    'gru': _build_gru,
    'bidirectional': _build_bidirectional,
    'dropout': _build_dropout,
    'spatialdropout1d': _build_spatialdropout1d,
    'flatten': _build_flatten,
}

layer_names = '\n'.join([layer_name for layer_name in LAYERS])


def build_layers(last_hidden_layer, hidden_layers_descr: List[str]):
    skip = False
    for i, hidden_layer_descr in enumerate(hidden_layers_descr):
        if not skip:
            descr = hidden_layer_descr.split()
            if descr[0] not in LAYERS:
                raise LayerNotFoundError(
                    f'Layer with name: {descr[0]} not found. Please try one of the following:\n{layer_names}')
            else:
                build_layer_func = get_build_function(descr)
                if descr[0] == 'bidirectional':
                    next_layer_func = get_build_function(hidden_layers_descr[i + 1].split())
                    layer = build_layer_func(descr, next_layer_func, hidden_layers_descr[i + 1].split())
                    skip = True
                else:
                    layer = build_layer_func(descr)
                last_hidden_layer = layer(last_hidden_layer)
        else:
            skip = False
    return last_hidden_layer


def get_build_function(descr):
    return LAYER_BUILDERS[descr[0]]
