from typing import List

from text_clsf_lib.models.build.layers_builder_utils import LAYER_BUILDERS, LayerNotFoundError

LAYER_NAMES = '\n'.join([layer_name for layer_name in LAYER_BUILDERS])


def build_layers(last_hidden_layer, layer_descriptions: List[str]):
    for layer_description in layer_descriptions:
        layer = None
        for layer_name in LAYER_BUILDERS:
            if layer_name in layer_description:
                layer = LAYER_BUILDERS[layer_name](layer_description)
                last_hidden_layer = layer(last_hidden_layer)
                break
        if layer is None:
            raise LayerNotFoundError(
                f'Layer not found. Please include in the description one of the following layers:\n{LAYER_BUILDERS}')
    return last_hidden_layer
