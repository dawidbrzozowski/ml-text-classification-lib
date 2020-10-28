from text_clsf_lib.models.presets.presets_base import PRESETS


def create_preset(
        # meta inf parameters
        preset_base: str,
        model_name: str = None,
        model_save_dir: str = None,
        # data parameters
        data_extractor=None,
        ner_cleaning: bool = None,
        ner_converter: bool = None,
        twitter_preprocessing: bool = None,
        output_verification_func=None,
        # vectorization parameters
        vector_width: int = None,
        preprocessor_save_dir: str = None,
        # architecture_parameters
        hidden_layers: int = None,
        hidden_layers_list: list = None,
        hidden_units: int = None,
        hidden_activation: str = None,
        output_activation: str = None,
        optimizer: str = None,
        loss: str = None,
        lr: float = None,
        metrics: list = None,
        output_units: str = None,
        # training params
        epochs: int = None,
        batch_size: int = None,
        validation_split: float = None,
        callbacks: list = None):

    preset = PRESETS[preset_base]
    _put_or_default(preset, model_name, '', 'model_name')
    _put_or_default(preset, model_save_dir, '', 'model_save_dir')
    model_save_dir = model_save_dir if model_save_dir is not None else preset['model_save_dir']
    _put_or_default(preset, f'{model_save_dir}/{model_name}', '', 'model_save_dir')
    _put_or_default(preset, data_extractor, 'data_params', 'data_extractor')
    _put_or_default(preset, ner_cleaning, 'data_params:cleaning_params:text', 'use_ner')
    _put_or_default(preset, ner_converter, 'data_params:cleaning_params:text', 'use_ner_converter')
    _put_or_default(preset, twitter_preprocessing, 'data_params:cleaning_params:text', 'use_twitter_data_preprocessing')
    _put_or_default(preset, output_verification_func, 'data_params:cleaning_params:output', 'output_verification_func')
    _put_or_default(preset, vector_width, 'vectorizer_params', 'vector_width')
    _put_or_default(preset, preprocessor_save_dir, 'vectorizer_params', 'save_dir')
    preprocessor_save_dir = preprocessor_save_dir if preprocessor_save_dir is not None else preset['vectorizer_params']['save_dir']
    _put_or_default(preset, f'{model_save_dir}/{model_name}/{preprocessor_save_dir}', 'vectorizer_params', 'save_dir')
    _put_or_default(preset, hidden_layers, 'architecture_params', 'hidden_layers')
    _put_or_default(preset, hidden_layers_list, 'architecture_params', 'hidden_layers_list')
    _put_or_default(preset, hidden_units, 'architecture_params', 'hidden_units')
    _put_or_default(preset, hidden_activation, 'architecture_params', 'hidden_activation')
    _put_or_default(preset, output_activation, 'architecture_params', 'output_activation')
    _put_or_default(preset, optimizer, 'architecture_params', 'optimizer')
    _put_or_default(preset, loss, 'architecture_params', 'loss')
    _put_or_default(preset, lr, 'architecture_params', 'lr')
    _put_or_default(preset, metrics, 'architecture_params', 'metrics')
    _put_or_default(preset, output_units, 'architecture_params', 'output_units')
    _put_or_default(preset, epochs, 'training_params', 'epochs')
    _put_or_default(preset, batch_size, 'training_params', 'batch_size')
    _put_or_default(preset, validation_split, 'training_params', 'validation_split')
    _put_or_default(preset, callbacks, 'training_params', 'callbacks')

    return preset


def _put_or_default(preset: dict, value, context_path: str, attribute_name: str):
    if value is None:
        return
    dict_path_list = context_path.split(':')
    context = preset
    for el in dict_path_list:
        if el:
            context = context.get(el)
    if attribute_name in context.keys():
        context[attribute_name] = value
