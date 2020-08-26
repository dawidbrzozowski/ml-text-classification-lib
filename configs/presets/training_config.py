DEFAULTS = {
    "hidden_units": 32,
    "hidden_layers": 2,
    "epochs": 5,
    "batch_size": 128,
    "hidden_activation": "relu",
    "output_activation": "sigmoid",
    "trainable_embeddings": False,
    "optimizer": "adam",
    "loss": "default_loss",
    "validation_split": 0.1,
    "metrics": ['accuracy'],
    "callbacks": None,
    "lr": 0.01
}


class TrainingConfig:
    def __init__(self, params):
        self.hidden_units = params.get('hidden_units', DEFAULTS['hidden_units'])
        self.hidden_layers = params.get('hidden_layers', DEFAULTS['hidden_layers'])
        self.epochs = params.get('epochs', DEFAULTS['epochs'])
        self.batch_size = params.get('batch_size', DEFAULTS['batch_size'])
        self.hidden_activation = params.get('hidden_activation', DEFAULTS['hidden_activation'])
        self.output_activation = params.get('output_activation', DEFAULTS['output_activation'])
        self.trainable = params.get('trainable_embeddings', DEFAULTS['trainable_embeddings'])
        self.optimizer = params.get('optimizer', DEFAULTS['optimizer'])
        self.loss = params.get('loss', DEFAULTS['loss'])
        self.metrics = params.get('metrics', DEFAULTS['metrics'])
        self.callbacks = params.get('callbacks', DEFAULTS['callbacks'])
        self.validation_split = params.get('validation_split', DEFAULTS['validation_split'])
        self.lr = params.get('lr', DEFAULTS['lr'])
