from keras.layers import Input, Embedding, Flatten, Dense, Bidirectional, LSTM, GlobalMaxPooling1D
from keras import Model
from keras.optimizers import Adam

from configs.presets.training_config import TrainingConfig


class ModelPreparer:
    def __init__(self, model_config, vectorization_metainf):
        model_config = model_config.get('model')
        training_params: dict = model_config.get('training_params')
        self.training_config = TrainingConfig(params=training_params)
        self.vectorization_metainf = vectorization_metainf
        architecture_type = model_config.get('model_type')
        self.model = self.prepare_model_arch(architecture_type)

    def prepare_model_arch(self, architecture_type: str):
        if architecture_type == 'FF':
            return self.prepare_feedforward_model_arch()
        elif architecture_type == 'RNN':
            return self.prepare_rnn_arch()
        else:
            print(f'Architecture for {architecture_type} not found.')

    def prepare_feedforward_model_arch(self):
        input_ = None
        hidden = None
        if self.vectorization_metainf['type'] == 'embedding':
            input_ = Input(shape=(self.vectorization_metainf['max_seq_len'],))
            emb_layer = self.get_embedding_layer()(input_)
            hidden = Flatten()(emb_layer)
        elif self.vectorization_metainf['type'] == 'tfidf':
            input_ = Input(shape=(self.vectorization_metainf['max_features'],))
            hidden = input_
        for _ in range(self.training_config.hidden_layers):
            hidden = self.get_hidden_dense()(hidden)
        output_layer = Dense(1, activation=self.training_config.output_activation)(hidden)
        model = Model(input_, output_layer)
        optimizer = Adam(lr=self.training_config.lr) if self.training_config.optimizer == 'adam' else Adam()  # TODO
        model.compile(
            optimizer=optimizer,
            loss=self.training_config.loss,
            metrics=self.training_config.metrics
        )
        model.summary()
        return model

    def prepare_rnn_arch(self):
        if self.vectorization_metainf['type'] == 'embedding':
            input_ = Input(shape=(self.vectorization_metainf['max_seq_len'],))
            emb_layer = self.get_embedding_layer()(input_)
            hidden = emb_layer
            hidden = Bidirectional(LSTM(15, return_sequences=True))(hidden)
            hidden = GlobalMaxPooling1D()(hidden)
            output = Dense(1, activation=self.training_config.output_activation)(hidden)
            model = Model(input_, output)
            model.compile(
                optimizer=Adam(lr=0.01),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            return model

    def get_embedding_layer(self):
        embedding_matrix = self.vectorization_metainf['embedding_matrix']
        embedding_layer = Embedding(
            input_dim=len(embedding_matrix),
            output_dim=self.vectorization_metainf['embedding_dim'],
            weights=[embedding_matrix],
            input_length=self.vectorization_metainf['max_seq_len'],
            trainable=self.training_config.trainable
        )
        return embedding_layer

    def get_hidden_dense(self):
        return Dense(units=self.training_config.hidden_units,
                     activation=self.training_config.hidden_activation)
