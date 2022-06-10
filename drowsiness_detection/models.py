"""Classification Models"""
from sklearn.base import TransformerMixin
from sklearn.preprocessing import StandardScaler
from tensorflow import keras


class ThreeDStandardScaler(TransformerMixin):
    """Custom StandardScaler that can handle 3D input."""
    scaler = StandardScaler()

    def fit(self, X, y, **kwargs):
        self.scaler.fit(X.reshape(-1, X.shape[-1]), **kwargs)
        return self

    def transform(self, X, **kwargs):
        return self.scaler.transform(X.reshape(-1, X.shape[-1]), **kwargs).reshape(X.shape)


def build_dummy_tf_classifier(input_shape, activation: str = None, optimizer: str = "adam"):
    """Build a non-trainable keras classifier."""
    input_layer = keras.layers.Input(input_shape[1:])
    flatten_layer = keras.layers.Flatten()(input_layer)
    output_layer = keras.layers.Dense(1, trainable=False)(flatten_layer)
    if activation == "softmax":
        output_layer = keras.layers.Softmax()(output_layer)
    elif activation == "relu":
        output_layer = keras.layers.ReLU()(output_layer)
    else:
        print(f"Activation was set to {activation} which is not supported.")

    model = keras.models.Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    # model.summary()
    return model


def build_dense_model(input_shape, num_hidden: int = 64, optimizer: str = "adam"):
    # create model
    model = keras.Sequential()
    model.add(keras.layers.InputLayer(input_shape=input_shape[1:]))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(num_hidden, activation='relu'))
    model.add(keras.layers.Dense(1, activation='softmax'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    print(model.summary())
    return model


def build_cnn_model(input_shape,
                    kernel_size=5,
                    stride=1,
                    num_filters=32,
                    num_conv_layers=2,
                    padding="same",
                    use_batch_norm=True,
                    lr=0.1,
                    pooling="average"):
    input_layer = keras.layers.Input(input_shape[1:])
    prev_layer = input_layer
    for _ in range(num_conv_layers):
        conv_layer = keras.layers.Conv1D(filters=num_filters, kernel_size=kernel_size,
                                         strides=stride,
                                         padding=padding)(prev_layer)
        if use_batch_norm:
            conv_layer = keras.layers.BatchNormalization()(conv_layer)
        conv_layer = keras.layers.ReLU()(conv_layer)
        prev_layer = conv_layer

    if pooling == "average":
        pool_layer = keras.layers.GlobalAveragePooling1D()(prev_layer)
    elif pooling == "max":
        pool_layer = keras.layers.GlobalMaxPool1D()(prev_layer)
    else:
        pool_layer = prev_layer

    output_layer = keras.layers.Dense(1, activation="sigmoid")(pool_layer)

    model = keras.models.Model(inputs=input_layer, outputs=output_layer)
    optimizer = keras.optimizers.Adam(learning_rate=lr)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    print(model.summary())
    return model

def build_lstm_model(input_shape,
                     lstm1_units=128,
                     lstm2_units=64,
                     dropout=.2,
                     lr=.001):
    input_layer = keras.layers.Input(input_shape[1:])
    lstm1 = keras.layers.LSTM(units=lstm1_units, return_sequences=True)(input_layer)
    lstm2 = keras.layers.LSTM(units=lstm2_units, return_sequences=True)(lstm1)
    # gap = keras.layers.AveragePooling1D()(lstm2)
    flatten = keras.layers.Flatten()(lstm2)
    dropout = keras.layers.Dropout(dropout)(flatten)

    output_layer = keras.layers.Dense(1, activation="sigmoid")(dropout)
    model = keras.models.Model(inputs=input_layer, outputs=output_layer)

    optimizer = keras.optimizers.Adam(learning_rate=lr)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    print(model.summary())
    return model
