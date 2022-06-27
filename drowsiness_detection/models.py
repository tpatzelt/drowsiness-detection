"""Classification Models"""
import math

from sklearn.base import TransformerMixin
from sklearn.preprocessing import StandardScaler
from tensorflow import keras


class ThreeDStandardScaler(TransformerMixin):
    """Custom StandardScaler that can handle 3D input."""
    scaler = StandardScaler()

    def __init__(self, feature_axis):
        self.feature_axis = feature_axis
        super(ThreeDStandardScaler, self).__init__()

    def fit(self, X, y, **kwargs):
        self.scaler.fit(X.reshape(-1, X.shape[self.feature_axis]), **kwargs)
        return self

    def transform(self, X, **kwargs):
        return self.scaler.transform(X.reshape(-1, X.shape[self.feature_axis]), **kwargs).reshape(
            X.shape)


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
                    pooling="average",
                    dropout_rate=.2, learning_rate=.002):
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

    dense_layer = keras.layers.Dense(32, activation='relu')(pool_layer)
    dropout_layer = keras.layers.Dropout(dropout_rate)(dense_layer)
    output_layer = keras.layers.Dense(1, activation="sigmoid")(dropout_layer)

    model = keras.models.Model(inputs=input_layer, outputs=output_layer)
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    print(model.summary())
    return model

def build_lstm_model(input_shape,
                     lstm_units=128,
                     learning_rate=.001,
                     dropout_rate=0.2,
                     num_lstm_layers=2):
    input_layer = keras.layers.Input(input_shape[1:])
    prev_layer = input_layer
    return_sequences = True
    for i in range(num_lstm_layers):
        if i == num_lstm_layers-1:
            return_sequences = False
        lstm = keras.layers.LSTM(units=lstm_units, return_sequences=return_sequences)(prev_layer)
        prev_layer = lstm
    flatten = keras.layers.Flatten()(prev_layer)
    dropout = keras.layers.Dropout(dropout_rate)(flatten)

    output_layer = keras.layers.Dense(1, activation="sigmoid")(dropout)
    model = keras.models.Model(inputs=input_layer, outputs=output_layer)

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    print(model.summary())
    return model

# learning rate schedule
def step_decay(epoch):
	initial_lrate = 0.01
	drop = 0.5
	epochs_drop = 10.0
	lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
	return lrate
