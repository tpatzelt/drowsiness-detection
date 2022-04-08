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
    if isinstance(input_shape, str):
        input_shape = tuple([int(x) for x in input_shape.split(",")])
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
    if isinstance(input_shape, str):
        input_shape = tuple([int(x) for x in input_shape.split(",")])
    # create model
    model = keras.Sequential()
    model.add(keras.layers.InputLayer(input_shape=input_shape[1:]))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(num_hidden, activation='relu'))
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    # model.summary()
    return model


def build_cnn_model(input_shape,
                    kernel_size=(5, 1),
                    stride=(3, 1),
                    num_filters=32,
                    padding="valid",
                    use_batch_norm=True,
                    lr=.001):
    input_layer = keras.layers.Input(input_shape[1:])
    reshape_layer = keras.layers.Reshape(input_shape[1:] + (1,))(input_layer)

    conv1 = keras.layers.Conv2D(filters=num_filters, kernel_size=kernel_size, strides=stride,
                                padding=padding)(reshape_layer)
    if use_batch_norm:
        conv1 = keras.layers.BatchNormalization()(conv1)
    conv1 = keras.layers.ReLU()(conv1)

    conv2 = keras.layers.Conv2D(filters=num_filters, kernel_size=kernel_size, padding=padding)(
        conv1)
    if use_batch_norm:
        conv2 = keras.layers.BatchNormalization()(conv2)
    conv2 = keras.layers.ReLU()(conv2)

    conv3 = keras.layers.Conv2D(filters=num_filters, kernel_size=kernel_size, padding=padding)(
        conv2)
    if use_batch_norm:
        conv3 = keras.layers.BatchNormalization()(conv3)
    conv3 = keras.layers.ReLU()(conv3)

    gap = keras.layers.GlobalAveragePooling2D()(conv3)

    output_layer = keras.layers.Dense(1, activation="sigmoid")(gap)
    model = keras.models.Model(inputs=input_layer, outputs=output_layer)
    # Compile model
    # callbacks = [
    # keras.callbacks.ReduceLROnPlateau(
    #     monitor="val_loss", factor=0.5, patience=20, min_lr=0.0001
    # ),
    # keras.callbacks.EarlyStopping(monitor="val_loss", patience=50, verbose=1),
    # ]
    optimizer = keras.optimizers.Adam(learning_rate=lr)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model.summary()
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
    return model
