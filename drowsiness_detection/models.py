"""Classification Models for drowsiness detection."""

from sklearn.base import TransformerMixin
from sklearn.preprocessing import StandardScaler
from tensorflow import keras


class ThreeDStandardScaler(TransformerMixin):
    """Custom StandardScaler that can handle 3D input.
    
    Reshapes 3D array to 2D for scaling along a specified feature axis.
    
    Attributes:
        feature_axis: Axis along which to compute scaling statistics
        scaler: Underlying StandardScaler instance
    """
    scaler = StandardScaler()

    def __init__(self, feature_axis: int):
        """Initialize ThreeDStandardScaler.
        
        Args:
            feature_axis: Axis to standardize along (e.g., 2 for features in shape (batch, time, features))
        """
        self.feature_axis = feature_axis
        super(ThreeDStandardScaler, self).__init__()

    def fit(self, X, y=None, **kwargs):
        """Fit scaler on 3D data.
        
        Args:
            X: 3D array of shape (n_samples, n_timesteps, n_features)
            y: Ignored, for sklearn compatibility
            **kwargs: Additional arguments for StandardScaler
            
        Returns:
            self
        """
        self.scaler.fit(X.reshape(-1, X.shape[self.feature_axis]), **kwargs)
        return self

    def transform(self, X, **kwargs):
        """Transform 3D data using fitted scaler.
        
        Args:
            X: 3D array to transform
            **kwargs: Additional arguments for StandardScaler
            
        Returns:
            Transformed 3D array with same shape as input
        """
        return self.scaler.transform(X.reshape(-1, X.shape[self.feature_axis]), **kwargs).reshape(
            X.shape)


def build_dummy_tf_classifier(input_shape: tuple, activation: str = None, optimizer: str = "adam"):
    """Build a non-trainable keras classifier for baseline comparison.
    
    Args:
        input_shape: Shape of input data including batch dimension
        activation: Activation function ('softmax', 'relu', or None)
        optimizer: Optimizer name
        
    Returns:
        Compiled Keras model
    """
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
    return model


def build_dense_model(input_shape: tuple, num_hidden: int = 64, optimizer: str = "adam"):
    """Build a simple dense (fully connected) neural network.
    
    Args:
        input_shape: Shape of input data including batch dimension
        num_hidden: Number of hidden units
        optimizer: Optimizer name
        
    Returns:
        Compiled Keras model
    """
    model = keras.Sequential()
    model.add(keras.layers.InputLayer(input_shape=input_shape[1:]))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(num_hidden, activation='relu'))
    model.add(keras.layers.Dense(1, activation='softmax'))
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    print(model.summary())
    return model


def build_cnn_model(input_shape: tuple,
                    kernel_size: int = 5,
                    stride: int = 1,
                    num_filters: int = 32,
                    num_conv_layers: int = 2,
                    padding: str = "same",
                    use_batch_norm: bool = True,
                    pooling: str = "average",
                    dropout_rate: float = 0.2,
                    learning_rate: float = 0.002):
    """Build a 1D Convolutional Neural Network for time series classification.
    
    Architecture:
        Input → Conv1D(s) → [BatchNorm → ReLU](×num_conv_layers) → 
        GlobalPooling → Dense(32, relu) → Dropout → Output(sigmoid)
    
    Args:
        input_shape: Shape of input data including batch dimension
        kernel_size: Size of convolutional kernels
        stride: Stride for convolutions
        num_filters: Number of filters in convolution layers
        num_conv_layers: Number of convolutional layers
        padding: Padding mode ('same' or 'valid')
        use_batch_norm: Whether to use batch normalization
        pooling: Pooling strategy ('average', 'max', or None)
        dropout_rate: Dropout rate after dense layer
        learning_rate: Learning rate for Adam optimizer
        
    Returns:
        Compiled Keras model
    """
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


def build_lstm_model(input_shape: tuple,
                     lstm_units: int = 128,
                     learning_rate: float = 0.001,
                     dropout_rate: float = 0.2,
                     num_lstm_layers: int = 2):
    """Build an LSTM (Long Short-Term Memory) network.
    
    Architecture:
        Input → LSTM(s, return_sequences=True) → ... → LSTM(..., return_sequences=False) → 
        Flatten → Dropout → Output(sigmoid)
    
    Args:
        input_shape: Shape of input data including batch dimension
        lstm_units: Number of units in LSTM layers
        learning_rate: Learning rate for Adam optimizer
        dropout_rate: Dropout rate after flattening
        num_lstm_layers: Number of LSTM layers
        
    Returns:
        Compiled Keras model
    """
    input_layer = keras.layers.Input(input_shape[1:])
    prev_layer = input_layer
    return_sequences = True
    for i in range(num_lstm_layers):
        if i == num_lstm_layers - 1:
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


def build_bi_lstm_model(input_shape: tuple,
                        lstm_units: int = 128,
                        learning_rate: float = 0.001,
                        dropout_rate: float = 0.2,
                        num_lstm_layers: int = 2):
    """Build a Bidirectional LSTM network.
    
    Bidirectional LSTMs process sequences in both forward and backward directions,
    allowing better context understanding for sequence classification.
    
    Architecture:
        Input → BiLSTM(s, return_sequences=True) → ... → BiLSTM(..., return_sequences=False) → 
        Flatten → Dropout → Output(sigmoid)
    
    Args:
        input_shape: Shape of input data including batch dimension
        lstm_units: Number of units in each LSTM direction
        learning_rate: Learning rate for Adam optimizer
        dropout_rate: Dropout rate after flattening
        num_lstm_layers: Number of bidirectional LSTM layers
        
    Returns:
        Compiled Keras model
    """
    input_layer = keras.layers.Input(input_shape[1:])
    prev_layer = input_layer
    return_sequences = True
    for i in range(num_lstm_layers):
        if i == num_lstm_layers - 1:
            return_sequences = False
        lstm = keras.layers.Bidirectional(
            keras.layers.LSTM(units=lstm_units, return_sequences=return_sequences))(prev_layer)
        prev_layer = lstm
    flatten = keras.layers.Flatten()(prev_layer)
    dropout = keras.layers.Dropout(dropout_rate)(flatten)

    output_layer = keras.layers.Dense(1, activation="sigmoid")(dropout)
    model = keras.models.Model(inputs=input_layer, outputs=output_layer)

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    print(model.summary())
    return model
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
        if i == num_lstm_layers - 1:
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


def build_bi_lstm_model(input_shape,
                        lstm_units=128,
                        learning_rate=.001,
                        dropout_rate=0.2,
                        num_lstm_layers=2):
    input_layer = keras.layers.Input(input_shape[1:])
    prev_layer = input_layer
    return_sequences = True
    for i in range(num_lstm_layers):
        if i == num_lstm_layers - 1:
            return_sequences = False
        lstm = keras.layers.Bidirectional(
            keras.layers.LSTM(units=lstm_units, return_sequences=return_sequences))(prev_layer)
        prev_layer = lstm
    flatten = keras.layers.Flatten()(prev_layer)
    dropout = keras.layers.Dropout(dropout_rate)(flatten)

    output_layer = keras.layers.Dense(1, activation="sigmoid")(dropout)
    model = keras.models.Model(inputs=input_layer, outputs=output_layer)

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    print(model.summary())
    return model
