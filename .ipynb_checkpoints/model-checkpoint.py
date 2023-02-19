from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import SGD

def create_model(hp)
    model = Sequential()
    activation=hp.Choice("activation", ["relu", "selu"])
    # 1st convolution block
    model.add(Conv2D(hp.Int("filters1", min_value=8, max_value=16, step=4), kernel_size=(3,3), padding='same', input_shape=input_shape))
    model.add(BatchNormalization(axis=batch_norm_axis))
    model.add(Activation(activation))
    model.add(MaxPooling2D(pool_size=(2,2), strides=2))
    
    # 2nd convolution block
    model.add(Conv2D(hp.Int("filters2", min_value=12, max_value=32, step=4), kernel_size=(3,3), padding='valid'))
    model.add(BatchNormalization(axis=batch_norm_axis))
    model.add(Activation(activation))
    model.add(MaxPooling2D(pool_size=(2,2), strides=2))

    # Fully connected block
    model.add(Flatten())
    model.add(Dense(hp.Int("dense_units", min_value=24, max_value=64, step=8)))
    model.add(Activation(activation))
    model.add(Dropout(dropout))

    # Output layer
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True),
                  metrics=['accuracy'])
    
    return model