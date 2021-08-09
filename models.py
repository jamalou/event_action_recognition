import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def se_block(inputs, ch, ratio=16):
    gap_layer = layers.GlobalAveragePooling2D()
    ds_1_layer = layers.Dense(ch//ratio, activation='relu')
    ds_2_layer = layers.Dense(ch, activation='sigmoid')
    mp_layer = layers.Multiply()


    X = [gap_layer(inp) for inp in inputs]
    X = [ds_1_layer(inp) for inp in X]
    X = [ds_2_layer(inp) for inp in X]
    X = [mp_layer([inp, x]) for inp, x in zip(inputs, X)]

    return X


def cnn_multiple_images(inputs, n_filters, se_ratio=16):

    conv_1 = layers.Conv2D(n_filters, 3, activation='relu', padding='same')
    conv_2 = layers.Conv2D(n_filters, 3, activation='relu', padding='same')
    bn_ = layers.BatchNormalization()

    X = [conv_1(inp) for inp in inputs]
    X = [conv_2(inp) for inp in X]
    X = [bn_(inp) for inp in X]
    X = se_block(X, ch=n_filters, ratio=se_ratio)


    return X


def create_se_convlstm_model(n_frames=3):
	inputs = [layers.Input(shape=(260, 346, 3)) for _ in range(n_frames)]
	X = cnn_multiple_images(inputs, n_filters=16, se_ratio=16)

	#x_1, x_2, x_3 = cnn_multiple_images(x_1, x_2, x_3, n_filters=32, se_ratio=16)
	#x = layers.Concatenate()([x_1, x_2, x_3])

	X = tf.keras.backend.stack(
		X,
		axis=-1
	)

	X = layers.ConvLSTM2D(
		filters=64,
		kernel_size=(5, 5),
		padding="same",
		return_sequences=True,
		activation="relu",
	)(X)

	X = layers.BatchNormalization()(X)

	X = layers.Conv3D(
		filters=1, kernel_size=(3, 3, 3), activation="sigmoid", padding="same"
	)(X)

	X = layers.GlobalAveragePooling3D()(X)
	X = layers.Dense(units=512, activation="relu")(X)
	X = layers.Dropout(0.3)(X)

	outputs = layers.Dense(units=10, activation="softmax")(X)
	# Next, we will build the complete model and compile it."""

	model = keras.models.Model(inputs, outputs)
	return model