import numpy as np
import tensorflow as tf


class UNETModel:

    _model = None
    _batch_size = 1

    def __init__(self, image_shape, n_classes=1, weights=None):
        self.shape = image_shape
        self.n_classes = n_classes
        self.create_model()

        if weights is not None:
            self.load_weights(weights)

    def get_model(self):
        if self._model is None:
            self.create_model()

        return self._model

    def get_batch_size(self):
        return self._batch_size

    def set_model(self, model):
        self._model = model

    def create_model(self):
        inputs = tf.keras.layers.Input((self.shape[0], self.shape[1], self.n_classes))

        convolution_1 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same',
                                               kernel_initializer='he_normal')(inputs)
        convolution_1 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same',
                                               kernel_initializer='he_normal')(convolution_1)
        pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(convolution_1)

        convolution_2 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same',
                                               kernel_initializer='he_normal')(pool1)
        convolution_2 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same',
                                               kernel_initializer='he_normal')(convolution_2)
        pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(convolution_2)

        convolution_3 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same',
                                               kernel_initializer='he_normal')(pool2)
        convolution_3 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same',
                                               kernel_initializer='he_normal')(convolution_3)
        pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(convolution_3)

        convolution_4 = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same',
                                               kernel_initializer='he_normal')(pool3)
        convolution_4 = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same',
                                               kernel_initializer='he_normal')(convolution_4)
        drop4 = tf.keras.layers.Dropout(0.5)(convolution_4)
        pool4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(drop4)

        convolution_5 = tf.keras.layers.Conv2D(1024, 3, activation='relu', padding='same',
                                               kernel_initializer='he_normal')(pool4)
        convolution_5 = tf.keras.layers.Conv2D(1024, 3, activation='relu', padding='same',
                                               kernel_initializer='he_normal')(convolution_5)
        drop5 = tf.keras.layers.Dropout(0.5)(convolution_5)

        up6 = tf.keras.layers.Conv2D(512, 2, activation='relu', padding='same',
                                     kernel_initializer='he_normal')(
            tf.keras.layers.UpSampling2D(size=(2, 2))(drop5))
        merge6 = tf.keras.layers.concatenate([drop4, up6], axis=3)
        convolution_6 = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same',
                                               kernel_initializer='he_normal')(merge6)
        convolution_6 = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same',
                                               kernel_initializer='he_normal')(convolution_6)

        up7 = tf.keras.layers.Conv2D(256, 2, activation='relu', padding='same',
                                     kernel_initializer='he_normal')(
            tf.keras.layers.UpSampling2D(size=(2, 2))(convolution_6))
        merge7 = tf.keras.layers.concatenate([convolution_3, up7], axis=3)
        convolution_7 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same',
                                               kernel_initializer='he_normal')(merge7)
        convolution_7 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same',
                                               kernel_initializer='he_normal')(convolution_7)

        up8 = tf.keras.layers.Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            tf.keras.layers.UpSampling2D(size=(2, 2))(convolution_7))
        merge8 = tf.keras.layers.concatenate([convolution_2, up8], axis=3)
        convolution_8 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same',
                                               kernel_initializer='he_normal')(merge8)
        convolution_8 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same',
                                               kernel_initializer='he_normal')(convolution_8)

        up9 = tf.keras.layers.Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            tf.keras.layers.UpSampling2D(size=(2, 2))(convolution_8))
        merge9 = tf.keras.layers.concatenate([convolution_1, up9], axis=3)

        convolution_9 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same',
                                               kernel_initializer='he_normal')(merge9)
        convolution_9 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same',
                                               kernel_initializer='he_normal')(convolution_9)
        convolution_9 = tf.keras.layers.Conv2D(2, 3, activation='relu', padding='same',
                                               kernel_initializer='he_normal')(convolution_9)
        convolution_10 = tf.keras.layers.Conv2D(1, 1, activation='sigmoid')(convolution_9)

        model = tf.keras.models.Model(inputs=inputs, outputs=convolution_10)
        model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

        self.set_model(model)

    def load_weights(self, location):
        self.get_model().load_weights(location)

    def predict(self, images):
        images = np.expand_dims(images.astype(np.float32), axis=3)
        prediction = self.get_model().predict(images / 255)

        return prediction
