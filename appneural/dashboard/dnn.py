# https://www.kaggle.com/wonchanleee/simple-dnn/notebook

import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

train_target = train_df['label']
train_features = train_df.drop(['label'], axis=1)

train_features = train_features/255.0

# before train_features type: DataFrame
print(train_features.shape, type(train_features))

# after train_features type: ndarray
train_features = train_features.values.reshape(-1, 28, 28, 1)
print(train_features.shape, type(train_features))

from keras.utils.np_utils import to_categorical 
train_target = to_categorical(train_target, num_classes=10)

# split train_df into train data and validation data for the fitting
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train_features, train_target, test_size=0.1,\
                                                random_state=156)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

from tensorflow import keras
import tensorflow as tf

# SELU has self-normalization property with lecun_normal in DNN
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=X_train.shape[1:]),
    keras.layers.Dense(70, activation='selu', kernel_initializer='lecun_normal'),
    keras.layers.Dense(50, activation='selu', kernel_initializer='lecun_normal'),
    keras.layers.Dense(30, activation='selu', kernel_initializer='lecun_normal'),
    keras.layers.AlphaDropout(rate=0.5),
    keras.layers.Dense(10, activation='softmax', kernel_initializer='glorot_uniform')
])

K = keras.backend

class ExponentialLearningRate(keras.callbacks.Callback):
    def __init__(self, factor):
        self.factor = factor
        self.rates = []
        self.losses = []
    def on_batch_end(self, batch, logs):
        self.rates.append(K.get_value(self.model.optimizer.lr))
        self.losses.append(logs["loss"])
        K.set_value(self.model.optimizer.lr, self.model.optimizer.lr * self.factor)

def find_learning_rate(model, X, y, epochs=1, batch_size=32, min_rate=10**-5, max_rate=10):
    init_weights = model.get_weights()
    iterations = np.math.ceil(len(X) / batch_size) * epochs
    factor = np.exp(np.log(max_rate / min_rate) / iterations)
    init_lr = K.get_value(model.optimizer.lr)
    K.set_value(model.optimizer.lr, min_rate)
    exp_lr = ExponentialLearningRate(factor)
    history = model.fit(X, y, epochs=epochs, batch_size=batch_size,
                        callbacks=[exp_lr])
    K.set_value(model.optimizer.lr, init_lr)
    model.set_weights(init_weights)
    return exp_lr.rates, exp_lr.losses

# plot the acc during testing.
def plot_lr_vs_loss(rates, losses):
    plt.plot(rates, losses)
    plt.gca().set_xscale('log')
    plt.hlines(min(losses), min(rates), max(rates))
    plt.axis([min(rates), max(rates), min(losses), (losses[0] + min(losses)) / 2])
    plt.xlabel("Learning rate")
    plt.ylabel("Loss")
    print('proper_rate:',min(rates))

batch_size=32
n_epochs=30

# class is identical so we use categorical_crossentropy
model.compile(optimizer=keras.optimizers.Nadam(lr=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])

import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import seaborn as sns


history = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=batch_size, 
                    epochs=n_epochs, callbacks=[keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5)])


plt.plot(history.history['val_accuracy'], label='validation accuracy')
plt.plot(history.history['accuracy'], label='accuracy')
plt.title('Test accuracy')
plt.xlabel("Number of Epochs")
plt.ylabel('accuracy')
plt.legend()
plt.show()


test_df = test_df/255.0
test_df = test_df.values.reshape(-1, 28, 28, 1)


results = model.predict(test_df)

results = np.argmax(results,axis = 1)
# save model 
model.save('final_try2.h5')

# load model
# global model
model = tf.keras.models.load_model('final_try.h5',compile=False)