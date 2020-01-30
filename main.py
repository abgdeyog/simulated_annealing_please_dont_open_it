import numpy as np
# linear algebra
import pandas as pd
from sklearn import datasets

iris = datasets.load_iris()

X = iris.data[:, :]
y = iris.target

from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
y1 = encoder.fit_transform(y)

Y = pd.get_dummies(y1).values

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0, shuffle=True)

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.python.keras.optimizers import SGD, Adam
from math import exp




model_gradient = Sequential()

model_gradient.add(Dense(10,input_shape=(4,),activation='tanh'))
model_gradient.add(Dense(8,activation='tanh'))
model_gradient.add(Dense(6,activation='tanh'))
model_gradient.add(Dense(3,activation='softmax'))

model_gradient.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model_gradient.summary()

model_gradient.fit(X_train,y_train,epochs=100)
y_pred = model_gradient.predict(X_test)

y_test_class = np.argmax(y_test, axis=1)
y_pred_class = np.argmax(y_pred, axis=1)

from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test_class,y_pred_class))
print(confusion_matrix(y_test_class,y_pred_class))


# Annealing
model_annealing = Sequential()

model_annealing.add(Dense(10, input_shape=(4,), activation='tanh'))
model_annealing.add(Dense(8, activation='tanh'))
model_annealing.add(Dense(6, activation='tanh'))
model_annealing.add(Dense(3, activation='softmax'))

model_annealing_tmp = Sequential()

model_annealing_tmp.add(Dense(10, input_shape=(4,), activation='tanh'))
model_annealing_tmp.add(Dense(8, activation='tanh'))
model_annealing_tmp.add(Dense(6, activation='tanh'))
model_annealing_tmp.add(Dense(3, activation='softmax'))

categorical_cross_entropy = CategoricalCrossentropy()

T = 1000.0
a = 0.5

rnd = np.random
import random

def g(weight, t):
    return random.gauss(weight, t)


weights = model_annealing.get_weights()
count1 = 0
count2 = 0
count3 = 0
while T > 0.5:
    for t in range(1, 10):
        # model_annealing_tmp.set_weights(np.array([g(connection_weight, t) for connection_weight in
        #                                           [neuron_weight for neuron_weight in [layer.get_weights()
        #                                                                                for layer in
        #                                                                                model_annealing.layers]]]))
        for layer, layer_tmp in zip(model_annealing.layers, model_annealing_tmp.layers):
            # print(np.array([g(weight, t) for weight in layer.get_weights()[0]]))
            # print(layer_tmp.get_weights())
            layer_tmp.set_weights([np.array([g(weight, t) for weight in layer.get_weights()[0]]),
                                   layer.get_weights()[1]])
            loss_new = categorical_cross_entropy(y_train, model_annealing_tmp.predict(X_train))
            loss_old = categorical_cross_entropy(y_train, model_annealing.predict(X_train))
            diff = loss_new - loss_old
            u = rnd.rand()
            if diff < 0:
                model_annealing.set_weights(model_annealing_tmp.get_weights())
                count1 += 1
            elif exp((0 - loss_new) / T) / exp((0 - loss_old / T)) > u:
                model_annealing.set_weights(model_annealing_tmp.get_weights())
                count2 += 1
            # layers.append(np.array(new_weights))
        # for layer in model_annealing.layers:
        #     model_annealing_tmp.set_weights(np.array([g(weight) for weight in layer.get_weights()[0]]))
      #  prediction = np.argmax(model_annealing.predict(X_train), axis=0)

        count3 += 1
    T *= a

# print(model_annealing.get_weights())
# print("----------------------------------------------")
# print(model_gradient.get_weights())
y_pred = model_annealing.predict(X_test)
#
y_test_class = np.argmax(y_test, axis=1)
y_pred_class = np.argmax(y_pred, axis=1)
print(y_pred_class)
print(y_test_class)
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.metrics import categorical_accuracy

print(model_annealing.get_weights())
# print(classification_report(y_test_class, y_pred_class))
print(confusion_matrix(y_test_class, y_pred_class))
print(categorical_accuracy(y_test_class, y_pred_class))

