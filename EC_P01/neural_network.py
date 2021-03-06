#Projeto 01, Evolutinary Computing
#Neural NetWork Code
#Srta Camelo

#----------Imports ----------------------
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model, Model
import numpy as np

# fix random seed for reproducibility
np.random.seed(7)

def neural_network(pesos):
    np.random.seed(7)
    #keras_layer.set_weights([np.ones((1, 1, 3, 1))])
    #keras_layer.set_weights([pesos)
    # create model
    model = Sequential()
    model.add(Dense(2, input_dim=2005, activation='relu'))
    #model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    pesos = np.array(pesos)
    aux = [0,0]
    aux = np.array(aux)
    #model.get_layer["dense_1"].set_weights((np.array(pesos), aux))
    model.set_weights((np.array(pesos), aux))

    return model
def use_network(model,x_train, y_train, x_test, y_test):
    #Compile
    # mean_squared_error
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    #Train
    model.fit(x_train, y_train, epochs=150, batch_size=10, verbose= 0)
    #Test
    scores = model.evaluate(x_test, y_test,verbose= 0)
    return scores[1]

