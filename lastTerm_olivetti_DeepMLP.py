from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_olivetti_faces
import tensorflow as tf

olivetti = fetch_olivetti_faces()
avg_accuracy = 0;
max_accuracy = 0;
min_accuracy = 100;

for i in range(0, 100):
    x_train, x_test, y_train, y_test = train_test_split(olivetti.data, olivetti.target)
    x_train = x_train.reshape(300, 4096)
    x_test = x_test.reshape(100, 4096)
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    y_train = tf.keras.utils.to_categorical(y_train, 40)
    y_test = tf.keras.utils.to_categorical(y_test, 40)
    
    n_input = 4096
    n_hidden1 = 2048
    n_hidden2 = 2048
    n_hidden3 = 2048
    n_hidden4 = 1024
    n_output = 40
    
    mlp = Sequential()
    mlp.add(Dense(units = n_hidden1, activation = 'tanh',
                  input_shape = (n_input,), kernel_initializer = 'random_uniform', bias_initializer = 'zeros'))
    mlp.add(Dense(units = n_hidden2, activation = 'tanh', kernel_initializer = 'random_uniform', bias_initializer = 'zeros'))
    mlp.add(Dense(units = n_hidden3, activation = 'tanh', kernel_initializer = 'random_uniform', bias_initializer = 'zeros'))
    mlp.add(Dense(units = n_hidden4, activation = 'tanh', kernel_initializer = 'random_uniform', bias_initializer = 'zeros'))
    mlp.add(Dense(units = n_output, activation = 'tanh', kernel_initializer = 'random_uniform', bias_initializer = 'zeros'))
    
    mlp.compile(loss = 'mse', optimizer = Adam(learning_rate = 0.0001), metrics = ['accuracy'])
    hist = mlp.fit(x_train, y_train, batch_size = 64, epochs = 100, validation_data = (x_test, y_test), verbose = 2)
    
    res = mlp.evaluate(x_test, y_test, verbose = 0)
    print("Accuracy is", res[1] * 100)
    
    if (res[1] * 100) > max_accuracy:
        max_accuracy = (res[1] * 100)
    if (res[1] * 100) < min_accuracy:
        min_accuracy = (res[1] * 100)
    avg_accuracy += (res[1] * 100)
print("DeepMLP 평균 정확률: ", (avg_accuracy / 100), "%")
print("DeepMLP 최대 정확률: ", max_accuracy, "%")
print("DeepMLP 최소 정확률: ", min_accuracy, "%")