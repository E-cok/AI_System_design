from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_olivetti_faces
from sklearn.neural_network import MLPClassifier
import numpy as np

olivetti = fetch_olivetti_faces()
avg_accuracy = 0;
max_accuracy = 0;
min_accuracy = 100;
repeat = 0;

for i in range(0, 50):
    x_train, x_test, y_train, y_test = train_test_split(olivetti.data, olivetti.target, train_size= 0.7)
    
    mlp = MLPClassifier(hidden_layer_sizes = (4096), learning_rate_init = 0.0001,
                        batch_size = 128, solver = 'adam', max_iter = 100, verbose = True)
    mlp.fit(x_train, y_train)
    
    res = mlp.predict(x_test)
    
    conf = np.zeros((40, 40))
    for i in range(len(res)):
        conf[res[i]][y_test[i]] += 1
    print(conf)
    
    correct = 0
    for i in range(40):
        correct += conf[i][i]
    accuracy = correct / len(res)
    print("Accuracy is : ", accuracy * 100, "%")
    
    if accuracy > max_accuracy:
        max_accuracy = accuracy
    if accuracy < min_accuracy:
        min_accuracy = accuracy
    avg_accuracy += accuracy
    repeat += 1
    print("반복횟수 : ", repeat)
print("MLP 평균 정확률: ", (avg_accuracy / 50) * 100, "%")
print("MLP 최대 정확률: ", max_accuracy * 100, "%")
print("MLP 최소 정확률: ", min_accuracy * 100, "%")