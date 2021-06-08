from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_olivetti_faces
import numpy as np

olivetti = fetch_olivetti_faces()
avg_accuracy = 0;
max_accuracy = 0;
min_accuracy = 100;

for i in range(0, 100):
    x_train, x_test, y_train, y_test = train_test_split(olivetti.data, olivetti.target, train_size = 0.7)
    
    s = svm.SVC(gamma = 0.001, C = 45)
    s.fit(x_train, y_train)
    
    res = s.predict(x_test)
    
    conf = np.zeros((40,40))
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
print("SVM 평균 정확률: ", avg_accuracy, "%")
print("SVM 최대 정확률: ", max_accuracy * 100, "%")
print("SVM 최소 정확률: ", min_accuracy * 100, "%")