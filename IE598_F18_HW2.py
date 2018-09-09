#Stephen Pretto (spretto2), HW2, 9/2018
#Code samples from Raschka book
import sklearn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import warnings
from mlxtend.plotting import plot_decision_regions
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from IPython.display import Image
from IPython.display import display
from sklearn.tree import export_graphviz
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import sklearn.metrics as metrics
import pydotplus


def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):

    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], 
                    y=X[y == cl, 1],
                    alpha=0.6, 
                    c=cmap(idx),
                    edgecolor='black',
                    marker=markers[idx], 
                    label=cl)

    # highlight test samples
    if test_idx:
        X_test, y_test = X[test_idx, :], y[test_idx]

        plt.scatter(X_test[:, 0],
                    X_test[:, 1],
                    c='',
                    alpha=1.0,
                    edgecolor='black',
                    linewidths=1,
                    marker='o',
                    s=55, label='test set')

iris = load_iris()
X = iris.data[:, [2, 3]]
y = iris.target
print('Class labels:', np.unique(y))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)


tree = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=1)


tree.fit(X_train_std, y_train)
X_combined = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))
plot_decision_regions(X_combined, y_combined, classifier=tree, test_idx=range(105, 150))
plt.xlabel('petal length [cm]')
plt.ylabel('petal width [cm]')
plt.legend(loc="upper left")
plt.show()

#K nearest neighbor KNN problem
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')
knn.fit(X_train_std, y_train)
plot_decision_regions(X_combined, y_combined, classifier=knn, test_idx=range(105,150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc="upper left")
plt.show()

#Find the optimal value of K, using a large test sisze
#We can change the value of the test size to influence the distribution of accuracy scores for k
print("K accuracy for 80% testing set")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=0)
k_range = range(1,26)
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    #Show all the graphs for each K
    #plot_decision_regions(X_test, y_test, classifier=knn)
    #plt.xlabel('petal length [standardized]')
    #plt.ylabel('petal width [standardized]')
    #plt.legend(loc="upper left")
    #plt.show()
    y_pred = knn.predict(X_test)
    scores.append((k, metrics.accuracy_score(y_test, y_pred))) #Appends a tuple as k, score
   
print(scores)

print("K accuracy for 30% testing set")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
k_range = range(1,26)
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    #Show all the graphs for each K
    #plot_decision_regions(X_test, y_test, classifier=knn)
    #plt.xlabel('petal length [standardized]')
    #plt.ylabel('petal width [standardized]')
    #plt.legend(loc="upper left")
    #plt.show()
    y_pred = knn.predict(X_test)
    scores.append((k, metrics.accuracy_score(y_test, y_pred))) #Appends a tuple as k, score
   
print(scores)
    
print("My name is Stephen Pretto")
print("My NetID is: spretto2")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")

#Extra to graph the decision tree
dot_data = export_graphviz(tree, out_file=None, feature_names=['petal length', 'petal width'],  class_names=['setosa', 'versicolor', 'virginica'], filled=True, rounded=True) 
graph = pydotplus.graph_from_dot_data(dot_data) 
display(Image(graph.create_png()))