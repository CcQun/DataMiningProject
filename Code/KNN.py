from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import DataSet

X, y = DataSet.dataset()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

knn = KNeighborsClassifier(n_neighbors=5, algorithm='kd_tree')

knn.fit(X_train, y_train)

print('模型的准确度:{}'.format(knn.score(X_test, y_test)))
