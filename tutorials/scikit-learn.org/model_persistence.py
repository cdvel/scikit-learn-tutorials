print 'USING PICKLE'
from sklearn import svm
from sklearn import datasets
clf = svm.SVC()
iris = datasets.load_iris()
X, y = iris.data, iris.target
print clf.fit(X, y)  
#SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, degree=3, gamma=0.0,
#  kernel='rbf', max_iter=-1, probability=False, random_state=None,
#  shrinking=True, tol=0.001, verbose=False)

import pickle
s = pickle.dumps(clf)
clf2 = pickle.loads(s)
print clf2.predict(X[0])
#array([0])
print y[0]
#0

print 'USING JOBLIB'
from sklearn.externals import joblib
joblib.dump(clf, 'iris_classifier/irisclf.pkl') #individual arrays in objects; must be complete
clf = joblib.load('iris_classifier/irisclf.pkl')  #saves and load where terminal left off
print clf