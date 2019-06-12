from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def rf(x_train, y_train,x_validate, y_validate):
    clf = RandomForestClassifier(n_estimators=100, max_depth=2,random_state = 0)
    #print(len(x_train))
    clf.fit(x_train, y_train)
    predict = clf.predict(x_validate)
    accu = accuracy_score(y_validate,predict)
    return accu, clf

def rf_best(clf,x_test,y_test):
    predict = clf.predict(x_test)
    accu = accuracy_score(y_test, predict)
    return accu