from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
import pandas as pd

# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

def cross_validate_models(X_train, y_train):
    rf = RandomForestClassifier()
    knn = Pipeline([('scaler', StandardScaler()),
                    ('knn', KNeighborsClassifier())])
    svm = Pipeline([('scaler', StandardScaler()),
                    ('svc', SVC())])
    lr = Pipeline([('scaler', StandardScaler()),
                   ('lg', LogisticRegression(max_iter=10000))])
    gb = GradientBoostingClassifier()

    rf_scores = cross_val_score(rf, X_train, y_train, cv=5)
    knn_scores = cross_val_score(knn, X_train, y_train, cv=5)
    svm_scores = cross_val_score(svm, X_train, y_train, cv=5)
    lr_scores = cross_val_score(lr, X_train, y_train, cv=5)
    gb_scores = cross_val_score(gb, X_train, y_train, cv=5)

    # Print the mean cross-validation score for each model
    # print("Logistic Regression:", lr_scores.mean())
    # print("Random Forest:", rf_scores.mean())
    # print("KNN:", knn_scores.mean())
    # print("SVM:", svm_scores.mean())
    # print("Gradient Boosting:", gb_scores.mean())
    return pd.Series(data={'lg':lr_scores.mean(), 'rf':rf_scores.mean(), 'knn': knn_scores.mean(), 'svc':svm_scores.mean(), 'gb':gb_scores.mean()})

def grid_search(mod,params, X,y):
    model = GridSearchCV(mod, cv=5, n_jobs=-1, param_grid=params)
    model.fit(X,y)
    print("Params: ",model.best_params_)
    return model.score(X,y), model.best_estimator_

def randomize_search(mod,params,X,y):
    model = RandomizedSearchCV(mod,cv=5,n_iter=100,n_jobs=-1,param_distributions=params)
    model.fit(X,y)
    print("Params: ",model.best_params_)
    return model.score(X,y)