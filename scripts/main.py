import preprocess as pre
import model_selection as ms

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.svm import SVC

# get training daa
data = pre.get_data('./data/train.csv')
X, y = pre.preprocess(data)

#cross validate 5 models
cv_scores = ms.cross_validate_models(X,y)
print("______ Cross Validation Scores ______")
print(cv_scores)

# Find best SVC model based on GridSearch
print("\n________  SVC Best Model Params  ________")
svc = Pipeline([('scaler', StandardScaler()),
               ('svc',SVC())])
svc_params = {'svc__C': [0.1, 1, 10, 100, 1000],
            'svc__kernel':['rbf','sigmoid'],
            'svc__degree':[1,3,5],
            'svc__gamma':[1, 0.1, 0.01, 0.001, 0.0001],
            'svc__class_weight':['balanced', None]}
svc_updated_score,svc_best_model = ms.grid_search(svc,svc_params,X,y)

print("\n________  SVC Accuracy Score  ________")
print(svc_updated_score)

# get test data
data = pre.get_data('./data/test.csv')
X_test, _ = pre.preprocess(data,True)

#check for missing coumns
X_test = pre.find_missing_cols(X_test,X)

#predict on the model
y_pred = svc_best_model.predict(X_test)

# save prediction for sumbission
temp = pd.DataFrame(pd.read_csv("./data/test.csv")['PassengerId'])
temp['Survived'] = y_pred
temp.to_csv("./solutions/submission_svc.csv", index = False)
print("\n_____________   DONE  ____________")