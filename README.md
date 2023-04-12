# Kaggle-Titanic
Challenge: build a predictive model that answers the question: `what sorts of people were more likely to survive?` using passenger data (ie name, age, gender, socio-economic class, etc).
Data provided by Kaggle platform.

## Descripion
* Kaggle username `CitaZane`
### Cross Validation scores
* Logistic Regression       : 79.19%
* Random Forest Classifier  : 80.07%
* K Nearest Neighbour       : 79.41%
* Support Vector Classifier : 80.54%
* Gradient Boosting         : 82.68%

### GridSearch results
#### Logistic Regression
Params:  {'lg__C': 2.7825594022071245, 
        'lg__class_weight': None, 
        'lg__max_iter': 10000, 
        'lg__solver': 'lbfgs'}

Score: 80.09%

#### KNN
Params:  {'knn__n_neighbors': 5, 
        'knn__p': 1, 
        'knn__weights': 'uniform'}

Score: 86.39%

#### SVC
Params:  {'svc__C': 2.0, 
        'svc__class_weight': None, 
        'svc__degree': 1, 
        'svc__gamma': 'scale', 
        'svc__kernel': 'rbf', 
        'svc__shrinking': False}

score: 84.36%

#### RandomForest

Params(randomized):  {'n_estimators': 200, 
            'min_samples_split': 10, 
            'min_samples_leaf': 2, 
            'max_features': None, 
            'max_depth': None, 
            'criterion': 'entropy'}
Score(randomized): 90.55%

Params:  {'criterion': 'entropy', 
        'max_depth': None, 
        'max_features': None, 
        'min_samples_leaf': 1, 
        'min_samples_split': 10, 
        'n_estimators': 200}

Score: 92.01%

#### Grdient Boosting
Params(randomized):  {
        'warm_start': False, 
        'subsample': 0.7, 
        'n_estimators': 100, 
        'min_samples_split': 10, 
        'min_samples_leaf': 2, 
        'max_depth': 5, 
        'loss': 'exponential', 
        'learning_rate': 0.01, 
        'criterion': 'friedman_mse'}

Score(randomized): 85.38%

Params:  {
    'criterion': 'squared_error', 
    'learning_rate': 0.1, 
    'loss': 'exponential', 
    'max_depth': 5, 
    'min_samples_leaf': 1, 
    'min_samples_split': 2, 
    'n_estimators': 100, 
    'subsample': 0.1, 
    'warm_start': True}

Score: 86.39%

### First iteration scores
* KNN : 76.01%
* rf  : 77.51%
* svc : 77.99%

TODO -> describe the project, feature engeneering, basic conclusions and best score achieved.

## Run the project
Activate conda environment
```bash
conda env create -f environment.yml
conda activate titanic
```
Run python scripts
```bash
python3 ./scripts/main.py
```
<!-- Save currne env packages -->
<!-- conda env export --from-history > environment.yml -->