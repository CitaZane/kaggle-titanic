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