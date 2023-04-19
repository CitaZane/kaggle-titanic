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

Based on data simplicity, looks  models like GradientBoosting and RandomForestClassifier overfits the data even whithout adjusting hyperparameters.
Further working with KNN and SVC models, SVC seems to have achived a bit better results.
Best score achieved : **0,79186** with **SVC**

### Feature Engeneering
* **Family Size** : calculated by summing up Parch and SibSp + 1. Family size better represents the family sittuation. After calculating sum family size binne and LabelEncoded. Sibling and partner count values later dropped.
* **Deck** : extracted first letter from cabin name. Further inspecting the Deck and pasanger class corrolation Decks also grouped in bigger groupes. For passanges that does not have cabin data used deck **N**. Cabin value dropped.
* **Age** : Age refactored to create equal size buckets and LabelEncoded.
* **Fare** : refactored to create equal size buckets and LabelEncoded.
* **Titles** : Using provided passanger Names extracted titles. Titles grouped/mapped in larger groupes based on regional differences. For mapping main [resource](https://www.kaggle.com/code/konstantinmasich/titanic-0-82-0-83/notebook)

* Sex, Embarked, Deck and Tile encoded using OneHotEncoder.

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
<!-- Save currnet env packages -->
<!-- conda env export --from-history > environment.yml -->
