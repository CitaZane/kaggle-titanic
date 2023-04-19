# Kaggle-Titanic
Challenge: build a predictive model that answers the question: `what sorts of people were more likely to survive?` using passenger data (ie name, age, gender, socio-economic class, etc).
Data provided by Kaggle platform.

## Descripion
* Kaggle username `CitaZane`
* for easy audit check [kaggle leaderboard](https://www.kaggle.com/competitions/titanic/leaderboard?search=CitaZane) if the score is at least **0.79**
* otherwise choose one of the methods for running the project

### If you have 

### File structure
- /notebooks
        - EDA : explanatory  data analysis
        - model_selction : find/trai/predict using notebook environment
- /scripts
        - preprocess : funcs for data cleaning and feature engeneering
        - model_selcection: funcs for helping with selecting model
        - main : find/trai/predict using python script (SVC model)

- /solutions : hold solution file after running script or notebook,
this file is uploded in kaggle for evaluation.
- /data : holds train and test files from kaggle-titanic

### Feature Engeneering
* **Family Size** : calculated by summing up Parch and SibSp + 1. Family size better represents the family sittuation. After calculating sum family size binne and LabelEncoded. Sibling and partner count values later dropped.
* **Deck** : extracted first letter from cabin name. Further inspecting the Deck and pasanger class corrolation Decks also grouped in bigger groupes. For passanges that does not have cabin data used deck **N**. Cabin value dropped.
* **Age** : Age refactored to create equal size buckets and LabelEncoded.
* **Fare** : refactored to create equal size buckets and LabelEncoded.
* **Titles** : Using provided passanger Names extracted titles. Titles grouped/mapped in larger groupes based on regional differences. For mapping main [resource](https://www.kaggle.com/code/konstantinmasich/titanic-0-82-0-83/notebook)

* Sex, Embarked, Deck and Tile encoded using OneHotEncoder.

### Results
Based on data simplicity, looks  models like GradientBoosting and RandomForestClassifier overfits the data even whithout adjusting hyperparameters.
Further working with KNN and SVC models, SVC seems to have achived a bit better results.
Best score achieved : **0,79186** with **SVC**
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
