import preprocess as pre
import model_selection as ms

data = pre.get_data('./data/train.csv')
X, y = pre.preprocess(data)

ms.cross_validate_models(X,y)
# print(data.head(), y)
