import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import train_test_split, ShuffleSplit, cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeRegressor

def find_best_model(x_data, y_data):
    algorithms = {
        'linear_regression' : {
            'model': LinearRegression(),
            'params': {
                'fit_intercept': [True, False]
            }
        },
        'lasso': {
            'model': Lasso(),
            'params': {
                'alpha': [1,2],
                'selection': ['random', 'cyclic']
            }
        },
        'decision_tree': {
            'model': DecisionTreeRegressor(),
            'params': {
                'criterion' : ['squared_error','friedman_mse'],
                'splitter': ['best','random']
            }
        }
    }
    scores = []
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    for name, config in algorithms.items():
        gridSearch =  GridSearchCV(config['model'], config['params'], cv=cv, return_train_score=False)
        gridSearch.fit(x_data, y_data)
        scores.append({
            'model': name,
            'best_score': gridSearch.best_score_,
            'best_params': gridSearch.best_params_
        })

    return pd.DataFrame(scores,columns=['model','best_score','best_params'])

def predict_price(location, sqft, bath, bhk, model):
    loc_index = np.where(X.columns==location)[0][0]

    x = np.zeros(len(X.columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    return model.predict([x])[0]

def save_model_to_pickle(pickle_file, model):
    with open(pickle_file, 'wb') as f:
        pickle.dump(model, f)

if __name__ == "__main__":
    data = pd.read_csv("Cleaned_Dataset.csv")
    X = data.drop(['price'], axis='columns')
    y = data.price
    
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

    linear_regression = LinearRegression()
    linear_regression.fit(x_train, y_train)
    acc = linear_regression.score(x_test, y_test)

    cross_val = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    cv_scores = cross_val_score(LinearRegression(), X, y, cv=cross_val)
    best_models = find_best_model(x_train, y_train)
    # print(best_models)
    predicted_price = predict_price('1st Block Jayanagar',1235, 2, 2, linear_regression)
    # save_model_to_pickle('linear_regression_model.pickle', linear_regression)

