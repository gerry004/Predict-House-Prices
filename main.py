import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
#matplotlib.rcParams["figure.figsize"] = (20, 10)

#data.head shows beginning of dataset
#data.shape shows rows and columns of dataset
data = pd.read_csv("Initial_Prices.csv")
initial_data = data.head()
rows_columns = data.shape

#number of houses in a certain area type
data.groupby('area_type')['area_type'].agg('count')

#data.drop gets rid of data
data2 = data.drop(["area_type", "society", "balcony", "availability"], axis="columns")
data2.head()

#number of null values
data2.isnull().sum()

#dropping null values
data3 = data2.dropna()

#data3 is dataset without null values
data3.isnull().sum()

#data.unique shows - all sizes that are unique - no repetitive values - check consistency in dataset
data3['size'].unique()

#create new 'bhk' column with number of bedrooms as one integer
data3['bhk'] = data3['size'].apply(lambda x: int(x.split(" ")[0]))
data3.head()

#all unique bhk values
data3['bhk'].unique()

#all rows where there are more than 20 bhk - checking for anomalies
data3[data3.bhk > 20]

#all unique total sqft values
data3.total_sqft.unique()


def is_float(x):
    try:
        float(x)
    except:
        return False
    return True


#all values for total sqft that aren't floats - i.e within a range
data3[~data3["total_sqft"].apply(is_float)].head()


#finding the average total sqft for the values that are in ranges
def convert_to_float(x):
    tokens = x.split(" - ")
    if len(tokens) == 2:
        return (float(tokens[0]) + float(tokens[1]))/2
    try:
        return float(x)
    except:
        return None


#check if above function works
convert_to_float("54 - 62")

#data4 is dataset where all values of total_sqft are floats
data4 = data3.copy()
data4["total_sqft"] = data4["total_sqft"].apply(convert_to_float)

data4.head()

#accessing values at index 30
data4.loc[30]

#creating new column price_per_sqft
data5 = data4.copy()
data5['price_per_sqft'] = data5["price"]*100000/data5["total_sqft"]

data5.head()

#number of unique values for location
len(data5.location.unique())

#number of properties at each location
data5.location = data5.location.apply(lambda x: x.strip())
location_stats = data5.groupby('location')['location'].agg('count').sort_values(ascending=False)
#print(location_stats)

#all locations with less than 10 prroperties
len(location_stats[location_stats <= 10])
location_stats_less_than_ten = location_stats[location_stats <= 10]

len(data5.location.unique())

#putting locations with less than 10 properties into an other column in dataset
data5.location = data5.location.apply(lambda x: 'other' if x in location_stats_less_than_ten else x)
#number of unique locations in dataset
len(data5.location.unique())

data5.head(20)

print(data5[data5.total_sqft/data5.bhk < 300].head())

data6 = data5[~(data5.total_sqft/data5.bhk < 300)]
print(data6.shape)

print(data6.price_per_sqft.describe())


def remove_pps_outliers(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('location'):
        m = np.mean(subdf.price_per_sqft)
        st = np.std(subdf.price_per_sqft)
        reduced_df = subdf[(subdf.price_per_sqft > (m-st)) & (subdf.price_per_sqft<=(m+st))]
        df_out = pd.concat([df_out, reduced_df], ignore_index=True)
    return df_out


data7 = remove_pps_outliers(data6)
print(data7.shape)


def plot_scatter_chart(df, location):
    bhk2 = df[(df.location == location) & (df.bhk == 2)]
    bhk3 = df[(df.location == location) & (df.bhk == 3)]
    matplotlib.rcParams['figure.figsize'] = (15, 10)
    plt.scatter(bhk2.total_sqft, bhk2.price, color='blue', label='2 BHK', s=50)
    plt.scatter(bhk3.total_sqft, bhk3.price, marker='+', color='green', label='3 BHK', s=50)
    plt.xlabel("Total Square Feet Area")
    plt.ylabel("Price (Lakh Indian Rupees)")
    plt.title(location)
    plt.legend()
    plt.show()


# plot_scatter_chart(data7, "Rajaji Nagar")

#plot_scatter_chart(data7,"Hebbal")


def remove_bhk_outliers(df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby('location'):
        bhk_stats = {}
        for bhk, bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk] = {
                'mean': np.mean(bhk_df.price_per_sqft),
                'std': np.std(bhk_df.price_per_sqft),
                'count': bhk_df.shape[0]
            }
        for bhk, bhk_df in location_df.groupby('bhk'):
            stats = bhk_stats.get(bhk-1)
            if stats and stats['count']>5:
                exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values)
    return df.drop(exclude_indices, axis='index')


data8 = remove_bhk_outliers(data7)
print(data8.shape)

#plot_scatter_chart(data8, "Rajaji Nagar")
#plot_scatter_chart(data8, "Hebbal")


matplotlib.rcParams["figure.figsize"] = (20, 10)
plt.hist(data8.price_per_sqft, rwidth=0.8)
plt.xlabel("Price Per Square Feet")
plt.ylabel("Count")
#plt.show()


print(data8.bath.unique())
plt.hist(data8.bath,rwidth=0.8)
plt.xlabel("Number of bathrooms")
plt.ylabel("Count")
# plt.show()

print(data8[data8.bath > data8.bhk+2])

data9 = data8[data8.bath < data8.bhk+2]
print(data9.shape)

data10 = data9.drop(['size', 'price_per_sqft'], axis='columns')
print(data10.head(3))

dummies = pd.get_dummies(data10.location)
print(dummies.head(3))

df11 = pd.concat([data10, dummies.drop('other', axis='columns')], axis='columns')
print(df11.head())

df12 = df11.drop('location', axis='columns')
print(df12.head(2))

X = df12.drop(['price'], axis='columns')
print(X.head(3))

y = df12.price
print(y.head(3))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=10)

from sklearn.linear_model import LinearRegression
lr_clf = LinearRegression()
lr_clf.fit(X_train,y_train)
acc = lr_clf.score(X_test,y_test)
print(acc)

from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score

cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)

crossValScore = cross_val_score(LinearRegression(), X, y, cv=cv)
print(crossValScore)

from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor


def find_best_model_using_gridsearchcv(X,y):
    algos = {
        'linear_regression' : {
            'model': LinearRegression(),
            'params': {
                'normalize': [True, False]
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
                'criterion' : ['mse','friedman_mse'],
                'splitter': ['best','random']
            }
        }
    }
    scores = []
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    for algo_name, config in algos.items():
        gs =  GridSearchCV(config['model'], config['params'], cv=cv, return_train_score=False)
        gs.fit(X,y)
        scores.append({
            'model': algo_name,
            'best_score': gs.best_score_,
            'best_params': gs.best_params_
        })

    return pd.DataFrame(scores,columns=['model','best_score','best_params'])


#print(find_best_model_using_gridsearchcv(X,y))

def predict_price(location,sqft,bath,bhk):
    loc_index = np.where(X.columns==location)[0][0]

    x = np.zeros(len(X.columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    return lr_clf.predict([x])[0]


print(predict_price('1st Phase JP Nagar',1000, 2, 2))

#saving model to a pickle file
#import pickle
#with open('banglore_home_prices_model.pickle', 'wb') as f:
    #pickle.dump(lr_clf, f)

#exporting columns into a json file
#import json
#columns = {
    #'data_columns' : [col.lower() for col in X.columns]
#}
#with open("columns.json","w") as f:
#    f.write(json.dumps(columns))