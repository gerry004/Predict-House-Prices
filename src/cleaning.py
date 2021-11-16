import pandas as pd
from helpers import is_float, convert_range_to_float, remove_pps_outliers, remove_bhk_outliers

data = pd.read_csv("Initial_Prices.csv")
print("Initial Data", data.head())

# number of houses in a certain area type
data.groupby('area_type')['area_type'].agg('count')

# drop data
data2 = data.drop(["area_type", "society", "balcony", "availability"], axis="columns")
data3 = data2.dropna()

# new 'bhk' column = number of bedrooms
data3['bhk'] = data3['size'].apply(lambda x: int(x.split(" ")[0]))

# check for houses with more than 20 bedrooms
data3[data3.bhk > 20]

# total_sqft convert all to float value
data3[~data3["total_sqft"].apply(is_float)].head()
data4 = data3.copy()
data4["total_sqft"] = data4["total_sqft"].apply(convert_range_to_float)

# new "price_per_sqft" column
data5 = data4.copy()
data5['price_per_sqft'] = data5["price"]*100000/data5["total_sqft"]

# sort "location" column
data5.location = data5.location.apply(lambda x: x.strip())
location_stats = data5.groupby('location')['location'].agg('count').sort_values(ascending=False)
location_stats_less_than_ten = location_stats[location_stats <= 10]
data5.location = data5.location.apply(lambda x: 'other' if x in location_stats_less_than_ten else x)

# check for houses with too many bedrooms per sqft
data6 = data5[~(data5.total_sqft/data5.bhk < 300)]

# remove outliers
data7 = remove_pps_outliers(data6)
data8 = remove_bhk_outliers(data7)

# check if there are more bathrooms than bedrooms
data9 = data8[data8.bath < data8.bhk+2]

# convert location into numerical data 
data10 = data9.drop(['size', 'price_per_sqft'], axis='columns')
dummies = pd.get_dummies(data10.location)
data11 = pd.concat([data10, dummies.drop('other', axis='columns')], axis='columns')
data12 = data11.drop('location', axis='columns')

# save cleaned dataset to csv
data12.to_csv('Cleaned_Dataset.csv')
