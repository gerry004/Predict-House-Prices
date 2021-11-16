import matplotlib
from matplotlib import pyplot as plt
from cleaning import data8

matplotlib.rcParams["figure.figsize"] = (20, 10)

def plot_histogram(data, xlabel, ylabel):
    plt.hist(data, rwidth=0.8)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

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

if __name__ == "__main__":
    plot_scatter_chart(data8, "Rajaji Nagar")