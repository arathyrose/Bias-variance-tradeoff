'''
# MACHINE, DATA AND LEARNING: ASSIGNMENT 1: QUESTION 1
**TEAM NUMBER: 86**  
**TEAM MEMBERS:**
Tathagata Raha (2018114017)
Arathy Rose Tony(2018101042)
**TASK**:
Calculate the bias and variance of a dataset that is not sampled yet, and then predict the degree of the best fit curve
'''

# Header files included

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
import numpy
import random
import pickle
import pandas
import matplotlib.pyplot as plt

# Some global variables that can be used to check specific outputs

debug = 0  # 1 if you want to see the variable values during the program execution
graphing = 0  # 1 to see the necessary graphs

# STEP 1: LOADING THE DATASET AND VISUALISING IT

# Load the dataset:

# Here we load the data_set from data.pkl file which is stored in the same directory as the current notebook.
f = open('data.pkl', 'rb')
data_set = pickle.load(f)
data_set_size = len(data_set)
f.close()

# Now the data_set contains the entire data set points in the format <x,y>. The total number of elements in the data set is stored in data_set_size.
if(debug == 1):
    print(data_set)
    print(data_set_size)

# Split The Dataset Into X and Y

# Here we split the <x,y> pairs into x and y separate arrays
x = data_set[:, 0]
y = data_set[:, 1]
if(debug == 1):
    print("X: ", x)
    print("Y: ", y)

# Graphing the given dataset

# Here we plot the given dataset, just to get the feel of the dataset provided.
fig = plt.figure()
plt.plot(x, y, 'r.', markersize=2)
plt.show()

# STEP 2: RESAMPLING DATA SETS

# Split The Dataset Into Testing And Training Datasets
xTrain, xTest, yTrain, yTest = train_test_split(
    x, y, test_size=0.1, random_state=3, shuffle=True)
test_data_size = len(xTest)
train_data_size = len(xTrain)

if(debug == 1):
    print(test_data_size)
    print(train_data_size)

# Split The Training Dataset Into 10 Different Training Datasets

# Here, we run a loop 10 times and store the train dataset values from start_in to the end_in as a numpy array in a list (X_train_data_sets and Y_train_data_sets), and update the values of start_in and end_in in each iteration of the loop.
X_train_data_sets = list()
Y_train_data_sets = list()
start_in = 0
train_data_sets_size = int(train_data_size/10)
end_in = train_data_sets_size
for i in range(10):
    X_train_data_sets.append(xTrain[start_in:end_in])
    Y_train_data_sets.append(yTrain[start_in:end_in])
    start_in += train_data_sets_size
    end_in += train_data_sets_size

if(debug == 1):
    print(train_data_sets_size)

# Graphing Each Of The Training Datasets

# Here we graph each of the training datasets separately. (to check if the datasets are sampled properly)
if(debug == 1):
    for i in range(10):
        print("Training set ", i)
        fig = plt.figure()
        plt.plot(X_train_data_sets[i],
                 Y_train_data_sets[i], 'r.', markersize=2)
        plt.show()

# STEP 3: TRAINING A MODEL

# Function just to store what exactly to be done to train a polynomial regression model of a given degree and given training dataset number


def create_polynomial_regression_model(degree, i):
    '''
    The following function returns a polynomial regression model for the given degree. I use this to find the lines required to make a model, of a given degree
    '''
    i = 0
    # prepare the matrix of the powers of x
    poly_features = PolynomialFeatures(degree=degree)
    # transpose the x row matrix into a column matrix
    x = X_train_data_sets[i][:, numpy.newaxis]
    # get a matrix containing the higher powers of X in the format: [1 X X^2 X^3 ...]
    X_train_poly = poly_features.fit_transform(x)
    poly_model = LinearRegression()
    # fit the transformed features to Linear Regression
    poly_model.fit(X_train_poly, Y_train_data_sets[0])
    y_test_predict = poly_model.predict(
        poly_features.fit_transform(xTest))    # predicting on test data-set
    return poly_model


# Plotting A Graph Of The Trained Polynomial Regression Model
'''
Here, we take each of the training datasets, and plot the training dataset points and the values predicted by the model on the test dataset points, to visualise the provided data.
For each training set, 9 graphs are plotted, each corresponding to the model of each degree (from 0 to 9)
'''
if(graphing == 1):
    for i in range(10):
        print("TRAINING SET ", i)
        f = plt.figure()
        f, axes = plt.subplots(nrows=3, ncols=3, sharex=True,
                               sharey=True, figsize=(30, 30))
        x = X_train_data_sets[i][:, numpy.newaxis]  # transposing it
        y = Y_train_data_sets[i]
        for degree in range(0, 9):
            axes[int(degree/3)][int(degree % 3)].plot(x, y, 'r.', markersize=4)
            poly_features = PolynomialFeatures(degree=degree+1)
            X_train_poly = poly_features.fit_transform(x)
            poly_model = LinearRegression()
            poly_model.fit(X_train_poly, Y_train_data_sets[i])
            y_test_predict = poly_model.predict(
                poly_features.fit_transform(xTest[:, numpy.newaxis]))
            axes[int(degree/3)][int(degree % 3)].plot(xTest[:,
                                                            numpy.newaxis], y_test_predict, 'b.', markersize=4)
            plt.title("DEGREE "+str(degree+1))
            plt.xlabel("X")
            plt.ylabel("Y")
        plt.show()

# STEP 4: CALCULATE THE BIAS AND VARIANCE OF THE MODEL

# Get the list of all the predicted values

'''
First we get the list of all the y predicted values for all the models and for all the degrees separately in a 2-D array.
Here, for each model of each degree, we get the predicted y values for the given test datasets.
The values are stored as follows: y[train_data_set_no][degree]
'''
y_predicted = []
for i in range(10):
    x = X_train_data_sets[i][:, numpy.newaxis]  # transposing it
    y = Y_train_data_sets[i]
    temp = []
    for degree in range(0, 9):
        poly_features = PolynomialFeatures(degree=degree)
        X_train_poly = poly_features.fit_transform(x)
        poly_model = LinearRegression()
        poly_model.fit(X_train_poly, Y_train_data_sets[i])
        y_test_predict = poly_model.predict(
            poly_features.fit_transform(xTest[:, numpy.newaxis]))
        temp.append(y_test_predict)
    y_predicted.append(temp)

# Function for calculating the bias and the variance
'''
Then we calculate the bias and variance as follows:
- For a given degree we append the values of the y_predicted for each model to a list
- Convert this list to a numpy array y_predicted_part
- Calculate the bias of this list by subtracting the mean of the model from the testing dataset
- Bias corresponding to the models of a given degree is the mean of this list
- Similarly calculate the variance of this list
- Variance corresponding to the models of a given degree is the mean of this list
'''


def find_bias_variance(order):
    y_predicted_part = []
    for i in range(10):
        y_predicted_part.append(y_predicted[i][order])
    y_predicted_part = numpy.asarray(y_predicted_part)
    bias = numpy.abs(numpy.mean(y_predicted_part, axis=0) - yTest)
    variance = numpy.var(y_predicted_part, axis=0)
    return(numpy.mean(bias), numpy.mean(variance))


# Then we call the function as follows, in order to populate the lists, bias and variance.
bias = []
variance = []
for i in range(9):
    b, v = find_bias_variance(i)
    bias.append(b)
    variance.append(v)
print("Bias:", bias)
print("Variance:", variance)
# The lists, bias and variance, now contain the bias and variance corresponding to a particular degree.

# Tabulate the values

# We use the pandas library in order to display the required items in a table format
final_table = dict()
final_table["DEGREE"] = range(1, 10)
final_table["BIAS"] = bias
final_table["BIAS^2"] = list(numpy.array(bias)**2)
final_table["VARIANCE"] = variance
final_table["MSE"] = list(numpy.array(
    final_table["BIAS^2"])+numpy.array(variance))
df = pandas.DataFrame(final_table)
print(df)

# Plot the bias-variance tradeoff
plt.plot(final_table["DEGREE"], final_table["VARIANCE"], color="blue")
plt.plot(final_table["DEGREE"], final_table["BIAS^2"], color="red")
plt.plot(final_table["DEGREE"], final_table["MSE"], color="green")
plt.title("BIAS VARIANCE TRADEOFF")
plt.xlabel("Degree")
plt.ylabel("Bias^2/Variance")
plt.legend(["VARIANCE", "BIAS^2", "MSE"])
plt.show()

# FITTING THE TRAINED MODEL TO THE TESTING DATASET FOR DISPLAYING THE LINE OF BEST FIT

f = plt.figure()
f, axes = plt.subplots(nrows=3, ncols=3, sharex=True,
                       sharey=True, figsize=(30, 30))
for degree in range(0, 9):
    xtemp = numpy.concatenate([xTest for i in range(10)])
    y_predicted_part = []
    for i in range(10):
        y_predicted_part.append(y_predicted[i][degree])
    ytemp = numpy.array(y_predicted_part).reshape(-1)
    axes[int((degree)/3)][int((degree) % 3)
                          ].plot(xTest, yTest, 'r.', markersize=10)
    axes[int((degree)/3)][int((degree) % 3)
                          ].plot(xtemp, ytemp, 'b.', markersize=1)
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

# Here the red line represents the actual testing data, while the blue line shows the model we have made

print("----------------------------------------")

# IF THE BIAS IS CALCULATED AS THE RMS OF THE DIFFERENCE BETWEEN THE ACTUAL AND THE PREDICTED y VALUES

# For the explanation, refer the step 2. Also as it was mentioned in the news forum, this has been done. (no report has been done on this)


def find_bias_variance(order):
    y_predicted_part = []
    for i in range(10):
        y_predicted_part.append(y_predicted[i][order])
    y_predicted_part = numpy.asarray(y_predicted_part)
    bias2 = numpy.abs((numpy.mean(y_predicted_part, axis=0) - yTest)**2)
    variance = numpy.var(y_predicted_part, axis=0)
    return(numpy.mean(bias2), numpy.mean(variance))


bias2 = []
variance = []
bias = []
for i in range(9):
    b, v = find_bias_variance(i)
    bias2.append(b)
    bias.append(b**0.5)
    variance.append(v)
print("Bias^2:", bias2)
print("Bias:", bias)
print("Variance:", variance)

final_table = dict()
final_table["DEGREE"] = range(1, 10)
final_table["BIAS"] = bias
final_table["BIAS^2"] = list(numpy.array(bias)**2)
final_table["VARIANCE"] = variance
final_table["MSE"] = list(numpy.array(
    final_table["BIAS^2"])+numpy.array(variance))
df = pandas.DataFrame(final_table)
print(df)

plt.plot(final_table["DEGREE"], final_table["VARIANCE"], color="blue")
plt.plot(final_table["DEGREE"], final_table["BIAS^2"], color="red")
plt.plot(final_table["DEGREE"], final_table["MSE"], color="green")
plt.title("BIAS VARIANCE TRADEOFF")
plt.xlabel("Degree")
plt.ylabel("Bias^2/Variance")
plt.legend(["VARIANCE", "BIAS^2", "MSE"])
plt.show()
