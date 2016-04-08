"""Load the Boston dataset and examine its target (label) distribution."""

# Load libraries
import numpy as np
import pylab as pl
from sklearn import datasets
from sklearn.tree import DecisionTreeRegressor

################################
### ADD EXTRA LIBRARIES HERE ###
################################
import collections
import sklearn.metrics as skmetrics
from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV
from scipy.stats import mode as scipymode
from sklearn.neighbors import NearestNeighbors


def load_data():
    """Load the Boston dataset."""

    boston = datasets.load_boston()
    return boston


def explore_city_data(city_data):
    """Calculate the Boston housing statistics."""

    # Get the labels and features from the housing data
    housing_prices = city_data.target
    housing_features = city_data.data

    ###################################
    ### Step 1. YOUR CODE GOES HERE ###
    ###################################

    # Please calculate the following values using the Numpy library
    output = collections.OrderedDict()

    # Size of data (number of houses)?
    assert len(housing_prices) == len(housing_features)
    output["size of data"] = len(housing_prices)

    # Number of features?
    output["number of features"] = housing_features.shape[1]

    # Minimum price?
    output["minimum price"] = housing_prices.min()

    # Maximum price?
    output["maximum price"] = housing_prices.max()

    # Calculate mean price?
    output["mean price"] = np.mean(housing_prices)

    # Calculate median price?
    output["median price"] = np.median(housing_prices)

    # Calculate standard deviation?
    output["standard deviation"] = np.std(housing_prices)

    for key, val in output.items():
        print "{} : {:g}".format(key.rjust(20), val)


def split_data(city_data):
    """Randomly shuffle the sample set. Divide it into 70 percent training and
    30 percent testing data."""

    # Get the features and labels from the Boston housing data
    X, y = city_data.data, city_data.target

    ###################################
    ### Step 2. YOUR CODE GOES HERE ###
    ###################################
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(
        X, y, train_size=0.7, random_state=0)

    return X_train, y_train, X_test, y_test


def performance_metric(label, prediction):
    """Calculate and return the appropriate error performance metric."""

    ###################################
    ### Step 3. YOUR CODE GOES HERE ###
    ###################################

    # The following page has a table of scoring functions in sklearn:
    # http://scikit-learn.org/stable/modules/classes.html#sklearn-metrics-metrics
    return skmetrics.median_absolute_error(label, prediction)


def learning_curve(depth, X_train, y_train, X_test, y_test):
    """Calculate the performance of the model after a set of training data."""

    # We will vary the training set size so that we have 50 different sizes
    sizes = np.round(np.linspace(1, len(X_train), 50))
    train_err = np.zeros(len(sizes))
    test_err = np.zeros(len(sizes))

    print "Decision Tree with Max Depth: "
    print depth

    for i, s in enumerate(sizes):

        # Create and fit the decision tree regressor model
        regressor = DecisionTreeRegressor(max_depth=depth)
        regressor.fit(X_train[:s], y_train[:s])

        # Find the performance on the training and testing set
        train_err[i] = performance_metric(y_train[:s],
                                          regressor.predict(X_train[:s]))

        test_err[i] = performance_metric(y_test, regressor.predict(X_test))

    # Plot learning curve graph
    learning_curve_graph(sizes, train_err, test_err)


def learning_curve_graph(sizes, train_err, test_err):
    """Plot training and test error as a function of the training size."""

    pl.figure()
    pl.title('Decision Trees: Performance vs Training Size')
    pl.plot(sizes, test_err, lw=2, label='test error')
    pl.plot(sizes, train_err, lw=2, label='training error')
    pl.legend()
    pl.xlabel('Training Size')
    pl.ylabel('Error')
    pl.show()


def model_complexity(X_train, y_train, X_test, y_test):
    """Calculate the performance of the model as model complexity increases."""

    print "Model Complexity: "

    # We will vary the depth of decision trees from 2 to 25
    max_depth = np.arange(1, 25)
    train_err = np.zeros(len(max_depth))
    test_err = np.zeros(len(max_depth))

    for i, d in enumerate(max_depth):
        # Setup a Decision Tree Regressor so that it learns a tree with depth d
        regressor = DecisionTreeRegressor(max_depth=d)

        # Fit the learner to the training data
        regressor.fit(X_train, y_train)

        # Find the performance on the training set
        train_err[i] = performance_metric(y_train, regressor.predict(X_train))

        # Find the performance on the testing set
        test_err[i] = performance_metric(y_test, regressor.predict(X_test))

    # Plot the model complexity graph
    model_complexity_graph(max_depth, train_err, test_err)


def model_complexity_graph(max_depth, train_err, test_err):
    """Plot training and test error as a function of the depth of the decision
    tree learn."""

    pl.figure()
    pl.title('Decision Trees: Performance vs Max Depth')
    pl.plot(max_depth, test_err, lw=2, label='test error')
    pl.plot(max_depth, train_err, lw=2, label='training error')
    pl.legend()
    pl.xlabel('Max Depth')
    pl.ylabel('Error')
    pl.show()


def fit_predict_model(city_data):
    """Find and tune the optimal model. Make a prediction on housing data."""

    # Get the features and labels from the Boston housing data
    X, y = city_data.data, city_data.target

    # Setup a Decision Tree Regressor

    # IMPORTANT NOTE:  Note that this regressor has not been given a specific
    # random_state.  Each time we run it we will get a different result.

    regressor = DecisionTreeRegressor(random_state=None)

    parameters = {'max_depth': (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)}

    ###################################
    ### Step 4. YOUR CODE GOES HERE ###
    ###################################

    # 1. Find an appropriate performance metric. This should be the same as the
    # one used in your performance_metric procedure above:
    # http://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html
    scorer = skmetrics.make_scorer(skmetrics.median_absolute_error,
                                   greater_is_better=False)

    # 2. We will use grid search to fine tune the Decision Tree Regressor and
    # obtain the parameters that generate the best training performance. Set up
    # the grid search object here.
    # http://scikit-learn.org/stable/modules/generated/sklearn.grid_search.GridSearchCV.html#sklearn.grid_search.GridSearchCV

    # then we setup the grid search:
    reg = GridSearchCV(estimator=regressor,
                       param_grid=parameters,
                       scoring=scorer,
                       verbose=0)

    # Fit the learner to the training data to obtain the best parameter set

    # Keeping in mind that the regressor we are using has a random state, we may
    # obtain a different "best depth" every time we run the grid search.

    # I have written a function that runs the grid search a n times, and
    # returns some statistics about the "best depth" obtained.

    ntimes = 50

    print "\n\n" + "="*80

    mean, mode, stdev = run_search_ntimes(reg, X, y, ntimes)

    print "Statistics for max_depth, after running grid search {} times:".\
          format(ntimes)

    print "    mean = {:g}".format(mean)
    print "    mode = {:g}".format(mode)
    print " std dev = {:g}".format(stdev)

    # The mode of the best_depths will be used as our final model:
    final_model_depth = np.round(mean)
    print "\nFinal Model: ", "best_depth = ", final_model_depth

    # We fit our final model using the best depth and the entire data set:
    best_tree = DecisionTreeRegressor(max_depth=final_model_depth)
    best_tree.fit(X, y)

    # Use the model to predict the output of a particular sample
    xsample = np.array([11.95, 0.00, 18.100, 0, 0.6590, 5.6090,
                        90.00, 1.385, 24, 680.0, 20.20, 332.09, 12.13])

    # reshape the xsample to comply with sklearn (1 row, nfeatuares cols)
    xsample = xsample.reshape(1, -1)

    ypred = best_tree.predict(xsample)
    print "\nHouse: " + str(xsample)
    print "Prediction: " + str(ypred)

    # From the Pro Tip given in my first review, I wil find the actual prices
    # for some of the nearest neighbors of the xsample in question and will
    # use that to validate if my prediction makes sense.

    nn = NearestNeighbors()


    nn.fit(X)
    _, neighbor_indices = nn.kneighbors(xsample)
    print "\nShowing the 5 nearest neighbors:"
    for index in neighbor_indices:
        print X[index]

    for num in [5, 10, 20, 30]:
        _, neighbor_indices = nn.kneighbors(xsample, num)
        average = np.mean([y[index] for index in neighbor_indices])
        print "\nAverage Price of {} nearest neighbors: ".format(num) + \
            str(average)


def run_search_ntimes(regressor, Xdat, ydat, ntimes):

    print
    best_depths = np.zeros(ntimes)
    for ii in range(ntimes):
        regressor.fit(Xdat, ydat)
        depth = regressor.best_params_['max_depth']
        print "Iteration #{:02d} best depth = {:d}".format(ii, depth)
        best_depths[ii] = depth

    return (best_depths.mean(),
            scipymode(best_depths).mode[0],
            np.std(best_depths))


def main():
    """Analyze the Boston housing data. Evaluate and validate the
    performanance of a Decision Tree regressor on the housing data.
    Fine tune the model to make prediction on unseen data."""

    # Load data
    city_data = load_data()

    # Explore the data
    explore_city_data(city_data)

    # Training/Test dataset split
    X_train, y_train, X_test, y_test = split_data(city_data)

    # Learning Curve Graphs
    max_depths = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    for max_depth in max_depths:
        learning_curve(max_depth, X_train, y_train, X_test, y_test)

    # Model Complexity Graph
    model_complexity(X_train, y_train, X_test, y_test)

    # Tune and predict Model
    fit_predict_model(city_data)


if __name__ == "__main__":
    main()
