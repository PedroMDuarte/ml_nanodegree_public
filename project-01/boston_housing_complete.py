"""Load the Boston dataset and examine its target (label) distribution."""

# Load libraries
import numpy as np
from sklearn import datasets
from sklearn.tree import DecisionTreeRegressor

################################
### ADD EXTRA LIBRARIES HERE ###
################################

from sys import stdout
import pandas as pd
import matplotlib.pyplot as plt
import collections
from sklearn.tree import export_graphviz
import sklearn.metrics as skmetrics
from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV
from scipy.stats import mode as scipymode

try:
    import tabulate
except:
    print "tabulate module failed to import."


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

    return output


def split_data(city_data):
    """Randomly shuffle the sample set. Divide it into 70 percent training and
    30 percent testing data."""

    # Get the features and labels from the Boston housing data
    X, y = city_data.data, city_data.target

    ###################################
    ### Step 2. YOUR CODE GOES HERE ###
    ###################################
    assert len(X) == len(y)

    X_train, X_test, y_train, y_test = cross_validation.train_test_split(
        X, y, train_size=0.7, random_state=0)

    output = collections.OrderedDict()
    output["train set size"] = len(y_train)
    output["test set size"] = len(y_test)

    return X_train, y_train, X_test, y_test, output


def performance_metric(label, prediction):
    """Calculate and return the appropriate error performance metric."""

    ###################################
    ### Step 3. YOUR CODE GOES HERE ###
    ###################################

    # The following page has a table of scoring functions in sklearn:
    # http://scikit-learn.org/stable/modules/classes.html#sklearn-metrics-metrics

    # In order to study all of the different performance metrics, I will simply
    # calculate them all and return a dictionary with all of the results
    l, p = label, prediction

    output = collections.OrderedDict()

    output["explained variance score"] = skmetrics.explained_variance_score(
        l, p)
    output["mean absolute error"] = skmetrics.mean_absolute_error(l, p)

    output["mean squared error"] = skmetrics.mean_squared_error(l, p)

    output["root mean squared error"] = np.sqrt(
        skmetrics.mean_squared_error(l, p))

    output["median absolute error"] = skmetrics.median_absolute_error(l, p)

    output["r2 score"] = skmetrics.r2_score(l, p)

    return output


def learning_curve(depth, X_train, y_train, X_test, y_test):
    """Calculate the performance of the model after a set of training data."""

    # We will vary the training set size so that we have 50 different sizes
    sizes = np.round(np.linspace(1, len(X_train), 50))

    print "\nDecision Tree with Max Depth: "
    print depth

    all_metrics = {
        "explained variance score": (-0.5, 1.2),
        "mean absolute error": (-0.5, 10.),
        "mean squared error": (-5., 120.),
        "root mean squared error": (-0.5, 10.),
        "median absolute error": (-0.5, 10.),
        "r2 score": (-0.5, 1.2)}

    # Note that I incorporated the graphing functionality directly into the
    # body of `learning_curve`, to make it easiar to collect results for the
    # different metrics and the various repetitions of the tree fitting.
    #
    # The code below allows us to make a learning plot for each error metric
    # at each value of the training depth. That is a total of 6 plots per
    # metric.

    figs = {
        metric_used: plt.subplots(1, 2,
                                  figsize=(9.6, 2.8),
                                  gridspec_kw={
                                      "width_ratios": [1.3, 1.],
                                      "left": 0.07,
                                      "right": 0.95,
                                      "bottom": 0.17,
                                      "top": 0.9,
                                      "wspace": 0.24,
                                  })
        for metric_used in all_metrics.keys()}

    # Given the random nature of the DecisionTreeRegressor, we will fit the
    # tree a number of times to get an idea of the average result.
    ntimes = 10
    print "iter =",
    for ii in range(ntimes):
        print ii,
        stdout.flush()
        train_error = []
        test_error = []

        ytest_target_all_samples = None
        ytest_predicted_all_samples = None

        ytrain_target_all_samples = None
        ytrain_predicted_all_samples = None

        for i, s in enumerate(sizes):
            # Create and fit the decision tree regressor model
            regressor = DecisionTreeRegressor(
                max_depth=depth, random_state=None)
            regressor.fit(X_train[:s], y_train[:s])

            # Find the performance on the training and testing set
            ytrain_target = y_train[:s]
            ytrain_predicted = regressor.predict(X_train[:s])

            ytest_target = y_test
            ytest_predicted = regressor.predict(X_test)

            train_error.append(performance_metric(ytrain_target,
                                                  ytrain_predicted))
            test_error.append(performance_metric(ytest_target,
                                                 ytest_predicted))

            if i == len(sizes)-1:
                ytrain_target_all_samples = ytrain_target
                ytrain_predicted_all_samples = ytrain_predicted

                ytest_target_all_samples = ytest_target
                ytest_predicted_all_samples = ytest_predicted

        train_error_df = pd.DataFrame(train_error)
        test_error_df = pd.DataFrame(test_error)

        for metric_used in all_metrics.keys():
            ax = figs[metric_used][1]

            ax[0].plot(sizes, test_error_df[metric_used],
                       lw=1, color='b', alpha=0.6,
                       label='test error')
            ax[0].plot(sizes, train_error_df[metric_used],
                       lw=1, color='g', alpha=0.6,
                       label='training error')

            if ii == 0:
            # Plot also the yhat vs. y to see how well the prediction actually
            # goes for the test and train set
                slope1 = np.linspace(0, 60., 10)
                ax[1].plot(slope1, slope1, '-', color='0.5')
                ax[1].plot(ytest_target_all_samples, ytest_predicted_all_samples,
                           's', ms=2.5, mfc='b', mec='None', alpha=0.4)
                ax[1].plot(ytrain_target_all_samples, ytrain_predicted_all_samples,
                           'o', ms=2.5, mfc='g', mec='None', alpha=0.4)


    # Format and save the graphs:
    print "saving figures ..."
    for metric_used, ylims in all_metrics.items():
        ax = figs[metric_used][1]
        ax[0].text(0.05, 0.95, 'depth = {}'.format(depth),
                   ha='left', va='top', transform=ax[0].transAxes)
        ax[0].set_xlabel('Training Size')
        ax[0].set_ylabel(metric_used)
        ax[0].set_ylim(*ylims)
        ax[0].set_xlim(0., 360.)
        handles, labels = ax[0].get_legend_handles_labels()
        ax[0].legend(handles[:2], labels[:2], loc='best', prop={'size': 10})

        ax[1].set_xlim(0., 55.)
        ax[1].set_ylim(0., 55.)
        ax[1].set_xlabel('target')
        ax[1].set_ylabel('prediction')
        ax[1].set_title(
            '$\hat{y}\ \mathrm{vs.}\ y,\ \mathrm{size}=' +
            "{:d}".format(len(X_train)) +
            '\ \mathrm{(only\ 0th\ iter)}$')

        # fig.tight_layout()
        figs[metric_used][0].savefig(
            'learning_curves_depth_{:02d}_{}.png'.format(
                depth, metric_used.replace(
                    ' ', '')), dpi=120)
    plt.close('all')

def model_complexity(X_train, y_train, X_test, y_test):
    """Calculate the performance of the model as model complexity increases.

    From the lecture notes:

    The key is to find the sweet spot that minimizes bias and variance by
    finding the right model complexity. And of course the more data the
    better any model can improve over time.


    Also from the lecture notes:

    Unlike a learning curve graph, a model complexity graph looks at how the
    complexity of a model changes the training and testing curves rather than
    the number of data points to train on. The general trend of is that as a
    model increases, the more variability exists in the model for a fixed set of
    data.
    """

    print "Model Complexity: "

    # We will vary the depth of decision trees from 2 to 25
    max_depth = np.arange(1, 25)

    # Similarly to what I did in `learning_curve`, I will use a list here, so
    # that I can collect the results for all of the performance metrics.
    train_err = []
    test_err = []

    for i, d in enumerate(max_depth):
        # Setup a Decision Tree Regressor so that it learns a tree with depth d
        regressor = DecisionTreeRegressor(max_depth=d)

        # Fit the learner to the training data
        regressor.fit(X_train, y_train)

        # Find the performance on the training set
        train_err.append(
            performance_metric(
                y_train,
                regressor.predict(X_train)))

        # Find the performance on the testing set
        test_err.append(performance_metric(y_test, regressor.predict(X_test)))

    # Plot the model complexity graph
    train_error_df = pd.DataFrame(train_err)
    test_error_df = pd.DataFrame(test_err)

    # For the model complexity graph I am only intersted in comparing two of the
    # metrics.  I make a plot fore each of them with the loop below:
    for metric_used in ["mean squared error", "median absolute error"]:
        model_complexity_graph(
            max_depth,
            train_error_df[metric_used],
            test_error_df[metric_used],
            metric_used)


def model_complexity_graph(max_depth, train_err, test_err, metric):
    """Plot training and test error as a function of the depth of the decision
    tree learn."""

    fig, ax = plt.subplots(1, 1, figsize=(7.5, 3.0))
    ax.plot(max_depth, test_err, '-s', lw=2, label='test error')
    ax.plot(max_depth, train_err, '-o', lw=2, label='training error')
    ax.legend(loc='best', prop={'size': 10})
    ax.set_xlabel('Max Depth')
    ax.set_ylabel(metric)
    y0, y1 = ax.get_ylim()
    ax.set_ylim(y0 - (y1-y0)*0.02, None)
    fig.tight_layout()

    metric_nospace = metric.replace(' ', '')
    fig.savefig('model_complexity_{}.png'.format(metric_nospace), dpi=150)


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

    # We will work with two scores here, one using the median absolute error
    # and one using R^2:

    scorer_median = skmetrics.make_scorer(skmetrics.median_absolute_error,
                                          greater_is_better=False)

    scorer_r2 = skmetrics.make_scorer(skmetrics.r2_score)

    # 2. We will use grid search to fine tune the Decision Tree Regressor and
    # obtain the parameters that generate the best training performance. Set up
    # the grid search object here.
    # http://scikit-learn.org/stable/modules/generated/sklearn.grid_search.GridSearchCV.html#sklearn.grid_search.GridSearchCV

    # --------------------------------------------------------------------------
    # First we setup KFold with shuffling (to avoid the issue with data that
    # may be initially ordered).
    cv = cross_validation.KFold(len(y), 3, shuffle=True, random_state=0)

    # To make sure the folds are properly shuffled, I will plot histograms of
    # each of the folds and check that they are distributed similarly to the
    # entire dataset:
    plot_cv_histograms(y, cv)

    # With the cross_validation setup and ready to go, we can go ahead and setup
    # the GridSearchCVs (remember we will be using two scorers and will compare
    # the results later):

    grid_searches = [GridSearchCV(estimator=regressor,
                                  param_grid=parameters,
                                  scoring=scorer,
                                  verbose=0)
                     for scorer in [scorer_median, scorer_r2]]

    # Fit the learner to the training data to obtain the best parameter set

    # Keeping in mind that the regressor we are using has a random state, we may
    # obtain a different "best depth" every time we run the grid search.

    # I have written a function that runs the grid search a few times, and
    # returns some statistics about the "best depth" obtained.

    for gs, name in zip(grid_searches,  ["median", "r2"]):
        ntimes = 50

        print "\n\n" + "="*80
        print name.upper()
        print "="*80

        mean, mode, stdev = run_search_ntimes(gs, X, y, ntimes)

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

        # save the resulting tree as a graph
        export_graphviz(
            best_tree.tree_,
            out_file='tree_depth_{}.dot'.format(name))

        # Use the model to predict the output of a particular sample
        xsample = np.array([11.95, 0.00, 18.100, 0, 0.6590, 5.6090,
                            90.00, 1.385, 24, 680.0, 20.20, 332.09, 12.13])
        ypred = best_tree.predict(xsample.reshape(1, -1))
        print "\nHouse: " + str(xsample)
        print "Prediction: " + str(ypred)


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


def plot_cv_histograms(ydata, cv):

    # setup 4 bins, based on the entire target data:
    counts, bins = np.histogram(ydata, bins=4)
    binw = np.diff(bins)[0]
    target_bins = [bins[0]-binw] + list(bins) + [bins[-1] + binw]
    x = np.arange(len(target_bins)-1) * binw + binw/2. + target_bins[0]

    # setup figure:
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

    # plot bin values (distribution) of the target data on bottom axis:
    counts, bins = np.histogram(ydata, bins=target_bins)
    ax2.bar(x - binw / 2., counts / float(sum(counts)), binw,
            label='all target data')

    # plot distribution for each of the folds on top axis,
    colors = ['blue', 'green', 'red']
    for ii, (train, test) in enumerate(cv):
        counts, bins = np.histogram(ydata[test], bins=target_bins)
        ax1.bar(x - binw/2. + binw/3.*ii, counts/float(sum(counts)), binw/3.,
                color=colors[ii], label='fold #{}'.format(ii))

    # save the figure:
    ax2.set_xlabel('target value')
    ax1.set_ylabel('normalized frequency')
    ax2.set_ylabel('normalized frequency')
    ax1.legend(loc='best', prop={'size': 10})
    ax2.legend(loc='best', prop={'size': 10})
    fig.tight_layout()
    fig.savefig('kfold_histograms.png', dpi=150)


def main():
    """Analyze the Boston housing data. Evaluate and validate the
    performanance of a Decision Tree regressor on the housing data.
    Fine tune the model to make prediction on unseen data."""

    # Load data
    city_data = load_data()

    # Explore the data
    explore_output = explore_city_data(city_data)
    for item in explore_output.items():
        print item[0].rjust(20), " : ", "{:g}".format(item[1]).rjust(10)
    with open('section-01.tex', 'w') as fout:
        try:
            fout.write(tabulate.tabulate(explore_output.items(),
                                         tablefmt='latex'))
        except:
            print "failed to save output as latex table"

    # Training/Test dataset split
    X_train, y_train, X_test, y_test, split_output = split_data(city_data)
    for item in split_output.items():
        print item[0].rjust(20), " : ", "{:g}".format(item[1]).rjust(10)

    # Learning Curve Graphs
    max_depths = [1,2,3,4,5,6,7,8,9,10]
    for i, max_depth in enumerate(max_depths):
        learning_curve(max_depth, X_train, y_train, X_test, y_test)

    # Model Complexity Graph
    model_complexity(X_train, y_train, X_test, y_test)

    # Tune and predict Model
    fit_predict_model(city_data)


if __name__ == "__main__":
    main()
