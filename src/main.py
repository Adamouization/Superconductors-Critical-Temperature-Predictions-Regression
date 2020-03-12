import joblib
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR


RANDOM_SEED = 42


def main():
    # The data is imported into a pandas DataFrame.
    df = pd.read_csv("data/train.csv")

    # Split Train/Test data
    train_set, test_set = split_data(df)

    # We make a copy of the training set used for data exploration purposes without introducing the risk of
    # unvoluntarily harming the original training set.
    exploration_set = train_set.copy()

    # Visualising the data to gain insights.
    # visualise_data(exploration_set)
    # data_correlation(exploration_set)

    # Preparing the inputs and choosing a suitable subset.
    X, y = input_preparation(train_set)

    # Selecting and training a regression model.
    training_regression_models(X, y)

    # Evaluating the final performance of the model on the test set.
    # final_evaluation()

    # Function used to generate scatter plot used to show stratified sampling example in report.
    # random_plot()


def split_data(data):
    # Before even viewing the data to avoid data snooping, it is split between a training and a testing set.
    # A randomised 80%/20% split is used rather than stratified sampling because the dataset is large enough.
    # The random number generator's seed is set at a fixed value to ensure that it always generates the same shuffle
    # indices when re-running the code, ensuring reproducibility.
    train_set, test_set = train_test_split(data, test_size=0.2, random_state=RANDOM_SEED)

    # The train set should be 80% the size of the original, whereas the test set should 20%.
    # Now, only the train_set is used, while the test_set is set aside and forgotten about until the very end when
    # the model's performance will be evaluated.
    print("Train set size = {}, Test set size = {}".format(train_set.shape, test_set.shape))

    return train_set, test_set


def visualise_data(exploration_set) -> None:
    """

    :param exploration_set:
    :return:
    """
    # Visualising the data table.
    print("Visualising the first 5 rows of the training set:")
    print(exploration_set.head(5))

    # Number of columns and rows in the dataset.
    print("\nData shape: {} rows, {} columns".format(exploration_set.shape[0], exploration_set.shape[1]))

    # Visualise information about each attribute, ensuring that there are not missing features (the number of
    # non-null count should be equal to the number of rows) and whether attributes are numerical or not.
    print("\nGeneral information about the data attributes:")
    print(exploration_set.info())

    # Get a summary of numerical attributes (mean, min, max, standard deviation *std*, percentiles, ...).
    print("\nSummary of numerical attributes:")
    print(exploration_set.describe())

    # Plot each numerical attribute in a small histogram to more easily visualise it and make a quick analysis.
    # todo: fix
    # get_ipython().run_line_magic('matplotlib', 'inline')
    # exploration_set.hist(bins=50, figsize=(25, 18))
    # plt.savefig("plots/all_features_histogram.png")
    # plt.show()


def data_correlation(exploration_set):
    # The correlation matrix is computed to identify the correlation between each feature and the target value. The
    # closer it is to 1, the stronger the correlation is, and vice versa when the correlation is close to -1.
    # A correlation of 0 can be translated as a lack of linear correlation.
    correlation_matrix = exploration_set.corr()
    print("10 features with the strongest correlation with 'critical_temp':")
    print(correlation_matrix["critical_temp"].sort_values(ascending=False).head(11))
    print("\n10 features with the weakest correlation with 'critical_temp':")
    print(correlation_matrix["critical_temp"].sort_values(ascending=True).head(10))

    # The standard correlation coefficients of the target attribute and the 5 attributes with the highest correlation
    # are plotted in a scatter plot.
    attributes = [
        "critical_temp",
        "wtd_std_ThermalConductivity",
        "range_ThermalConductivity",
        "std_ThermalConductivity",
        "range_atomic_radius",
        "wtd_entropy_atomic_mass"
    ]

    # todo fix
    pd.plotting.scatter_matrix(exploration_set[attributes], figsize=(25, 18))

    # Because wtd_std_ThermalConductivity has the highest correlation with critical_temp, we can zoom in on the scatter
    # plot between these two variables.
    # todo fix
    exploration_set.plot(kind="scatter", x="wtd_std_ThermalConductivity", y="critical_temp", alpha=0.1)


def input_preparation(train_set):
    """
    In terms of data cleaning there are no attributes missing values, no 0 or NaN values, and inferences on non-sensical
    values cannot be made due to lack of expert knowledge in the domain of supercondutors.
    Additionally, there are no non-numerical attributes to deal with, so there is nothing to handle on this front.
    :param train_set:
    :return:
    """
    # Start by separating the predictors and the target values.
    X = train_set.drop("critical_temp", axis=1)
    y = train_set["critical_temp"].copy()

    # We can check that the split was correctly executed by ensuring that there are 82-1=81 columns in X, and only 1
    # column in y.
    print("Predictors X size = {}, Target y size = {}".format(X.shape, y.shape))

    return X, y


def training_regression_models(X, y):
    """
    Selecting and Training a regression model.
    The following errors are achieved by the paper when using multiple regression:
        -RMSE = 17.6K
        - R^2 = 0.74
    These results will be compared to our different models' results
    :param X:
    :param y:
    :return:
    """
    # general_linear_regression(X, y)
    linear_ridge_regression(X, y)
    # if "linear_reg":
    #     general_linear_regression(X, y)
    # elif "ridge_reg":
    #     linear_ridge_regression(X, y)
    # elif "lasso_reg":



def quick_prediction_test(model, X, y):
    """
    Function to briefly test the trained model with some data from the training set.
    :param model:
    :param X:
    :param y:
    :return:
    """
    some_data = X.iloc[:5]
    some_labels = y.iloc[:5]
    print("Predictions: {}".format(model.predict(some_data)))
    print("Real value: {}".format(list(some_labels)))


def evaluate_model_error(model, target, prediction):
    """
    Function to evaluate the RMSE and the R^2 score of the trained model.
    :param model:
    :param target:
    :param prediction:
    :return:
    """
    mse = mean_squared_error(target, prediction)
    rmse = np.sqrt(mse)
    print("RMSE={}K".format(round(rmse, 2)))
    r_squared = r2_score(target, prediction)
    print("R^2={}%".format(round(r_squared, 2)))


def display_detailed_scores(scores: list, metric_name: str) -> None:
    """
    Function to display cross-validation scores.
    :param scores:
    :param metric_name:
    :return:
    """
    if metric_name == "RMSE":
        metric_unit = "K"
    elif metric_name == "R2":
        metric_unit = "%"
    else:
        metric_unit = ""
    print("\n{} scores achieved: {}".format(metric_name, scores))
    print("\nMean {}: {}{} (+/-{})".format(metric_name, round(scores.mean(), 2), metric_unit, round(scores.std(), 2)))


def plot_best_fit(y_actual, y_predicted, fig_name: str = "") -> None:
    """

    :param y_actual:
    :param y_predicted:
    :param fig_name:
    :return:
    """
    m, b = np.polyfit(y_actual, y_predicted, 1)
    plt.scatter(y_actual, y_predicted, s=20, alpha=0.25, color="green")
    plt.plot(y_actual, (m * y_actual) + b, color='orange')
    plt.savefig("plots/plot_{}.png".format(fig_name))
    plt.show()


def k_fold_cross_validation(model, X, y, folds=10):
    """
    Function to perform k-fold cross validation on a model.
    :param model:
    :param X:
    :param y:
    :param folds:
    :return:
    """
    reg_model_mse_scores = cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=folds)
    # Computer negative score because sklearn's cross_val_score function expects a utility function rather than a cost
    # function, so it returns the reverse of the MSE.
    reg_model_rmse_scores = np.sqrt(-reg_model_mse_scores)
    reg_model_r2_scores = cross_val_score(model, X, y, scoring="r2", cv=folds)
    print("{}-fold cross validation results:".format(folds))
    display_detailed_scores(reg_model_rmse_scores, metric_name="RMSE")
    display_detailed_scores(reg_model_r2_scores, metric_name="R2")


def save_model(model, model_type):
    """
    Function to save the model to a file.
    :param model:
    :param model_type:
    :return:
    """
    joblib.dump(model, "trained_models/{}.pkl".format(model_type))


def load_model(model_type):
    """
    Function to load model.
    :param model_type:
    :return:
    """
    return joblib.load("trained_models/{}.pkl".format(model_type))


def grid_search_algorithm(model, X, y, parameters, folds=5):
    """
    Function to perform a grid search on a regression model.
    :param model:
    :param X:
    :param y:
    :param parameters:
    :param folds:
    :return:
    """
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=parameters,
        cv=folds,  # perform 5-fold cross validation on each iteration.
        scoring="neg_mean_squared_error",
        return_train_score=True
    )
    grid_search.fit(X, y)
    return grid_search


def general_linear_regression(X, y):
    """

    :param X:
    :param y:
    :return:
    """
    # Train a linear regression model and assess its performance.
    linear_regression = LinearRegression(fit_intercept=True, normalize=True)
    linear_regression.fit(X, y)
    quick_prediction_test(linear_regression, X, y)
    y_prediction = linear_regression.predict(X)
    evaluate_model_error(linear_regression, y, y_prediction)
    plot_best_fit(y, y_prediction, "linear_reg")
    save_model(linear_regression, "linear_reg")

    # Using the same model with 10-fold cross validation.
    k_fold_cross_validation(linear_regression, X, y)


def linear_ridge_regression(X, y):
    """

    :param X:
    :param y:
    :return:
    """
    # Train a linear regression model with Ridge regression and assess its performance.
    ridge_regression = Ridge(alpha=0.1, solver='svd', normalize=False)
    ridge_regression.fit(X, y)
    quick_prediction_test(ridge_regression, X, y)
    y_prediction = ridge_regression.predict(X)
    evaluate_model_error(ridge_regression, y, y_prediction)
    plot_best_fit(y, y_prediction, "ridge_reg")
    save_model(ridge_regression, "ridge_reg")

    # Using the same model with 10-fold cross validation.
    k_fold_cross_validation(ridge_regression, X, y)

    # Applying grid search algorithm to the Ridge Regression model to explore combinations of hyperparameters to find
    # the best hyperparameters for a model, since doing so manually is a very time consuming task.
    parameters = {
        "alpha": [0.1, 1, 10],
        "normalize": [True, False],
        "solver": ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"]
    }
    ridge_reg_gs_results = grid_search_algorithm(Ridge(), X, y, parameters, folds=5)
    ridge_reg_gs_results_df = pd.DataFrame(ridge_reg_gs_results.cv_results_)
    ridge_reg_gs_results_df.to_csv("grid_search_results/ridge_reg_grid_search_results.csv")

    # Top 5 hyperparameters found for Ridge Regression.
    print("Top 5 hyperparameters combinations found for Ridge Regression:")
    ridge_reg_gs_results_df.sort_values(by=['rank_test_score']).head(5)

    # Best model found by grid search algorithm for Ridge Regression.
    final_ridge_reg_model = ridge_reg_gs_results.best_estimator_
    print("\nBest model hyperparameters found by grid search algorithm for Ridge Regression:")
    print(final_ridge_reg_model)


def linear_lasso_regression(X, y):
    """

    :param X:
    :param y:
    :return:
    """
    # Train a linear regression model with Lasso regression and assess its performance.
    lasso_regression = Lasso(alpha=0.0001, tol=0.1, selection="random", positive=False, max_iter=1000)
    lasso_regression.fit(X, y)
    quick_prediction_test(lasso_regression, X, y)
    y_prediction = lasso_regression.predict(X)
    evaluate_model_error(lasso_regression, y, y_prediction)
    plot_best_fit(y, y_prediction, "lasso_reg")
    save_model(lasso_regression, "lasso_reg")

    # Using the same model with 10-fold cross validation.
    k_fold_cross_validation(lasso_regression, X, y, folds=5)

    # Applying grid search algorithm to the Ridge Regression model
    parameters = {
        "alpha": [0.0001, 0.001, 0.01, 0.1, 0, 1, 10],
        "tol": [0.01, 0.1, 1],
        "positive": [True, False],
        "selection": ["cyclic", "random"]
    }
    lasso_reg_gs_results = grid_search_algorithm(Lasso(), parameters, folds=5)
    lasso_reg_gs_results_df = pd.DataFrame(lasso_reg_gs_results.cv_results_)
    lasso_reg_gs_results_df.to_csv("grid_search_resultslasso_reg_grid_search_results.csv")

    # Top 5 hyperparameters found for Lasso Regression.
    print("Top 5 hyperparameters combinations found for Lasso Regression:")
    lasso_reg_gs_results_df.sort_values(by=['rank_test_score']).head(5)

    # Best model found by grid search algorithm for Lasso Regression.
    final_lasso_reg_model = lasso_reg_gs_results.best_estimator_
    print("\nBest model hyperparameters found by grid search algorithm for Lasso Regression:")
    print(final_lasso_reg_model)


def elastic_net_regression(X, y):
    """

    :param X:
    :param y:
    :return:
    """
    elastic_net_regression = ElasticNet(alpha=0.1, tol=0.1, max_iter=10000)
    elastic_net_regression.fit(X, y)
    quick_prediction_test(elastic_net_regression, X, y)
    evaluate_model_error(elastic_net_regression, y, elastic_net_regression.predict(X))
    save_model(elastic_net_regression, "elasticnet_reg")
    k_fold_cross_validation(elastic_net_regression, X, y)


def decision_tree_regression(X, y):
    """

    :param X:
    :param y:
    :return:
    """
    tree_regressor = DecisionTreeRegressor()
    tree_regressor.fit(X, y)
    quick_prediction_test(tree_regressor, X, y)
    evaluate_model_error(tree_regressor, y, tree_regressor.predict(X))
    save_model(tree_regressor, "tree_reg")
    k_fold_cross_validation(tree_regressor, X, y)


def mlp_regression(X, y):
    """

    :param X:
    :param y:
    :return:
    """
    mlp_regressor = MLPRegressor()
    mlp_regressor.fit(X, y)
    quick_prediction_test(mlp_regressor, X, y)
    evaluate_model_error(mlp_regressor, y, mlp_regressor.predict(X))
    save_model(mlp_regressor, "mlp_reg")
    k_fold_cross_validation(mlp_regressor, X, y)


def svm_regression(X, y):
    """

    :param X:
    :param y:
    :return:
    """
    svr = SVR()
    svr.fit(X, y)
    quick_prediction_test(svr, X, y)
    evaluate_model_error(svr, y, svr.predict(X))
    save_model(svr, "svr")
    k_fold_cross_validation(svr, X, y)


def random_forest_generator_regression(X, y):
    """

    :param X:
    :param y:
    :return:
    """
    random_forest_generator_reg = RandomForestRegressor()
    random_forest_generator_reg.fit(X, y)
    quick_prediction_test(random_forest_generator_reg, X, y)
    evaluate_model_error(random_forest_generator_reg, y, random_forest_generator_reg.predict(X))
    save_model(random_forest_generator_reg, "random_forest_generator_reg")
    k_fold_cross_validation(random_forest_generator_reg, X, y)


def final_evaluation(test_set, final_model):
    """

    :param test_set:
    :param final_model:
    :return:
    """
    X_test = test_set.drop("critical_temp", axis=1)
    y_test = test_set["critical_temp"].copy()
    evaluate_model_error(final_model, y_test, final_model.predict(X_test))


def random_plot() -> None:
    """
    Small scatter plot to demonstrate stratified splitting in the report.
    :return: None
    """
    random.seed(0)
    x = [random.randint(0, 100) for i in range(0, 100)]
    y = [random.randint(0, 100) for i in range(0, 100)]
    plt.scatter(x, y)
    plt.savefig("plots/plot_stratified_example.png")
    plt.show()


if __name__ == "__main__":
    main()
