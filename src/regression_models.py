import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR

import config as config
from helpers import save_model


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
    print("\n{} scores: {}".format(metric_name, scores))
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
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("{} best fit".format(fig_name))
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

    if config.is_grid_search:
        print("\nPerforming grid search to find optimal hyperparameters...")

        # Applying grid search algorithm to the Ridge Regression model to explore combinations of hyperparameters to
        # find the best hyperparameters for a model, since doing so manually is a very time consuming task.
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

    if config.is_grid_search:
        print("\nPerforming grid search to find optimal hyperparameters...")

        # Applying grid search algorithm to the Lasso Regression model
        parameters = {
            "alpha": [0.0001, 0.001, 0.01, 0.1, 0, 1, 10],
            "tol": [0.01, 0.1, 1],
            "positive": [True, False],
            "selection": ["cyclic", "random"]
        }
        lasso_reg_gs_results = grid_search_algorithm(Lasso(), parameters, folds=5)
        lasso_reg_gs_results_df = pd.DataFrame(lasso_reg_gs_results.cv_results_)
        lasso_reg_gs_results_df.to_csv("grid_search_results/lasso_reg_grid_search_results.csv")

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
    elastic_net_regression = ElasticNet()
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
