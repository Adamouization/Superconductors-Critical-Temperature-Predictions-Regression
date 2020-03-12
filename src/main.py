import argparse
import time

import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import config as config
from helpers import *
from regression_models import decision_tree_regression, elastic_net_regression, evaluate_model_error, \
    general_linear_regression, linear_lasso_regression, linear_ridge_regression, mlp_regression, \
    random_forest_generator_regression, svm_regression


def main() -> None:
    """
    Program entry point. Parses command line arguments to decide which agent to run.
    :return: None
    """
    parse_command_line_arguments()

    # The data is imported into a pandas DataFrame.
    df = pd.read_csv("data/train.csv")

    # Split Train/Test data
    train_set, test_set = split_data(df)

    if config.section == "data_vis":
        # We make a copy of the training set used for data exploration purposes without introducing the risk of
        # unvoluntarily harming the original training set.
        exploration_set = train_set.copy()
        # Visualising the data to gain insights.
        visualise_data(exploration_set)
        data_correlation(exploration_set)
    elif config.section == "train" or config.section == "test":
        X, y = input_preparation(train_set)  # Preparing the inputs and choosing a suitable subset.
        if config.section == "train":
            training_regression_models(X, y)  # Selecting and training a regression model.
        elif config.section == "test":
            final_model = load_model("ridge_reg")
            final_evaluation(test_set, final_model=final_model)
        else:
            print_error_message()
    else:
        print_error_message()

    # Function used to generate scatter plot used to show stratified sampling example in report.
    # random_plot()


def parse_command_line_arguments():
    """
    Parse command line arguments and save them in config.py.
    :return: None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--section",
                        required=True,
                        help="The section of the code to run. Must be either 'data_vis', 'train' or 'test'."
                        )
    parser.add_argument("-m", "--model",
                        default="linear",
                        help="The regression model to use for training. Must be either 'linear', 'ridge', 'lasso', "
                             "'elastic_net', 'decision_tree', 'mlp', 'svm' or 'random_forest_generator'."
                        )
    parser.add_argument("-g", "--gridsearch",
                        action="store_true",
                        default=False,
                        help="Include this flag to run the grid search algorithm to determine the optimal "
                             "hyperparameters for the regression model. Only works for linear regression with either"
                             "Ridge or Lasso regularisation."
                        )
    parser.add_argument("-d", "--debug",
                        action="store_true",
                        help="Include this flag additional print statements for debugging purposes."
                        )
    args = parser.parse_args()
    config.section = args.section
    config.model = args.model
    config.is_grid_search = args.gridsearch
    config.debug = args.debug


def split_data(data):
    """

    :param data:
    :return:
    """
    # Before even viewing the data to avoid data snooping, it is split between a training and a testing set.
    # A randomised 80%/20% split is used rather than stratified sampling because the dataset is large enough.
    # The random number generator's seed is set at a fixed value to ensure that it always generates the same shuffle
    # indices when re-running the code, ensuring reproducibility.
    train_set, test_set = train_test_split(data, test_size=0.2, random_state=config.RANDOM_SEED)

    # The train set should be 80% the size of the original, whereas the test set should 20%.
    # Now, only the train_set is used, while the test_set is set aside and forgotten about until the very end when
    # the model's performance will be evaluated.
    if config.debug:
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
    exploration_set.hist(bins=50, figsize=(25, 18))
    plt.savefig("plots/all_features_histogram.png")
    plt.show()


def data_correlation(exploration_set):
    """

    :param exploration_set:
    :return:
    """
    # The correlation matrix is computed to identify the correlation between each feature and the target value. The
    # closer it is to 1, the stronger the correlation is, and vice versa when the correlation is close to -1.
    # A correlation of 0 can be translated as a lack of linear correlation.
    correlation_matrix = exploration_set.corr()
    print("\n10 features with the strongest positive correlation with 'critical_temp':")
    print(correlation_matrix["critical_temp"].sort_values(ascending=False).head(11))
    print("\n10 features with the strongest negative correlation with 'critical_temp':")
    print(correlation_matrix["critical_temp"].sort_values(ascending=True).head(10))

    # The standard correlation coefficients of the target attribute and the 5 attributes with the highest correlation
    # are plotted in a scatter plot.
    features_with_highest_correlation_top_5 = [
        "critical_temp",
        "wtd_std_ThermalConductivity",
        "range_ThermalConductivity",
        "std_ThermalConductivity",
        "range_atomic_radius",
        "wtd_entropy_atomic_mass"
    ]
    # Correlation matrix of top 10 features.
    features_with_highest_correlation_top_10 = features_with_highest_correlation_top_5 + [
        "wtd_entropy_atomic_radius",
        "number_of_elements",
        "range_fie",
        "entropy_Valence",
        "wtd_std_atomic_radius"
    ]

    # Correlation matrix.
    correlation_heatmap(
        correlation_matrix.sort_values(
            'critical_temp', ascending=False
        ).head(11)[features_with_highest_correlation_top_10]
    )

    # Scatter plots to visualise correlation
    pd.plotting.scatter_matrix(exploration_set[features_with_highest_correlation_top_5], figsize=(25, 18))
    plt.savefig("plots/scatter_plot_features_with_highest_correlation.png")
    plt.show()

    # Because wtd_std_ThermalConductivity has the highest correlation with critical_temp, we can zoom in on the scatter
    # plot between these two variables.
    exploration_set.plot(kind="scatter", x="wtd_std_ThermalConductivity", y="critical_temp", alpha=0.1)
    plt.savefig("plots/scatter_plot_wtd_std_ThermalConductivity.png")
    plt.show()


def correlation_heatmap(corr) -> None:
    """
    Generates a heatmap for the correlation matrix.
    :param corr: the correlation matrix.
    :return: None.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(
        corr, cmap="YlGnBu", vmax=1.0, center=0, square=True, linewidths=.5, annot=True, cbar_kws={"shrink": .75}
    )
    fig.savefig("plots/top_correlation_heatmap.png")
    plt.show()


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
    if config.debug:
        print("Predictors X size = {}, Target y size = {}".format(X.shape, y.shape))

    # Standardisation of the input because features have very different values.
    X[X.columns] = StandardScaler().fit_transform(X[X.columns])

    return X, y


def training_regression_models(X, y):
    """
    Selecting and Training a regression model.
    The following errors are achieved by the paper when using multiple regression:
        * RMSE = 17.6K
        * R^2 = 0.74
    These results will be compared to our different models' results
    :param X:
    :param y:
    :return:
    """
    # Start recording time.
    start_time = time.time()

    # Parse which model to train.
    if config.model == "linear":
        print("\nTraining Linear Regression model:\n")
        general_linear_regression(X, y)
    elif config.model == "ridge":
        print("\nTraining Linear Regression model with Ridge regularisation:\n")
        linear_ridge_regression(X, y)
    elif config.model == "lasso":
        print("\nTraining Linear Regression model with Lasso regularisation:\n")
        linear_lasso_regression(X, y)
    elif config.model == "elastic_net":
        print("\nTraining Linear Regression model with Elastic Net regularisation:\n")
        elastic_net_regression(X, y)
    elif config.model == "decision_tree":
        print("\nTraining Decision Tree Regression model:\n")
        decision_tree_regression(X, y)
    elif config.model == "mlp":
        print("\nTraining Multi Layer Perceptron Regression model:\n")
        mlp_regression(X, y)
    elif config.model == "svm":
        print("\nTraining SVM Regression model:\n")
        svm_regression(X, y)
    elif config.model == "random_forest_generator":
        print("\nTraining Random Forest Generator Regression model:\n")
        random_forest_generator_regression(X, y)
    else:
        print_error_message()

    # Print training runtime.
    print_runtime(round(time.time() - start_time, 2))


def final_evaluation(test_set, final_model):
    """

    :param test_set:
    :param final_model:
    :return:
    """
    X_test = test_set.drop("critical_temp", axis=1)
    y_test = test_set["critical_temp"].copy()
    evaluate_model_error(final_model, y_test, final_model.predict(X_test))


if __name__ == "__main__":
    main()
