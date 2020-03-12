import joblib
import random

import matplotlib.pyplot as plt


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


def print_runtime(runtime: float) -> None:
    """
    Print runtime in seconds.
    :param runtime: the runtime
    :return: None
    """
    print("\n--- Runtime: {} seconds ---".format(runtime))


def print_error_message():
    """

    :return:
    """
    print("Wrong command line arguments passed, please use 'python main.py --help' for instructions on which arguments"
          "to pass to the program.")
    exit(1)


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
