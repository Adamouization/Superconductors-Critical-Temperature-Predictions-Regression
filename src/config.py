"""
Variables set by the command line arguments dictating which parts of the program to execute.
"""

# Constants
RANDOM_SEED = 42

# Variables set by command line arguments/flags
section = "data_vis"    # The section of the code to run.
model = "linear"        # The regression model to train/test.
is_grid_search = False  # Run the grid search algorithm to determine the optimal hyperparameters for the model.
debug = False           # Boolean used to print additional logs for debugging purposes.
