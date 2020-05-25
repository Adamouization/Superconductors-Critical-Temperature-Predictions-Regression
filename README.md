# Predicting Superconductors' Critical Temperatures using Regression

This project covers the exploration of various ML techniques used for predicting the critical temperatures required for different superconductors to conduct electrical current with no resistance [1]. Various techniques and decisions are considered and discussed in the [report](https://github.com/Adamouization/ML-Predicting-Superconductivity-Critical-Temperature/blob/master/report/report.pdf) for visualising & analysing the data, extracting its features, training multiple regression models to evaluate which one is suits the data best and make final predictions compared to Hamidieh's findings in the papaer *"A data-driven statistical model for predicting the critical temperature of a superconductor"*.

You can read the [full report here](https://github.com/Adamouization/ML-Predicting-Superconductivity-Critical-Temperature/blob/master/report/report.pdf).

## Results

An improvement is noticed between the linear regression's final results and Hamidieh’s results [1], as the the RMSE is improved by 0.22K, while the R2 score remained the same:

* RMSE: 17.38K
* R2: 0.74

<p align="center">
  <img src="https://raw.githubusercontent.com/Adamouization/ML-Predicting-Superconductivity-Critical-Temperature/master/plots/plot_final_ridge_reg.png?token=AEI7XLE4AXQWEDFBW7CHXBC62TLSO" alt="results" width="50%"/>
</p>

## Usage

Create a new virtual environment and install the Python libraries used in the code by running the following command:

```
pip install -r requirements.txt
```

To run the program, move to the “src” directory and run the following command:

```
python main.py -s <section> [-m <model>] [-g] [-d]
```

where:

* *"-s section"*: is a setting that executes different parts of the program. It must be one of the following: `data_vis`, `train` or `test`.

* *"-m model"*: is an optional setting that selects the regression model to use for training. It must be one of the following: `linear`, `ridge`, `lasso`, `elastic_net`, `decision_tree`, `mlp`, `svm` or `random_forest_generator`.

* *"-g"*: is an optional flag the grid search algorithm to determine the optimal hyperparameters for the selected regression model. The flag only takes effect when using linear regression with either Ridge or Lasso regularisation.

* *"-d"*: is an optional flag that enters debugging mode, printing additional statements on the command line.


## License 
* see [LICENSE](https://github.com/Adamouization/ML-Predicting-Superconductivity-Critical-Temperature/blob/master/LICENSE) file.

## Contact
* Email: adam@jaamour.com
* Website: www.adam.jaamour.com
* LinkedIn: [linkedin.com/in/adamjaamour](https://www.linkedin.com/in/adamjaamour/)
* Twitter: [@Adamouization](https://twitter.com/Adamouization)

# References

[1] K. Hamidieh, “A data-driven statistical model for predicting the critical temperature of a superconductor,” Computational Materials Science, vol. 154, pp. 346–354, Nov. 2018.
