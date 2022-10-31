# Higgs Boson Challenge 2022 (EPFL)
In the repository you can find the code we used for [Higgs Boson Challenge 2022](https://www.aicrowd.com/challenges/epfl-machine-learning-higgs), proposed as project 1 in ML course at EPFL [(CS-433)](https://www.epfl.ch/labs/mlo/machine-learning-cs-433/). 

## Team:
Our team (named `FGRxML`) is composed by:  
- Brioschi Riccardo: [@RiccardoBrioschi](https://github.com/RiccardoBrioschi)  
- D'Angeli Gabriele: [@gabrieledangeli](https://github.com/gabrieledangeli)  
- Di Gennaro Federico: [@FedericoDiGenanro](https://github.com/FedericoDiGennaro)   

With an accuracy of 0.836 and an F1-Score of 0.749 we placed in the top 10% of the teams (>200 teams).

# Project pipeline

## Environment:
We worked with `python3.8.5`. The library used to compute our ML model is `numpy` and the library used for visualization is `matplotlib`.

## Data
For the code to work properly, the datasets must be placed in a folder called `data` that must be in the same working directory as the notebooks in this repo. The `data` folder must contain the datasets named `train.csv` and `test.csv`.

## Description of notebooks
Here you can find what each file in the repo does. The order in which they are described follows the pipeline we used to obtain our results.
- `helpers.py`: implementation of  all the "support" functions used in others .py files.
- `gradients.py`: implementation of all the gradients used to train 6 methods.
- `costs.py`:  the cost functions used in the implementations of the 6 methods.
- `implementations.py`: actual implementation of all 6 methods.
- `preprocessing.py`: implementation of functions used to process the data. The majority of them were used in `features_engineering.py`
- `feature_engineering.py`: python file for feature engineering (feature selection, feature transformation,...).
- `crossvalidation.py`: implemetation of functions used to train the best hyperparameters for our model (ridge regression).
- `choosing_hyperparameters.ipynb`: notebook to select the best hyperparameters for our model (ridge regression).
- `dataset_splitting.py`: implementation of functions to have a local computation of accuracy of our model before the submission.
- `run.py`: it returns our predictions after using the selected models to predict test data.


