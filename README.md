# RAMP starting kit for the Acrobot challenge

## Getting started

The starting kit requires Python 3.7 and the following packages:

- numpy
- scipy
- scikit-learn
- pandas
- xarray
- jupyter
- pytorch
- matplotlib
- altair (see below to install this package as you need to install specific dependencies)
- ramp-workflow (see below to install this package as you will need a specific version)

Python 3.7 and all these packages (except [altair](https://altair-viz.github.io/getting_started/installation.html) and ramp-workflow) can be easily installed using the [Anaconda distribution](https://www.anaconda.com/distribution/).

### altair installation

As we use altair in the starting kit notebook you need to install it with the required dependencies.
This can be done using conda
```
conda install -c conda-forge altair vega_datasets notebook vega
```
or using pip
```
pip install -U altair vega_datasets notebook vega
```
You can refer to the [altair installation documentation](https://altair-viz.github.io/getting_started/installation.html#quick-start-altair-notebook) for more information.

### ramp-workflow installation
For the purpose of this challenge we need to install a specific branch of [ramp-workflow](https://github.com/paris-saclay-cds/ramp-workflow). This can be done using pip
```
pip install git+https://github.com/paris-saclay-cds/ramp-workflow.git@generative_regression_clean
```

An alternative solution is to clone the [ramp-workflow repository](https://github.com/paris-saclay-cds/ramp-workflow) by running
```
git clone https://github.com/paris-saclay-cds/ramp-workflow.git
```
(you can use SSH instead of HTTPS). Then `cd` to the `ramp-workflow` folder and run
```
git checkout -b generative_regression_clean
pip install .
```

### Using the requirement.txt file
If you are using pip you can easily install all the required packages except pytorch with
```
pip install -r requirements.txt
```
To install pytorch you can follow the instructions available on the [pytorch website](https://pytorch.org/).

### Getting the starting kit

To get the starting kit with the notebook and the submission examples clone the [acrobot repository](https://github.com/ramp-kits/acrobot).
```
git clone https://github.com/ramp-kits/acrobot
```

To run the notebook, `cd` to the `acrobot` folder and run
```
jupyter notebook acrobot_starting_kit.ipynb
```
