# RAMP starting kit on the Acrobot dataset

## Getting started

The starting kit requires Python 3.7 and the following packages:

- numpy
- scipy
- scikit-learn
- pandas
- xarray
- jupyter
- pytorch
- altair
- ramp-workflow (see below to install this package as you will need a specific version)

Python 3.7 and all these packages (except [altair](https://altair-viz.github.io/getting_started/installation.html) and ramp-workflow) can be easily installed using the [Anaconda distribution](https://www.anaconda.com/distribution/).


### ramp-workflow installation
To install [ramp-workflow](https://github.com/paris-saclay-cds/ramp-workflow) please run the following command
```
pip install git+https://github.com/paris-saclay-cds/ramp-workflow.git@generative_regression_clean
```

An alternative solution is to clone the [ramp-workflow repository](https://github.com/paris-saclay-cds/ramp-workflow) by running
```
git clone https://github.com/paris-saclay-cds/ramp-workflow.git
```
(you can use SSH instead of HTTPS).

Then `cd` to the `ramp-workflow` folder and run
```
git checkout -b generative_regression_clean
pip install .
```

### Getting the starting kit

To get the starting kit with the notebook and the submission examples clone the [acrobot repository](https://github.com/ramp-kits/acrobot).
```
git clone https://github.com/ramp-kits/acrobot
```

To run the notebook, `cd` to the `acrobot` folder and run
```
jupyter notebook acrobot_starting_kit.ipynb
```
