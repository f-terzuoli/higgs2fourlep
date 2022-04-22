# higgs2fourlep
[![Documentation Status](https://readthedocs.org/projects/higgs-analysis/badge/?version=latest)](https://higgs-analysis.readthedocs.io/en/latest/?badge=latest)

This package provides the modules for analysing the Higgs dibosonic decay (ZZ) to four leptons final products, with the ATLAS OpenData sets.

While the `processData` module offers the tools for selecting the signal via selection cuts, the `XGBoostHiggs` module deploys Machine Learning for separating background from signal.

With this package you can plot numerous kinematic distributions of the decay, study how different selection affects te distribution and the fractions of the different components, and also perform a measurement of the Higgs mass.

The module `loadconfig` helps in setting the configuration parameters for all the analysis workflow.

The `main.py` file offers an example of how the modules can be deployed.

The `.json` and `.yaml` files offers the template for the configuration file, in the form and structures that the modules expect.

Running `source setup.sh` results in appending the package root folder to the `PYTHONPATH` so that the modules can be accessed from every location on your system, within your current shell. 

The project makes use of the package `pyROOT` and depends on `matplotlib`, `munch`, `PyYAML`, `sklearn` and `XGBoost`.

Read the documentation at https://higgs-analysis.readthedocs.io/
