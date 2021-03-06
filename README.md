# Bayesian inference and forecast of COVID-19, code repository

## Model for Qatar
See [here](scripts/example_qatar.py) for the code used to create the model and generate the plot used in [covid19-qatar.herokuapp.com](covid19-qatar.herokuapp.com). This repo's outputs are set for feeding into the [web app](https://github.com/A1337CBS/covid19MLPredictor). Input csv file can be obtained by running 
 ```   
    python pages/covid_parser.py
```
inside the web app directory. Run 
```    
    python scripts/example_qatar.py 
```
for the plots and forecasts.


## Original Repo
[![Documentation Status](https://readthedocs.org/projects/covid19-inference/badge/?version=latest)](https://covid19-inference.readthedocs.io/en/latest/doc/gettingstarted.html)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This is a Bayesian python toolbox for inference and forecast of the spread of the Coronavirus.

Check out our [documentation](https://covid19-inference.readthedocs.io/en/latest/doc/gettingstarted.html).

An example notebook for one bundesland is [here](scripts/example_one_bundesland.ipynb), and for an hierarchical analysis of the bundeslaender [here](scripts/example_bundeslaender.ipynb) (could still have some problems).

The research article [is available on arXiv](https://arxiv.org/abs/2004.01105) (**updated on April 13**).
The code used to produce the figures is available in the other repository [here](https://github.com/Priesemann-Group/covid19_inference_forecast)


**We are looking for support** to help us with analyzing other countries and to extend to an hierarchical regional model. We have received additional funding to do so and are recruiting PostDocs, PhD candidates and research assistants:
https://www.ds.mpg.de/3568943/job_full_offer_14729553
https://www.ds.mpg.de/3568926/job_full_offer_14729572
https://www.ds.mpg.de/3568909/job_full_offer_14729591

### Please take notice of our [disclaimer](DISCLAIMER.md).


