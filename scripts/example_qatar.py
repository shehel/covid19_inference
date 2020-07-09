# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # Example for one region (bundesland)
# %% [markdown]
# 
# The first thing we need to do is import some essential stuff. Theses have to be installed and are prerequisites.

# %%
import datetime
import time as time_module
import sys
import os 
import pickle

import urllib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.stats
import theano
import matplotlib
import pymc3 as pm
import theano.tensor as tt


# %%
try: 
    import covid19_inference as cov19
except ModuleNotFoundError:
    sys.path.append('..')
    import covid19_inference as cov19

from covid19_inference.plotting import *

# %% [markdown]
# ## Data retrieval
# 
# The next thing we want to do is load a dataset from somewhere. For now there are two different sources i.e. the robert koch institute and the johns hopkins university. We will chose the rki for now!

# In[91]
date_begin_data = datetime.datetime(2020,3,3)

# %%
#jhu = cov19.data_retrieval.JHU()
#It is important to download the dataset!
#One could also parse true to the constructor of the class to force an auto download
#jhu.download_all_available_data(); 

# (Old) JHU as source of data
#df = jhu.get_new_confirmed(country='Qatar', begin_date=date_begin_data)
#new_cases_obs = (df['confirmed'].values)

# In[99]
date_begin_data = datetime.datetime(2020,3,3)
df = pd.read_csv('../../covid19MLPredictor/data/covid_data.csv')
new_cases_obs = df['Number of New Positive Cases in Last 24 Hrs'].values
print (df)
# %%
diff_data_sim = 16 # should be significantly larger than the expected delay, in 
                   # order to always fit the same number of data points.
num_days_forecast = 10

prior_date_school_shutdown =  datetime.datetime(2020,3,10)
prior_date_border_closure =  datetime.datetime(2020,3,18)
prior_ramadan =  datetime.datetime(2020,4,23)
prior_date_mask_shopping =  datetime.datetime(2020,4,26)
prior_date_mask_compulsory = datetime.datetime(2020,5,17)
prior_date_phase1 = datetime.datetime(2020,6,15)
# List of change points
change_points = [dict(pr_mean_date_transient = prior_date_school_shutdown,
                      pr_sigma_date_transient = 6,
                      pr_median_lambda = 0.2,
                      pr_sigma_lambda = 1),
                 dict(pr_mean_date_transient = prior_date_border_closure,
                      pr_sigma_date_transient = 6,
                      pr_median_lambda = 1/8,
                      pr_sigma_lambda = 2),
                 dict(pr_mean_date_transient = prior_ramadan,
                      pr_sigma_date_transient = 6,
                      pr_median_lambda = 1/4,
                      pr_sigma_lambda=1),
                 dict(pr_mean_date_transient = prior_date_mask_shopping,
                      pr_sigma_date_transient = 10,
                      pr_median_lambda = 1/8/2,
                      pr_sigma_lambda = 1),
                 dict(pr_mean_date_transient = prior_date_mask_compulsory,
                      pr_sigma_date_transient = 10,
                      pr_median_lambda = 1/8/4,
                      pr_sigma_lambda = 1),
                 dict(pr_mean_date_transient = prior_date_phase1,
                      pr_sigma_date_transient = 6,
                      pr_median_lambda = 1/8/4,
                      pr_sigma_lambda = 1)]

# %% [markdown]
# Then, create the model:

# %%
params_model = dict(new_cases_obs = new_cases_obs[:],
                    data_begin = date_begin_data,
                    fcast_len = num_days_forecast,
                    diff_data_sim = diff_data_sim,
                    N_population = 2873728) 

# The model is specified in a context. Each function in this context has access to the model parameters set
with cov19.Cov19Model(**params_model) as model:
    # Create the an array of the time dependent infection rate lambda
    lambda_t_log = cov19.lambda_t_with_sigmoids(pr_median_lambda_0 = 0.4,
                                                change_points_list = change_points)
    
    # set prior distribution for the recovery rate
    mu = pm.Lognormal(name="mu", mu=np.log(1/8), sigma=0.2)

    # set prior distribution for the probability of testing of infectious cases
    prob_test = pm.Lognormal(name="prob_test", mu=np.log(0.45), sigma=0.2)
    pr_median_delay = 10
    
    # This builds a decorrelated prior for I_begin for faster inference. 
    # It is not necessary to use it, one can simply remove it and use the default argument 
    # for pr_I_begin in cov19.SIR
    prior_I = cov19.make_prior_I(lambda_t_log, mu, pr_median_delay = pr_median_delay)
    
    # Use lambda_t_log and mu to run the SIR model
    new_I_t = cov19.SIR(lambda_t_log,mu, pr_I_begin = prior_I, prob_test = prob_test)
    
    # Delay the cases by a lognormal reporting delay
    new_cases_inferred_raw = cov19.delay_cases(new_I_t, pr_median_delay=pr_median_delay, 
                                               pr_median_scale_delay=0.3)
    
    # Modulate the inferred cases by a abs(sin(x)) function, to account for weekend effects
    new_cases_inferred = cov19.week_modulation(new_cases_inferred_raw)
    
    # Define the likelihood, uses the new_cases_obs set as model parameter
    cov19.student_t_likelihood(new_cases_inferred)

# %% [markdown]
# ## MCMC sampling
start = pm.find_MAP(model=model)
# %%
trace = pm.sample(model=model, tune=500, draws=1000, init='advi+adapt_diag', start=start)

# %%
varnames = cov19.plotting.get_all_free_RVs_names(model)
num_cols = 5
num_rows = int(np.ceil(len(varnames)/num_cols))
x_size = num_cols * 2.5
y_size = num_rows * 2.5

fig, axes = plt.subplots(num_rows, num_cols, figsize = (x_size, y_size),squeeze=False)
i_ax = 0
for i_row, axes_row in enumerate(axes):
    for i_col, ax in enumerate(axes_row):
        if i_ax >= len(varnames):
            ax.set_visible(False)
            continue 
        else:
            cov19.plotting.plot_hist(model, trace, ax, varnames[i_ax], 
                                     colors=('tab:blue', 'tab:green'))
        if not i_col == 0:
            ax.set_ylabel('')
        if i_col == 0 and i_row == 0:
            ax.legend()
        i_ax += 1
fig.subplots_adjust(wspace=0.25, hspace=0.4)


# %%
# begin date of simulations which is set to 16 days before 
# first observation by default
sim_begin_data = date_begin_data - datetime.timedelta(days=diff_data_sim)

# end date including the forecasted dates
date_end_data = date_begin_data + datetime.timedelta(days=len(new_cases_obs)+num_days_forecast-1)

x = pd.date_range(date_begin_data,date_end_data)

# write forecasts to csv
df_write = pd.DataFrame({'DT': x, 'TOT':np.append(new_cases_obs, [0]*num_days_forecast),
                         'TOT_model': np.median(trace.new_cases, axis=0)})
df_write.TOT_model = df_write.TOT_model.astype(int)
print (df_write)

# write to heroku site repo for ingestion into the app
df_write.to_csv('../../covid19MLPredictor/data/model_output.csv', index=False);

df_write.to_csv('model_output.csv', index=False);
# In[51]:

# Draw plots
fig, _ = plot_cases(
    trace,
    new_cases_obs,
    date_begin_sim=sim_begin_data,
    diff_data_sim=16,
    start_date_plot=None,
    end_date_plot=None,
    ylim=3000,
    week_interval=2,
    colors=("tab:blue", "tab:orange"),
    country="Qatar",
)
fig.savefig("forecastplot.png")


# %%

# Write to heroku site repo for ingestion into the app for plotting
with open('../../covid19MLPredictor/data/trace_new_cases.pkl', 'wb') as handle:
    pickle.dump(trace.new_cases, handle)
with open('../../covid19MLPredictor/data/trace_lambda.pkl', 'wb') as handle:
    pickle.dump(trace.lambda_t, handle)
with open('../../covid19MLPredictor/data/trace_mu.pkl', 'wb') as handle:
    pickle.dump(trace.mu, handle)



# %%
