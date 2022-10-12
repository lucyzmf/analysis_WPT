from IPython import embed as shell
import os, sys, datetime, pickle
import scipy as sp
import numpy as np
import numpy.matlib
import matplotlib
import matplotlib.pylab as plt
from matplotlib.colors import hex2color, rgb2hex
import pandas as pd
import h5py
from scipy import stats, polyval, polyfit
import seaborn as sns
import math
from joblib import Parallel, delayed
 

from ideal_observer_fit import ideal_fit_data 
from cue_sign_learner_fit import cue_fit_data
from TTB_used_cue_update_fit import TTB_used_fit_data
from TTB_all_cue_update_fit import TTB_all_fit_data




##This script fits participant data to the models and find the best fit parameters of each model and the error for each participant 
##It then averages the results and decide based on BIC which model produces best fit

#import schedule, fit, calculate error given that fit, save error 

hdf5_path = '/Users/lucyzhang/Desktop/analysis/ready'
DF = pd.read_hdf(os.path.join(hdf5_path,"DF.h5"))
subjects = np.unique(DF['subject'].to_numpy())

hdf5_path = '/Users/lucyzhang/Desktop/analysis/model_fitting'

#d = pd.read_hdf(os.path.join(hdf5_path,"fitted_parameters.h5"),key ='ideal_observer')

###############################
#ideal observer
###############################
ideal_alphas = [] #output alphas
ideal_betas = [] #output betas 
ideal_errors = [] #output errors

def model_ideal(n):
	[alpha, beta, error] = ideal_fit_data(n)
	ideal_alphas.append(alpha)
	ideal_betas.append(beta)
	ideal_errors.append(error)


Parallel(n_jobs=2, verbose = 10)(delayed (model_ideal(n)) for n in subjects) #28

#build dataframe 
d = {'subject': subjects, 'alpha' : ideal_alphas, 'beta': ideal_betas, 'negLL' : ideal_errors}
df_ideal = pd.DataFrame (data = d) 
df_ideal.to_hdf('ideal_fitted_parameters.h5', key = 'ideal_observer', mode = 'w')




###############################
#cue sign learner
###############################
cue_alphas = []
cue_betas = []
cue_errors = []

def model_cue_sign(n):
	[alpha, beta, error] = cue_fit_data(n)
	cue_alphas.append(alpha)
	cue_betas.append(beta)
	cue_errors.append(error)
	print(n, 'done')

Parallel(n_jobs=2, verbose = 10)(delayed (model_cue_sign(n)) for n in subjects)

d = {'subject': subjects, 'alpha' : cue_alphas, 'beta': cue_betas, 'negLL' : cue_errors}
df_cue = pd.DataFrame (data = d) 
df_cue.to_hdf('cue_fitted_parameters.h5', key = 'cue_sign', mode = 'w')


###############################
#TTB used cue update
###############################
TTB_used_alphas = []
TTB_used_betas = []
TTB_used_k = []
TTB_used_errors = []
done = []

def model_TTB_used(n):
	[alpha, beta, k, error] = TTB_used_fit_data(n)
	TTB_used_alphas.append(alpha)
	TTB_used_betas.append(beta)
	TTB_used_k.append(k)
	TTB_used_errors.append(error)
	print(n, 'done')

Parallel(n_jobs=2, verbose = 10)(delayed (model_TTB_used(n)) for n in subjects)

d = {'subject': subjects, 'alpha' : TTB_used_alphas, 'beta': TTB_used_betas, 'k': TTB_used_k, 'negLL' : TTB_used_errors}
df_TTB_used = pd.DataFrame (data = d) 
df_TTB_used.to_hdf('fitted_parameters.h5', key = 'TTB_used_learner')


###############################
#TTB all cue update
###############################
TTB_all_alphas = []
TTB_all_betas = []
TTB_all_k = []
TTB_all_errors = []

for n in np.unique(DF['subject'].to_numpy()): 
	[alpha, beta, k, error] = TTB_all_fit_data(n)
	TTB_all_alphas.append(alpha)
	TTB_all_betas.append(beta)
	TTB_all_k.append(k)
	TTB_all_errors.append(error)
	print(n, 'done')

d = {'subject': np.unique(DF['subject'].to_numpy()), 'alpha' : TTB_all_alphas, 'beta': TTB_all_betas, 'k': TTB_all_k, 'negLL' : all_errors}
df_TTB_all = pd.DataFrame (data = d) 
df_TTB_all.to_hdf('fitted_parameters.h5', key = 'TTB_all_learner')



##############################################################
#model comparison
##############################################################
#BIC = -2 * LL + log(N) * k, k: num of parameters, N: num of examples in training set

# N = 800

# BIC_ideal = []; BIC_cue = []; k = 2
# for i in range(len(df_ideal)):
# 	BIC_ideal.append(2*df_ideal[df_ideal.subject==i]['negLL'] + math.log(N)*k)
# 	BIC_cue.append(2*df_cue[df_cue.subject==i]['negLL'] + math.log(N)*k)

# BIC_ideal = np.reshape(np.array(BIC_ideal), len(np.array(BIC_ideal)))
# BIC_cue = np.reshape(np.array(BIC_cue), len(np.array(BIC_cue)))
# Ideal_min_cue = BIC_ideal - BIC_cue


# fig, ax = plt.subplots()
# subject = range(len(df_ideal))
# ax.bar(subject, Ideal_min_cue)
# plt.show()








