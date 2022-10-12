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

from ideal_observer_sim import ideal_sim
from cue_sign_learner_sim import cue_sim
 
from sklearn.linear_model import LogisticRegression 
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.graphics.factorplots import interaction_plot



hdf5_path = '/Users/lucyzhang/Desktop/analysis/ready'
DF = pd.read_hdf(os.path.join(hdf5_path,"DF.h5"))
subjects = np.unique(DF['subject'].to_numpy())
condition = []
for i in subjects:
	if DF[DF.subject == i]['condition'].any() == 0: 
		condition.append('Curriculum')
	else:
		condition.append('Parallel')



### setting up data 
hdf5_path = '/Users/lucyzhang/Desktop/analysis/model_fitting'


DF_ideal_fitted = pd.read_hdf(os.path.join(hdf5_path,'ideal_fitted_parameters.h5'), key = 'ideal_observer')
DF_cue_fitted = pd.read_hdf(os.path.join(hdf5_path,'cue_fitted_parameters.h5'), key = 'cue_sign')
DF_TTB_used_fitted = pd.read_hdf(os.path.join(hdf5_path,'ttb_used_fitted_parameters.h5'))
DF_TTB_all_fitted = pd.read_hdf(os.path.join(hdf5_path,'ttb_all_fitted_parameters.h5'))


DF_ideal_fitted['condition'] = condition
DF_cue_fitted['condition'] = condition
DF_TTB_used_fitted['condition'] = condition
DF_TTB_all_fitted['condition'] = condition

###build data frame for regressino analysis per model 
##base model 
response_category = []
correct = []
for i in subjects:
	alpha = DF_ideal_fitted.iloc[i]['alpha']
	beta = DF_ideal_fitted.iloc[i]['beta']
	(response, corr) = ideal_sim(i, [alpha, beta])
	response_category += response
	correct += corr

d = {'subject': DF['subject'], 'response_category': response_category, 'correct': correct}
DF_base = pd.DataFrame(d)
DF_base['.1'] = DF['.1']
DF_base['.2'] = DF['.2']
DF_base['.3'] = DF['.3']
DF_base['.4'] = DF['.4']
DF_base['.6'] = DF['.6']
DF_base['.7'] = DF['.7']
DF_base['.8'] = DF['.8']
DF_base['.9'] = DF['.9']
DF_base['istest'] = DF['istest']




cue_reg_weights = []; cue_reg_intercept = []
for subj in range(len(subjects)):
	DF_slice = DF_base[DF_base.subject==subj][DF_base.istest==1]
	X = np.array(DF_slice[['.1', '.2', '.3', '.4', '.6', '.7', '.8', '.9']])
	y = np.array(DF_slice['response_category'])
	#print(subj)
	clf = LogisticRegression(penalty = 'l2', solver = 'lbfgs').fit(X,y)
	print(clf.intercept_, clf.coef_)
	cue_reg_intercept.append(clf.intercept_)
	cue_reg_weights.append(clf.coef_)


c = []
for i in range(N):
	tmp = np.unique(DF[DF.subject == i]['condition'].values)
	if  tmp[0] == 0:
		for k in range(9):
			c.append('Curriculum')
	else:
		for k in range(9):
			c.append('Parallel')


DF_reg_base = pd.DataFrame() 
DF_tmp = pd.DataFrame()
DF_tmp['cue'] = ['intercept','.1','.2','.3','.4','.6','.7','.8','.9']
for subj in range(N): 
	DF_tmp['subject']=subj
	DF_tmp['weights']=np.hstack((cue_reg_intercept[subj], np.squeeze(cue_reg_weights)[subj]))
	DF_reg_base=DF_reg_base.append(DF_tmp)
DF_reg_base['condition'] = c

logOdds = lambda x: np.log(x/(1-x)) # lambda indicates a simple function. this line is equivalent to: "def logOdds(x): <br><space> return np.log(x/(1-x))"
p = [.5, .1, .2, .3, .4, .6, .7, .8, .9]
w = logOdds(np.array(p)); w = np.abs(w); w = np.around(w, decimals = 3)
abs_woe = []
for i in range(int(len(DF_reg_base)/9)):
	for k in range(len(w)):
		abs_woe.append(w[k])
	
DF_reg_base['abs_woe'] = abs_woe
DF_reg_base['abs_weights'] = np.abs(DF_reg_base['weights'].values)

DF_tmp = DF_reg_base[DF_reg_base.cue != 'intercept']

model = ols('abs_weights ~ C(abs_woe) + C(condition) + C(abs_woe):C(condition)', data=DF_tmp).fit()
sm.stats.anova_lm(model, typ=2) 


##plot
fig = plt.figure(figsize = (8, 6))
sns.barplot(data = DF_reg_base, capsize = .05, x = 'abs_woe', y = 'abs_weights', hue = 'condition', ci = 95, palette='deep')
plt.xlabel('Absolute Ground Truth WOE')
plt.ylabel('Absolute Weight')
plt.legend(title = 'Condition')
plt.xticks(ticks = np.arange(5), labels = ['bias', '.41', '.85', '1.39', '2.20'])
sns.despine()
# plt.suptitle('Main effect of abs_woe: p = .00, main effect of condition: p = .04, no interaction')
# plt.show()
plt.savefig('regression cue weights by absolute values base model simulation')








###cue sign
response_category = []
correct = []
for i in subjects:
	alpha = DF_cue_fitted.iloc[i]['alpha']
	beta = DF_cue_fitted.iloc[i]['beta']
	(response, corr) = cue_sim(i, [alpha, beta])
	response_category += response
	correct += corr

d = {'subject': DF['subject'], 'response_category': response_category, 'correct': correct}
DF_cue = pd.DataFrame(d)
DF_cue['.1'] = DF['.1']
DF_cue['.2'] = DF['.2']
DF_cue['.3'] = DF['.3']
DF_cue['.4'] = DF['.4']
DF_cue['.6'] = DF['.6']
DF_cue['.7'] = DF['.7']
DF_cue['.8'] = DF['.8']
DF_cue['.9'] = DF['.9']
DF_cue['istest'] = DF['istest']


cue_reg_weights = []; cue_reg_intercept = []
for subj in range(len(subjects)):
	DF_slice = DF_cue[DF_cue.subject==subj][DF_cue.istest==1]
	X = np.array(DF_slice[['.1', '.2', '.3', '.4', '.6', '.7', '.8', '.9']])
	y = np.array(DF_slice['response_category'])
	#print(subj)
	clf = LogisticRegression(penalty = 'l2', solver = 'lbfgs').fit(X,y)
	print(clf.intercept_, clf.coef_)
	cue_reg_intercept.append(clf.intercept_)
	cue_reg_weights.append(clf.coef_)


DF_reg_cue = pd.DataFrame() 
DF_tmp = pd.DataFrame()
DF_tmp['cue'] = ['intercept','.1','.2','.3','.4','.6','.7','.8','.9']
for subj in range(N): 
	DF_tmp['subject']=subj
	DF_tmp['weights']=np.hstack((cue_reg_intercept[subj], np.squeeze(cue_reg_weights)[subj]))
	DF_reg_cue=DF_reg_cue.append(DF_tmp)
DF_reg_cue['condition'] = c

logOdds = lambda x: np.log(x/(1-x)) # lambda indicates a simple function. this line is equivalent to: "def logOdds(x): <br><space> return np.log(x/(1-x))"
p = [.5, .1, .2, .3, .4, .6, .7, .8, .9]
w = logOdds(np.array(p)); w = np.abs(w); w = np.around(w, decimals = 3)
abs_woe = []
for i in range(int(len(DF_reg_cue)/9)):
	for k in range(len(w)):
		abs_woe.append(w[k])
	
DF_reg_cue['abs_woe'] = abs_woe
DF_reg_cue['abs_weights'] = np.abs(DF_reg_cue['weights'].values)

DF_tmp = DF_reg_cue[DF_reg_cue.cue != 'intercept']

model = ols('abs_weights ~ C(abs_woe) + C(condition) + C(abs_woe):C(condition)', data=DF_tmp).fit()
sm.stats.anova_lm(model, typ=2) 

##plot
fig = plt.figure(figsize = (8, 6))
sns.barplot(data = DF_reg_cue, capsize = .05, x = 'abs_woe', y = 'abs_weights', hue = 'condition', ci = 95, palette='deep')
plt.xlabel('Absolute Ground Truth WOE')
plt.ylabel('Absolute Weight')
plt.legend(title = 'Condition')
plt.xticks(ticks = np.arange(5), labels = ['bias', '.41', '.85', '1.39', '2.20'])
sns.despine()
# plt.suptitle('Main effect of abs_woe: p = .00, main effect of condition: p = .04, no interaction')
# plt.show()
plt.savefig('regression cue weights by absolute values cue model simulation')





###TTB used 
from TTB_used_cue_update_sim import used_sim 

response_category = []
correct = []
for i in subjects:
	alpha = DF_TTB_used_fitted.iloc[i]['alpha']
	beta = DF_TTB_used_fitted.iloc[i]['beta']
	k = DF_TTB_used_fitted.iloc[i]['k']
	(response, corr) = used_sim(i, [alpha, beta, k])
	response_category += response
	correct += corr

d = {'subject': DF['subject'], 'response_category': response_category, 'correct': correct}
DF_used = pd.DataFrame(d)
DF_used['.1'] = DF['.1']
DF_used['.2'] = DF['.2']
DF_used['.3'] = DF['.3']
DF_used['.4'] = DF['.4']
DF_used['.6'] = DF['.6']
DF_used['.7'] = DF['.7']
DF_used['.8'] = DF['.8']
DF_used['.9'] = DF['.9']
DF_used['istest'] = DF['istest']


cue_reg_weights = []; cue_reg_intercept = []
for subj in range(len(subjects)):
	DF_slice = DF_used[DF_used.subject==subj][DF_used.istest==1]
	X = np.array(DF_slice[['.1', '.2', '.3', '.4', '.6', '.7', '.8', '.9']])
	y = np.array(DF_slice['response_category'])
	#print(subj)
	clf = LogisticRegression(penalty = 'l2', solver = 'lbfgs').fit(X,y)
	print(clf.intercept_, clf.coef_)
	cue_reg_intercept.append(clf.intercept_)
	cue_reg_weights.append(clf.coef_)


DF_reg_used = pd.DataFrame() 
DF_tmp = pd.DataFrame()
DF_tmp['cue'] = ['intercept','.1','.2','.3','.4','.6','.7','.8','.9']
for subj in range(N): 
	DF_tmp['subject']=subj
	DF_tmp['weights']=np.hstack((cue_reg_intercept[subj], np.squeeze(cue_reg_weights)[subj]))
	DF_reg_used=DF_reg_used.append(DF_tmp)
DF_reg_used['condition'] = c

logOdds = lambda x: np.log(x/(1-x)) # lambda indicates a simple function. this line is equivalent to: "def logOdds(x): <br><space> return np.log(x/(1-x))"
p = [.5, .1, .2, .3, .4, .6, .7, .8, .9]
w = logOdds(np.array(p)); w = np.abs(w); w = np.around(w, decimals = 3)
abs_woe = []
for i in range(int(len(DF_reg_used)/9)):
	for k in range(len(w)):
		abs_woe.append(w[k])
	
DF_reg_used['abs_woe'] = abs_woe
DF_reg_used['abs_weights'] = np.abs(DF_reg_used['weights'].values)

DF_tmp = DF_reg_used[DF_reg_used.cue != 'intercept']

model = ols('abs_weights ~ C(abs_woe) + C(condition) + C(abs_woe):C(condition)', data=DF_tmp).fit()
sm.stats.anova_lm(model, typ=2) 

###
from statsmodels.stats.multicomp import pairwise_tukeyhsd

pairwise_tukeyhsd(endog=DF_tmp['abs_weights'], groups=DF_tmp['abs_woe'], alpha=0.05)

##plot
fig = plt.figure(figsize = (8, 6))
sns.barplot(data = DF_reg_used, capsize = .05, x = 'abs_woe', y = 'abs_weights', hue = 'condition', ci = 68, palette='deep')
plt.xlabel('Absolute Ground Truth WOE')
plt.ylabel('Absolute Weight')
plt.legend(title = 'Condition')
plt.xticks(ticks = np.arange(5), labels = ['bias', '.41', '.85', '1.39', '2.20'])
sns.despine()
plt.suptitle('Regression Analysis of Response from TTB Used simulation (N = 79)')
plt.ylim(0, 1.8)
# plt.suptitle('Main effect of abs_woe: p = .00, main effect of condition: p = .04, no interaction')
# plt.show()
plt.savefig('regression cue weights by absolute values TTB used model simulation')








###TTB all 
from TTB_all_cue_update_sim import all_sim 

response_category = []
correct = []
for i in subjects:
	alpha = DF_TTB_all_fitted.iloc[i]['alpha']
	beta = DF_TTB_all_fitted.iloc[i]['beta']
	k = DF_TTB_all_fitted.iloc[i]['k']
	(response, corr) = all_sim(i, [alpha, beta, k])
	response_category += response
	correct += corr

d = {'subject': DF['subject'], 'response_category': response_category, 'correct': correct}
DF_all = pd.DataFrame(d)
DF_all['.1'] = DF['.1']
DF_all['.2'] = DF['.2']
DF_all['.3'] = DF['.3']
DF_all['.4'] = DF['.4']
DF_all['.6'] = DF['.6']
DF_all['.7'] = DF['.7']
DF_all['.8'] = DF['.8']
DF_all['.9'] = DF['.9']
DF_all['istest'] = DF['istest']


cue_reg_weights = []; cue_reg_intercept = []
for subj in range(len(subjects)):
	DF_slice = DF_all[DF_all.subject==subj][DF_all.istest==1]
	X = np.array(DF_slice[['.1', '.2', '.3', '.4', '.6', '.7', '.8', '.9']])
	y = np.array(DF_slice['response_category'])
	#print(subj)
	clf = LogisticRegression(penalty = 'none', solver = 'lbfgs').fit(X,y)
	print(clf.intercept_, clf.coef_)
	cue_reg_intercept.append(clf.intercept_)
	cue_reg_weights.append(clf.coef_)


DF_reg_all = pd.DataFrame() 
DF_tmp = pd.DataFrame()
DF_tmp['cue'] = ['intercept','.1','.2','.3','.4','.6','.7','.8','.9']
for subj in range(N): 
	DF_tmp['subject']=subj
	DF_tmp['weights']=np.hstack((cue_reg_intercept[subj], np.squeeze(cue_reg_weights)[subj]))
	DF_reg_all=DF_reg_all.append(DF_tmp)
DF_reg_all['condition'] = c

logOdds = lambda x: np.log(x/(1-x)) # lambda indicates a simple function. this line is equivalent to: "def logOdds(x): <br><space> return np.log(x/(1-x))"
p = [.5, .1, .2, .3, .4, .6, .7, .8, .9]
w = logOdds(np.array(p)); w = np.abs(w); w = np.around(w, decimals = 3)
abs_woe = []
for i in range(int(len(DF_reg_all)/9)):
	for k in range(len(w)):
		abs_woe.append(w[k])
	
DF_reg_all['abs_woe'] = abs_woe
DF_reg_all['abs_weights'] = np.abs(DF_reg_all['weights'].values)

DF_tmp = DF_reg_all[DF_reg_all.cue != 'intercept']

model = ols('abs_weights ~ C(abs_woe) + C(condition) + C(abs_woe):C(condition)', data=DF_tmp).fit()
sm.stats.anova_lm(model, typ=2) 

##plot
fig = plt.figure(figsize = (8, 6))
sns.barplot(data = DF_reg_all, capsize = .05, x = 'abs_woe', y = 'abs_weights', hue = 'condition', ci = 95, palette='deep')
plt.xlabel('Absolute Ground Truth WOE')
plt.ylabel('Absolute Weight')
plt.legend(title = 'Condition')
plt.xticks(ticks = np.arange(5), labels = ['bias', '.41', '.85', '1.39', '2.20'])
sns.despine()
# plt.suptitle('Main effect of abs_woe: p = .00, main effect of condition: p = .04, no interaction')
# plt.show()
plt.savefig('regression cue weights by absolute values TTB all model simulation')



###mix best fit
DF_AIC_best = pd.read_hdf(os.path.join(hdf5_path,'AIC_best.h5'), key = 'best fit')

DF_mix = pd.DataFrame()
for i in range(N):
	if DF_AIC_best.iloc[i]['Best Fit'] == 'Base Model':
		DF_mix = DF_mix.append(DF_reg_base[DF_reg_base.subject == i])
	elif DF_AIC_best.iloc[i]['Best Fit'] == 'Cue Sign':
		DF_mix = DF_mix.append(DF_reg_cue[DF_reg_cue.subject == i])
	elif DF_AIC_best.iloc[i]['Best Fit'] == 'TTB used':
		DF_mix = DF_mix.append(DF_reg_used[DF_reg_used.subject == i])
	else:
		DF_mix = DF_mix.append(DF_reg_all[DF_reg_all.subject == i])

DF_tmp = DF_mix[DF_mix.cue != 'intercept']

model = ols('abs_weights ~ C(abs_woe) + C(condition) + C(abs_woe):C(condition)', data=DF_tmp).fit()
sm.stats.anova_lm(model, typ=2) 

from statsmodels.stats.multicomp import pairwise_tukeyhsd

pairwise_tukeyhsd(endog=DF_tmp['abs_weights'], groups=DF_tmp['abs_woe'], alpha=0.05)

###mixed anova 
import pingouin as pg

aov = pg.mixed_anova(dv='abs_weights', within='abs_woe', between='condition', subject='subject', data=DF_tmp)
# Pretty printing of ANOVA summary
pg.print_table(aov)

posthocs = pg.pairwise_ttests(dv='abs_weights', within='abs_woe', between='condition', subject='subject', data=DF_tmp, padjust = 'holm', effsize = 'eta-square')
pg.print_table(posthocs)


fig = plt.figure(figsize = (8, 6))
sns.barplot(data = DF_mix, capsize = .05, x = 'abs_woe', y = 'abs_weights', hue = 'condition', ci = 68, palette='deep')
plt.suptitle('Cohort Regression Analysis of Response from Best Fit Model simulation')
plt.xlabel('Absolute Ground Truth WOE')
plt.ylabel('Absolute Weight')
plt.legend(title = 'Condition')
plt.xticks(ticks = np.arange(5), labels = ['bias', '.41', '.85', '1.39', '2.20'])
plt.ylim(0, 1.8)
sns.despine()
# plt.suptitle('Main effect of abs_woe: p = .00, main effect of condition: p = .04, no interaction')
# plt.show()
plt.savefig('regression cue weights by absolute values best fit simulation')



DF_tmp = pd.DataFrame()
for i in range(N):
	if DF_AIC_best.iloc[i]['Best Fit'] == 'Cue Sign':
		DF_tmp = DF_tmp.append(DF_reg_cue[DF_reg_cue.subject == i])

model = ols('abs_weights ~ C(abs_woe) + C(condition) + C(abs_woe):C(condition)', data=DF_tmp[DF_tmp.cue != 'intercept']).fit()
sm.stats.anova_lm(model, typ=2) 

fig = plt.figure(figsize = (8, 6))
sns.barplot(data = DF_tmp, capsize = .05, x = 'abs_woe', y = 'abs_weights', hue = 'condition', hue_order = ['Curriculum', 'Parallel'], ci = 68, palette='deep')
plt.suptitle('Regression Analysis of Response from Participants Best-fitted by Cue Sign(%.0f)'%(int(len(DF_tmp)/9)))
plt.xlabel('Absolute Ground Truth WOE')
plt.ylabel('Absolute Weight')
plt.legend(title = 'Condition')
plt.xticks(ticks = np.arange(5), labels = ['bias', '.41', '.85', '1.39', '2.20'])
sns.despine()
plt.ylim(0, 1.8)
# plt.show()
plt.savefig('best fitting cue regression analysis')
plt.close()





###side by side learning curve 
performance_mix = []

for i in range(N):
	if DF_AIC_best.iloc[i]['Best Fit'] == 'Base Model':
		correct = []
		# for m in range(50):
		alpha = DF_ideal_fitted.iloc[i]['alpha']
		beta = DF_ideal_fitted.iloc[i]['beta']
		(response, corr) = ideal_sim(i, [alpha, beta])
		correct.append(corr)
		performance_mix += list(np.mean(correct, axis = 0))
		print(len(np.mean(correct, axis = 0)))
	elif DF_AIC_best.iloc[i]['Best Fit'] == 'Cue Sign':
		correct = []
		# for m in range(50):
		alpha = DF_cue_fitted.iloc[i]['alpha']
		beta = DF_cue_fitted.iloc[i]['beta']
		(response, corr) = cue_sim(i, [alpha, beta])
		correct.append(corr)
		performance_mix += list(np.mean(correct, axis = 0))
		print(len(np.mean(correct, axis = 0)))
	elif DF_AIC_best.iloc[i]['Best Fit'] == 'TTB used':
		correct = []
		# for m in range(50):
		alpha = DF_TTB_used_fitted.iloc[i]['alpha']
		beta = DF_TTB_used_fitted.iloc[i]['beta']
		k = DF_TTB_used_fitted.iloc[i]['k']
		(response, corr) = used_sim(i, [alpha, beta, k])
		correct.append(corr)
		performance_mix += list(np.mean(correct, axis = 0))
		print(len(np.mean(correct, axis = 0)))
	else:
		correct = []
		# for m in range(50):
		alpha = DF_TTB_all_fitted.iloc[i]['alpha']
		beta = DF_TTB_all_fitted.iloc[i]['beta']
		k = DF_TTB_all_fitted.iloc[i]['k']
		(response, corr) = all_sim(i, [alpha, beta, k])
		correct.append(corr)
		performance_mix += list(np.mean(correct, axis = 0))
		print(len(np.mean(correct, axis = 0)))

d = {'performance': performance_mix}
df_block_acc = pd.DataFrame(d)
df_block_acc['subject'] = DF['subject'].values
df_block_acc['block'] = DF['block'].values
df_block_acc['condition'] = DF['condition']

###learning curve of cue sign model 
# p = []
# for i in range(N):
# 	# for m in range(50):
# 	alpha = DF_cue_fitted.iloc[i]['alpha']
# 	beta = DF_cue_fitted.iloc[i]['beta']
# 	(response, corr) = cue_sim(i, [alpha, beta])
# 	p += corr

# d = {'performance': p}
# df_cue_acc = pd.DataFrame(d)
# df_cue_acc['subject'] = DF['subject'].values
# df_cue_acc['block'] = DF['block'].values
# df_cue_acc['condition'] = DF['condition']




##plat 
sns.set_style('white')
fig = plt.figure(figsize=(6,4))
sns.lineplot(data = df_block_acc, x = 'block', y = 'performance', ci = 68, hue = 'condition', legend = False, palette = 'deep')
plt.legend(title = 'Condition', loc = 'upper right', labels = ['Curriculum', 'Parallel'])
plt.suptitle('Learning curves simulation')
plt.xlabel('Block')
plt.ylabel('Accuracy')
plt.axvline(x=8, alpha =.5, c="black", dashes=[10,10])
plt.ylim(0.5, 1)
plt.tight_layout()
# plt.show()
plt.savefig('learning curve simulations')






