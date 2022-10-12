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
DF_TTB_all_fitted = pd.read_hdf(os.path.join(hdf5_path,'ttb_all_fitted_parameters.h5'))
DF_TTB_used_fitted = pd.read_hdf(os.path.join(hdf5_path,'ttb_used_fitted_parameters.h5'))


DF_ideal_fitted['condition'] = condition
DF_cue_fitted['condition'] = condition
DF_TTB_all_fitted['condition'] = condition
DF_TTB_used_fitted['condition'] = condition


DF_all = pd.concat([DF_ideal_fitted, DF_cue_fitted, DF_TTB_all_fitted, DF_TTB_used_fitted])

m1 = []; m2 = []; m3 = []; m4 = []
for i in subjects:
	m1.append('Base Model')
	m2.append('Cue Sign')
	m3.append('TTB Used')
	m4.append('TTB All')

DF_all['Model'] = np.concatenate((m1, m2, m4, m3))
DF_all = DF_all.reset_index()

##fill in missing data loc 300 (s 63), 306(s 69), 312 (s 75)



###### box plot of fitted parameters for each model 

##alpha
fig1, (ax1, ax2, ax3) = plt.subplots(ncols = 3, figsize = (12, 4))
sns.set_style('white')
plt.suptitle('Fitted Parameters by Models')
sns.violinplot(data = DF_all, ax = ax1, x = 'Model', y = 'alpha', hue = 'condition', inner = "box", cut = 0, palette = 'deep')
ax1.set_title('Fitted alpha by Model')
ax1.get_legend().remove()
sns.despine()

##beta
sns.violinplot(data = DF_all, ax = ax2, x = 'Model', y = 'beta', hue = 'condition', inner = "box", cut = 0, palette = 'deep')
ax2.set_title('Fitted beta by Model')
ax2.get_legend().remove()
sns.despine()
# plt.show()

##k
sns.violinplot(data = DF_all[158:], ax = ax3, x = 'Model', y = 'k', hue = 'condition', inner = "box", cut = 0, palette = 'deep')
ax3.set_title('Fitted k by Model')
ax3.legend(loc = 'lower right')
sns.despine()
plt.tight_layout()
# plt.show()
plt.savefig('Fitted Parameters by Models')




###model comparison cohort analysis 
AIC = []
BIC = []

aic = lambda k, negLL: 2*k + 2*negLL
bic = lambda k, negLL: k*np.log(400) + 2*negLL

for i in range(len(DF_all)):
	k = 2 if i < 158 else 3
	negLL = DF_all.iloc[i]['negLL']
	AIC.append(aic(k, negLL))
	BIC.append(bic(k, negLL))

d = {'Subjects': DF_all['subject'], 'Condition': DF_all['condition'], 'Model': DF_all['Model'], 'AIC': AIC, 'BIC': BIC}
DF_comparison = pd.DataFrame(d)


AIC_ideal = DF_comparison[DF_comparison.Model=='Base Model']['AIC'].sum()
AIC_cue = DF_comparison[DF_comparison.Model=='Cue Sign']['AIC'].sum()
AIC_used = DF_comparison[DF_comparison.Model=='TTB Used']['AIC'].sum()
AIC_all = DF_comparison[DF_comparison.Model=='TTB All']['AIC'].sum()

print(AIC_ideal, AIC_cue, AIC_all, AIC_used)


BIC_ideal = DF_comparison[DF_comparison.Model=='Base Model']['BIC'].sum()
BIC_cue = DF_comparison[DF_comparison.Model=='Cue Sign']['BIC'].sum()
BIC_used = DF_comparison[DF_comparison.Model=='TTB Used']['BIC'].sum()
BIC_all = DF_comparison[DF_comparison.Model=='TTB All']['BIC'].sum()

print(BIC_ideal, BIC_cue, BIC_all, BIC_used)


models = ['Base Model', 'Cue Sign', 'TTB All', 'TTB Used']
y1 = [AIC_ideal, AIC_cue, AIC_all, AIC_used]
y2 = [BIC_ideal, BIC_cue, BIC_all, BIC_used]

y1 = y1 - np.amin(y1)
y2 = y2 - np.amin(y2)

x = np.arange(len(models))
width = .35

fig4, ax = plt.subplots()
r1 = ax.bar(x - width/2,y1, width, label = 'AIC')
r2 = ax.bar(x + width/2,y2, width, label = 'BIC')

ax.set_ylabel('Delta AIC/BIC value from best fit model')
ax.set_title('Model Fitness Comparison')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()


# plt.show()
plt.savefig('Model Comparison AIC_BIC')




###sum of AIC by condition 
AIC_curr = [DF_comparison[DF_comparison.Model=='Base Model'][DF_comparison.Condition == 'Curriculum']['AIC'].sum(), 
			DF_comparison[DF_comparison.Model=='Cue Sign'][DF_comparison.Condition == 'Curriculum']['AIC'].sum(), 
			DF_comparison[DF_comparison.Model=='TTB All'][DF_comparison.Condition == 'Curriculum']['AIC'].sum(), 
			DF_comparison[DF_comparison.Model=='TTB Used'][DF_comparison.Condition == 'Curriculum']['AIC'].sum()]
min1 = np.amin(AIC_curr)
AIC_curr = AIC_curr - min1

AIC_para = [DF_comparison[DF_comparison.Model=='Base Model'][DF_comparison.Condition == 'Parallel']['AIC'].sum(), 
			DF_comparison[DF_comparison.Model=='Cue Sign'][DF_comparison.Condition == 'Parallel']['AIC'].sum(), 
			DF_comparison[DF_comparison.Model=='TTB All'][DF_comparison.Condition == 'Parallel']['AIC'].sum(), 
			DF_comparison[DF_comparison.Model=='TTB Used'][DF_comparison.Condition == 'Parallel']['AIC'].sum()]
min2 = np.amin(AIC_para)
AIC_para = AIC_para - min2


sns.set_style('white')
fig, (ax1, ax2) = plt.subplots(ncols = 2, figsize = (8, 3))
sns.despine()
plt.suptitle('Model Fitness Comparison by Condition')
ax1.set_ylabel('Delta AIC value from best fit model')
ax1.set_ylim(0, 900)

sns.barplot(x = models, y = AIC_curr, ax = ax1, color = '#4c72b0') 
ax1.set_title('Curriculum')
sns.barplot(x = models, y = AIC_para, ax = ax2, color = '#dd8452')
ax2.set_title('Parallel')
ax2.set_ylim(0, 900)

plt.tight_layout()

# plt.show()
plt.savefig('Model Fitness Comparison by Condition')






###model comparison cohort and individual fit count side by side visualisation
##using AIC and BIC
best_fit_AIC = []
value_AIC = []
for i in subjects:
	a = DF_comparison[DF_comparison.Subjects==i]
	value = list(a['AIC'].values)
	minimum = min(value); value_AIC.append(minimum)
	index = value.index(minimum)
	best_fit_AIC.append(a.iloc[index]['Model'])

d = {'Subjects': subjects, 'Condition': condition, 'Best Fit': best_fit_AIC, 'AIC': value_AIC}
df_AIC = pd.DataFrame(d)

# fig = plt.figure()
# sns.scatterplot(data = df_AIC, x = 'Subjects', y = 'AIC', hue = 'Best Fit')
# plt.show()
df_AIC.to_hdf('AIC_best.h5', key = 'best fit', mode = 'w')



best_fit_BIC = []
for i in subjects:
	a = DF_comparison[DF_comparison.Subjects==i]
	value = list(a['BIC'].values)
	minimum = min(value); index = value.index(minimum)
	best_fit_BIC.append(a.iloc[index]['Model'])

d = {'Subjects': subjects, 'Condition': condition, 'Best Fit': best_fit_BIC}
df_BIC = pd.DataFrame(d)



#########plotting
sns.set_style('white')
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize = (8, 6))
plt.title('Model Comparison')
sns.set(font_scale = 1)
sns.barplot(x = models, y = y1, ax = ax1, color = 'grey')
ax1.set_ylabel('Delta AIC from Best Fit Model')
ax1.set_title('AIC by Model')
ax1.set_xlabel('')
ax1.set_ylim(0, 600)
sns.histplot(data = df_AIC.sort_values(by='Best Fit'), x = 'Best Fit', ax = ax2, shrink=.8, hue = 'Condition', palette = 'deep', edgecolor="none", multiple="stack", alpha = .9, legend = False)
ax2.legend(title = 'Condition', loc = 'upper right', labels = ['Curriculum', 'Parallel'], facecolor = 'white', edgecolor = 'grey')
ax2.set_xlabel('Model Type')
ax2.set_title('Number of Best Fit Subjects per Model(AIC)')
ax2.set_ylim(0, 35)


sns.barplot(x = models, y = y2, ax = ax3, color = 'grey')
ax3.set_title('BIC by Model')
ax3.set_ylabel('Delta BIC from Best Fit Model')
ax3.set_xlabel('')
ax3.set_ylim(0, 600)
sns.histplot(data = df_BIC.sort_values(by='Best Fit'), x = 'Best Fit', ax = ax4, shrink=.8, hue = 'Condition', palette = 'deep', edgecolor="none", multiple="stack", alpha = .9, legend = False)
ax4.legend(title = 'Condition', loc = 'upper right', labels = ['Curriculum', 'Parallel'], facecolor = 'white', edgecolor = 'grey')
ax4.set_title('Number of Best Fit Subjects per Model(BIC)')
ax4.set_xlabel('Model Type')
ax4.set_ylim(0, 35)

sns.despine()

plt.tight_layout()

# plt.show()
plt.savefig('Cohort and Inidividual comparison')


#####individual fit for each model 
mean_fit_subj_AIC = []
for i in subjects: 
	mean_fit_subj_AIC.append(np.mean(DF_comparison[DF_comparison.Subjects == i]['AIC'].values))

mean_fit_subj_BIC = []
for i in subjects: 
	mean_fit_subj_BIC.append(np.mean(DF_comparison[DF_comparison.Subjects == i]['BIC'].values))

fit_base_AIC = DF_comparison[DF_comparison.Model == 'Base Model']['AIC'].values - mean_fit_subj_AIC
fit_cue_AIC = DF_comparison[DF_comparison.Model == 'Cue Sign']['AIC'].values - mean_fit_subj_AIC
fit_used_AIC = DF_comparison[DF_comparison.Model == 'TTB used']['AIC'].values - mean_fit_subj_AIC
fit_all_AIC = DF_comparison[DF_comparison.Model == 'TTB all']['AIC'].values - mean_fit_subj_AIC

fit_base_BIC = DF_comparison[DF_comparison.Model == 'Base Model']['BIC'].values - mean_fit_subj_BIC
fit_cue_BIC = DF_comparison[DF_comparison.Model == 'Cue Sign']['BIC'].values - mean_fit_subj_BIC
fit_used_BIC = DF_comparison[DF_comparison.Model == 'TTB used']['BIC'].values - mean_fit_subj_BIC
fit_all_BIC = DF_comparison[DF_comparison.Model == 'TTB all']['BIC'].values - mean_fit_subj_BIC

###AIC graph
sns.set_style('white')
fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows = 4, sharex = True, figsize = (8, 10))
sns.set(font_scale = 1)
plt.title('AIC per subject')
plt.ylabel('individual fitness per model - mean individual fitness across models')

sns.barplot(x = subjects, y = fit_base_AIC, ax = ax1, color = 'darkblue')
ax1.set_title('Base Model')
ax1.set_ylim(-150, 100)

sns.barplot(x = subjects, y = fit_cue_AIC, ax = ax2, color = 'darkgreen')
ax2.set_title('Cue Sign')
ax2.set_ylim(-150, 100)

sns.barplot(x = subjects, y = fit_used_AIC, ax = ax3, color = 'darkred')
ax3.set_title('TTB Used')
ax3.set_ylim(-150, 100)

sns.barplot(x = subjects, y = fit_all_AIC, ax = ax4, color = 'chocolate')
ax4.set_title('TTB all')
ax4.set_xlabel('subjects')
ax4.set_xticklabels('')
ax4.set_ylim(-150, 100)

sns.despine()
plt.tight_layout()
# plt.show()
plt.savefig('Fitness comparison across models per subject AIC')


###BIC graph
sns.set_style('white')
fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows = 4, sharex = True, figsize = (8, 10))
sns.set(font_scale = 1)
plt.title('BIC per subject')
plt.ylabel('individual fitness per model - mean individual fitness across models')

sns.barplot(x = subjects, y = fit_base_BIC, ax = ax1, color = 'darkblue')
ax1.set_title('Base Model')
ax1.set_ylim(-150, 100)

sns.barplot(x = subjects, y = fit_cue_BIC, ax = ax2, color = 'darkgreen')
ax2.set_title('Cue Sign')
ax2.set_ylim(-150, 100)

sns.barplot(x = subjects, y = fit_used_BIC, ax = ax3, color = 'darkred')
ax3.set_title('TTB Used')
ax3.set_ylim(-150, 100)

sns.barplot(x = subjects, y = fit_all_BIC, ax = ax4, color = 'chocolate')
ax4.set_title('TTB all')
ax4.set_xlabel('subjects')
ax4.set_xticklabels('')
ax4.set_ylim(-150, 100)

sns.despine()
plt.tight_layout()
# plt.show()
plt.savefig('Fitness comparison across models per subject BIC')



###delta between best and second fit models per subject 
##AIC
delta_subj_AIC = []
for i in subjects:
	AICs = DF_comparison[DF_comparison.Subjects == i]['AIC'].values 
	AICs = np.sort(AICs)
	delta_subj_AIC.append(AICs[3] - AICs[2])

fig = plt.figure(figsize = (8, 4))
sns.set(font_scale = 0.8)
sns.barplot(x = subjects, y = delta_subj_AIC, color = 'darkblue')
tempx = np.linspace(0, 79, 100)
temp = 10 + 0*tempx
plt.plot(tempx, temp, color = 'red')
plt.title('delta between best and second best fit model per subject (AIC)')
plt.xlabel('count(delta > 10) = %i' % np.count_nonzero(np.array(delta_subj_AIC) > 10))
# plt.show()
plt.savefig('delta between best and second best fit model per subject (AIC)')



##BIC
delta_subj_BIC = []
for i in subjects:
	BICs = DF_comparison[DF_comparison.Subjects == i]['BIC'].values 
	BICs = np.sort(BICs)
	delta_subj_BIC.append(BICs[3] - BICs[2])

fig = plt.figure(figsize = (8, 4))
sns.set(font_scale = 0.8)
sns.barplot(x = subjects, y = delta_subj_BIC, color = 'darkblue')
tempx = np.linspace(0, 79, 100)
temp = 10 + 0*tempx
plt.plot(tempx, temp, color = 'red')
plt.title('delta between best and second best fit model per subject (BIC)')
plt.xlabel('count(delta > 10) = %i' % np.count_nonzero(np.array(delta_subj_BIC) > 10))
# plt.show()
plt.savefig('delta between best and second best fit model per subject (BIC)')




###distribution of AIC across individuals 
df_AIC['Mean AIC'] = mean_fit_subj_AIC

fig = plt.figure(figsize = (8, 6))
sns.histplot(data = df_AIC, x = mean_fit_subj_AIC, multiple="stack", hue = 'Best Fit')
plt.title('AIC Mean Fitness Histogram')
# plt.show()
plt.savefig('AIC Mean Fitness Histogram')




###AIC graph
# fig = plt.figure()
# sns.set(font_scale = 1)
# plt.title('BIC per subject')

# sns.barplot(x = subjects, y = fit_base_BIC, color = 'darkblue', label = 'Base Model')

# sns.barplot(x = subjects, y = fit_cue_BIC, color = 'darkgreen', label = 'Cue Sign')

# sns.barplot(x = subjects, y = fit_used_BIC, color = 'darkred', label = 'TTB used')

# sns.barplot(x = subjects, y = fit_all_BIC, color = 'chocolate', label = 'TTB all')
# plt.legend()
# plt.show()



###AIC difference between TTB used and base model in relation to TTB used k 
AIC_used_minus_base = DF_comparison[DF_comparison.Model == 'TTB used']['AIC'].values - DF_comparison[DF_comparison.Model == 'Base Model']['AIC'].values
k_used = DF_all[DF_all.Model == 'TTB used']['k'].values 

AIC_used_minus_all_curr = DF_comparison[DF_comparison.Condition == 'Curriculum'][DF_comparison.Model == 'TTB used']['AIC'].values - DF_comparison[DF_comparison.Condition == 'Curriculum'][DF_comparison.Model == 'TTB all']['AIC'].values
AIC_used_minus_all_para = DF_comparison[DF_comparison.Condition == 'Parallel'][DF_comparison.Model == 'TTB used']['AIC'].values - DF_comparison[DF_comparison.Condition == 'Parallel'][DF_comparison.Model == 'TTB all']['AIC'].values

k_used_curr = DF_all[DF_all.condition == 'Curriculum'][DF_all.Model == 'TTB used']['k'].values 
k_used_para = DF_all[DF_all.condition == 'Parallel'][DF_all.Model == 'TTB used']['k'].values 


AIC_all_minus_base = DF_comparison[DF_comparison.Model == 'TTB all']['AIC'].values - DF_comparison[DF_comparison.Model == 'Base Model']['AIC'].values
k_all = DF_all[DF_all.Model == 'TTB all']['k'].values 

[r1, p1] = stats.spearmanr(k_used_curr, AIC_used_minus_all_curr)
[r2, p2] = stats.spearmanr(k_used_para, AIC_used_minus_all_para)

fig = plt.figure()
# sns.scatterplot(x = k_used, y = AIC_used_minus_base)
# sns.scatterplot(x = k_all, y = AIC_all_minus_base)
sns.scatterplot(x = k_used_curr, y = AIC_used_minus_all_curr, label = 'curr')
sns.scatterplot(x = k_used_para, y = AIC_used_minus_all_para, label = 'para')
plt.legend()
plt.yscale('symlog')
plt.xlabel('k')
plt.ylabel('')
plt.ylabel('AIC(TTB used) - AIC(TTB all)')
plt.suptitle('curr: r = %.2f, p = %.2f, para: r = %.2f, p = %.2f'%(r1, p1, r2, p2))
plt.show()
plt.savefig('AIC difference between TTB used and base model in relation to TTB used k')











####
#ideal observer
###

### distribution of all three variables 
fig, (ax1, ax2, ax3) = plt.subplots(3)
fig.suptitle('Distribution of fitted parameters ideal observer')
sns.histplot(data = DF_all[DF_all.Model == 'Base Model'], x = 'alpha', ax = ax1)
# ax1.set_title('alpha')
sns.histplot(data = DF_all[DF_all.Model == 'Base Model'], x = 'beta', ax = ax2)
# ax2.set_title('beta')
sns.histplot(data = DF_all[DF_all.Model == 'Base Model'], x = 'negLL', ax = ax3)
plt.tight_layout()
# ax3.set_title('error')
plt.savefig('distribution of fitted parameters ideal ideal observer')
# plt.show


### scatter plot of alpha and beta 
fig = plt.figure()
fig.suptitle('alpha beta error ideal observer')
sns.scatterplot(data = DF_ideal_fitted, x = 'alpha', y = 'beta', hue = 'negLL')
# sns.scatterplot(data = DF_ideal_fitted[DF_ideal_fitted.negLL > 540], x = 'alpha', y = 'beta', marker = 'X', color = 'red')
plt.title('error')
plt.savefig('alpha beta error scatter ideal observer')
plt.show()




####
#cue sign
###

### distribution of all three variables 
fig, (ax1, ax2, ax3) = plt.subplots(3)
fig.suptitle('Distribution of fitted parameters cue sign')
sns.histplot(data = DF_all[DF_all.Model == 'Cue Sign'], x = 'alpha', ax = ax1)
# ax1.set_title('alpha')
sns.histplot(data = DF_all[DF_all.Model == 'Cue Sign'], x = 'beta', ax = ax2)
# ax2.set_title('beta')
sns.histplot(data = DF_all[DF_all.Model == 'Cue Sign'], x = 'negLL', ax = ax3)
plt.tight_layout()
# plt.show()
# ax3.set_title('error')
plt.savefig('distribution of fitted parameters cue sign learner')
# plt.show


### scatter plot of alpha and beta 
fig = plt.figure()
fig.suptitle('alpha beta error cue sign')
sns.scatterplot(data = DF_cue_fitted, x = 'alpha', y = 'beta', hue = 'negLL')
# sns.scatterplot(data = DF_ideal_fitted[DF_ideal_fitted.negLL > 540], x = 'alpha', y = 'beta', marker = 'X', color = 'red')
plt.show()
plt.savefig('alpha beta error scatter cue sign')





####
#TTB used
###

### distribution of all three variables 
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2)
fig.suptitle('Distribution of fitted parameters TTB used')
sns.histplot(data = DF_all[DF_all.Model == 'TTB used'], x = 'alpha', ax = ax1)
# ax1.set_title('alpha')
sns.histplot(data = DF_all[DF_all.Model == 'TTB used'], x = 'beta', ax = ax2)
# ax2.set_title('beta')
sns.histplot(data = DF_all[DF_all.Model == 'TTB used'], x = 'k', ax = ax3, binwidth = .05)
sns.histplot(data = DF_all[DF_all.Model == 'TTB used'], x = 'negLL', ax = ax4)
fig.tight_layout()
# plt.show()
# ax3.set_title('error')
plt.savefig('distribution of fitted parameters TTB used')
# plt.show


### scatter plot of alpha and beta 
fig = plt.figure()
fig.suptitle('alpha beta k TTB used')
sns.scatterplot(data = DF_TTB_used_fitted, x = 'alpha', y = 'beta', hue = 'k', size = 'negLL')
# sns.scatterplot(data = DF_ideal_fitted[DF_ideal_fitted.negLL > 540], x = 'alpha', y = 'beta', marker = 'X', color = 'red')
plt.show()
plt.savefig('alpha beta error scatter cue sign')







####
#TTB all
###

### distribution of all three variables 
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2)
fig.suptitle('Distribution of fitted parameters TTB all')
sns.histplot(data = DF_all[DF_all.Model == 'TTB all'], x = 'alpha', ax = ax1)
# ax1.set_title('alpha')
sns.histplot(data = DF_all[DF_all.Model == 'TTB all'], x = 'beta', ax = ax2)
# ax2.set_title('beta')
sns.histplot(data = DF_all[DF_all.Model == 'TTB all'], x = 'k', ax = ax3, binwidth = .05)
sns.histplot(data = DF_all[DF_all.Model == 'TTB all'], x = 'negLL', ax = ax4)
fig.tight_layout()
# plt.show()
# ax3.set_title('error')
plt.savefig('distribution of fitted parameters TTB all')
# plt.show









###statistical analysis 
from scipy import stats

##################
###fitted parameters of TTB used 
used_para_k = DF_all[DF_all.Model == 'TTB Used'][DF_all.condition == 'Parallel']['k'].values
used_curr_k = DF_all[DF_all.Model == 'TTB Used'][DF_all.condition == 'Curriculum']['k'].values

stats.mannwhitneyu(used_para_k, used_curr_k)
#(statistic=570.5, pvalue=0.020628946114231467) 

used_para_alpha = DF_all[DF_all.Model == 'TTB used'][DF_all.condition == 'Parallel']['alpha'].values
used_curr_alpha = DF_all[DF_all.Model == 'TTB used'][DF_all.condition == 'Curriculum']['alpha'].values

stats.mannwhitneyu(used_para_alpha, used_curr_alpha)
#statistic=504.0, pvalue=0.003535449086652764 

used_para_beta = DF_all[DF_all.Model == 'TTB Used'][DF_all.condition == 'Parallel']['beta'].values
used_curr_beta = DF_all[DF_all.Model == 'TTB Used'][DF_all.condition == 'Curriculum']['beta'].values

stats.mannwhitneyu(used_para_beta, used_curr_beta)
#statistic=616.0, pvalue=0.05540498906270963, not significant 


##################
###fitted parameters of TTB all 
all_para_k = DF_all[DF_all.Model == 'TTB all'][DF_all.condition == 'Parallel']['k'].values
all_curr_k = DF_all[DF_all.Model == 'TTB all'][DF_all.condition == 'Curriculum']['k'].values

stats.mannwhitneyu(all_para_k, all_curr_k)
#statistic=755.5, pvalue=0.41072048927587973 , not significant

all_para_alpha = DF_all[DF_all.Model == 'TTB all'][DF_all.condition == 'Parallel']['alpha'].values
all_curr_alpha = DF_all[DF_all.Model == 'TTB all'][DF_all.condition == 'Curriculum']['alpha'].values

stats.mannwhitneyu(all_para_alpha, all_curr_alpha)
#statistic=421.0, pvalue=0.00022589286237756825 

all_para_beta = DF_all[DF_all.Model == 'TTB All'][DF_all.condition == 'Parallel']['beta'].values
all_curr_beta = DF_all[DF_all.Model == 'TTB All'][DF_all.condition == 'Curriculum']['beta'].values

stats.mannwhitneyu(all_para_beta, all_curr_beta)
#statistic=640.5, pvalue=0.08781107730621246, not significant 


##################
###fitted parameters of base model 
base_para_alpha = DF_all[DF_all.Model == 'Base Model'][DF_all.condition == 'Parallel']['alpha'].values
base_curr_alpha = DF_all[DF_all.Model == 'Base Model'][DF_all.condition == 'Curriculum']['alpha'].values

stats.mannwhitneyu(base_para_alpha, base_curr_alpha)
#statistic=496.0, pvalue=0.0027863549257456707

base_para_beta = DF_all[DF_all.Model == 'Base Model'][DF_all.condition == 'Parallel']['beta'].values
base_curr_beta = DF_all[DF_all.Model == 'Base Model'][DF_all.condition == 'Curriculum']['beta'].values

stats.mannwhitneyu(base_para_beta, base_curr_beta)
#statistic=633.5, pvalue=0.07618369114526292, not significant 



##################
###fitted parameters of cue sign
cue_para_alpha = DF_all[DF_all.Model == 'Cue Sign'][DF_all.condition == 'Parallel']['alpha'].values
cue_curr_alpha = DF_all[DF_all.Model == 'Cue Sign'][DF_all.condition == 'Curriculum']['alpha'].values

stats.mannwhitneyu(cue_para_alpha, cue_curr_alpha)
#statistic=729.0, pvalue=0.31359033901699573, not significant 

cue_para_beta = DF_all[DF_all.Model == 'Cue Sign'][DF_all.condition == 'Parallel']['beta'].values
cue_curr_beta = DF_all[DF_all.Model == 'Cue Sign'][DF_all.condition == 'Curriculum']['beta'].values

stats.mannwhitneyu(cue_para_beta, cue_curr_beta)
#statistic=644.0, pvalue=0.09346304774252545, not significant 


beta_p = DF_all[DF_all.condition == 'Parallel']['beta'].values
beta_c = DF_all[DF_all.condition == 'Curriculum']['beta'].values

stats.mannwhitneyu(beta_p, beta_c)



#statistic=8907.5, pvalue=5.877816236143913e-06 















