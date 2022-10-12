from IPython import embed as shell
import os, sys, datetime, pickle

import scipy as sp
import numpy as np
import numpy.matlib
import matplotlib.pylab as plt
from matplotlib.colors import hex2color, rgb2hex
import pandas as pd
import itertools 
from lmfit import minimize, Parameters, Parameter, report_fit
from scipy import stats, polyval, polyfit

import os
# pip install statsmodels
import statsmodels

import seaborn as sns
import h5py

pre_process_data = True
generate_dataframe = True
make_DF = True
do_reject = True

data_path = '/Users/lucyzhang/Desktop/analysis/raw'
hdf5_path = '/Users/lucyzhang/Desktop/analysis/ready'
project_name = 'WPCurr' # no spaces - will become your hdf5 file name
plotting_dir = '/Users/lucyzhang/Desktop/analysis/images'

data_files = os.listdir(data_path)

data, sdata_group, edata_group, parameters_group = [],[],[],[]
hdfloc = os.path.join(hdf5_path,project_name+".h5")

## Pre-processing
if pre_process_data:
	for file in data_files:
		currentfile = open(os.path.join(data_path,file), "r+")
		temp = currentfile.read().replace('sdata :', 'sdata =').replace('edata :', 'edata =').replace('parameters :', 'parameters =').replace('false','False') # makes the file evaluatable by python
		currentfile.seek(0); currentfile.write(temp) #overwrite old txt

if generate_dataframe:
	null = None
	for file in data_files:
		exec(open(os.path.join(data_path,file)).read())
		sdata_group.append(sdata), edata_group.append(edata), parameters_group.append(parameters)
sdata_group = pd.DataFrame.from_dict(sdata_group); edata_group = pd.DataFrame.from_dict(edata_group); parameters_group = pd.DataFrame.from_dict(parameters_group); 

accs = np.array([np.nanmean(sdata_group.resp_correct_prob[subj][-400:]) for subj in range(len(sdata_group))])
accs2 = accs>.54
#RTs_rejec = np.array([sdata_group.resp_reactiontime[subj][-400:] for subj in range(len(sdata_group))])
#RTs2 = np.nansum(RTs_rejec<.3, axis=1)
#RTs3 = RTs2<50
extra_key = []
for s in range(len(sdata_group)):
    try:
        for i in range(len(sdata_group['betwTrialPressPerTrial'][s])):
            if sdata_group['betwTrialPressPerTrial'][s][i] == None:
                sdata_group['betwTrialPressPerTrial'][s][i] = np.nan
        extra_key.append(np.nansum(sdata_group['betwTrialPressPerTrial'][s]))
    except:
        extra_key.append(np.nan)

pd.DataFrame.from_dict(sdata_group).to_hdf(hdfloc, 'sdata')
pd.DataFrame.from_dict(edata_group).to_hdf(hdfloc, 'edata')
pd.DataFrame.from_dict(parameters_group).to_hdf(hdfloc, 'parameters')

c1=0
c2=0 

for n in range(len(accs)): #count acc>.54
	if accs2[n]:
		c1 +=1
print('number of acc>.54', c1)

for n in range(len(accs)): #count acc<.54
	if not accs2[n]:
		c2 +=1 
print('number of acc<.54', c2)



#to load data
s = pd.read_hdf(hdfloc, 'sdata')
e = pd.read_hdf(hdfloc, 'edata')
p = pd.read_hdf(hdfloc, 'parameters') 

#print out index, acc, and RT
# for i in range(len(accs)):  
# 	print ('%i, %.2f, %i'%(i,accs[i],RTs2[i]))

N = len(s)
print("No. of Participants:", N)

c0idx = np.array([i for i in range(N) if np.array(parameters_group['cond'] == 0)[i]]) 
c1idx = np.array([i for i in range(N) if np.array(parameters_group['cond'] == 1)[i]])

print("Total Curriculum: ", len(c0idx))
print("Total Parallel: ", len(c1idx))

# keypress screening
# len_tmp = [np.shape(i) for i in sdata_group['resp_correct']]
betw_keys_filtered = [] # remove None entries
for subjd in sdata_group['betwTrialPressPerTrial']:
	betw_keys_filtered.append([i for i in np.ravel([subjd]) if i is not None])
betw_keys_filtered = np.array(betw_keys_filtered)

# whether keys were logged using the new method, which distinguishes between pre-stim onset and post-stim onset key presses (all but first 8)
#if 0, then old method, if not 0 new method
keys_logged_bool = [1-np.sum(np.isnan(i)) for i in betw_keys_filtered] 

# for i in range(N):
# 	print(i, keys_logged_bool[i])

#preonset keys calculated using (total number of extraneous key presses)-(post onset keys)
pre_onset_keys = np.zeros(N); post_onset_all = [];
for i in range(N): 
	if keys_logged_bool[i]: # first new method subjects
		post_onset_tmp = [np.nansum(sdata_group['betwTrialKeyPress'][i][str(trial)]) for trial in range(len(sdata_group['betwTrialKeyPress'][i]))]
		post_onset_all.append(np.nansum(post_onset_tmp))
		pre_onset_tmp = np.array(betw_keys_filtered[i][:800]) - np.array(post_onset_tmp)
		pre_onset_keys[i] = np.nansum(pre_onset_tmp)


for i in range(N): 
	if not keys_logged_bool[i]: # now old method subjects
		tmp = [parameters_group['betwTrialKeyPress'][i][str(block)] for block in range(len(parameters_group['betwTrialKeyPress'][i]))]
# post_onset_sum.append(np.nansum(post_onset_tmp))
# pre_onset_tmp = np.nansum(parameters_group['betwTrialKeyPress'][i]) - np.nansum(post_onset_all)
		pre_onset_keys[i] = max([0, np.nansum(tmp[:16]) - np.nanmean(post_onset_all)])


# longest monotonic response sequence
mono_resp = []
for i in range(len(accs)):
	mono_resp.append(max(sum(1 for x in v) for _,v in itertools.groupby(sdata_group['resp_category'][i])))

#print out corresponding preonset 
# for i in range(N):
# 	print(i, "pre onset:", pre_onset_keys[i], "Max monotonic resp:", mono_resp[i])

# keep = np.logical_or((pre_onset_keys<400), (accs>.7)).astype("int") + (np.array(z_accs)>-1).astype("int")
keep = np.logical_or((pre_onset_keys<1e4), (accs>.7)).astype("int") + (np.array(mono_resp)<26).astype("int")
keep = keep == 2

edata_group = edata_group[keep]
print(edata_group.index)
parameters_group = parameters_group[keep]
sdata_group = sdata_group[keep] 

edata_group=edata_group.reset_index(); 
sdata_group=sdata_group.reset_index(); 
parameters_group=parameters_group.reset_index(); 

###################################
#build data frame
###################################
istest = np.zeros(800).astype('int')
istest[-400:]=1

istrain = np.zeros(800).astype('int')
istrain[-800:-400]=1

test_ts = []; train_ts = []; all_ts = [];
test_ts_bin10 = []; train_ts_bin5 = [];train_ts_bin10 = []; train_ts_bin25 = []; 
test_rs_bin10, train_rs_bin25 = [],[]
train_ts_bin50, test_ts_bin50 = [],[]
train_RT_ts, train_RT_ts_bin50, test_RT_ts, test_RT_ts_bin50 = [],[],[],[]

# print(type(edata_group['expt_subject']))
# print(edata_group['expt_subject'])

# for subj in edata_group['expt_subject']:
# 	print(subj)

# for subj,x in edata_group['expt_subject'].items(): 
# 	print(edata_group['expt_subject'][subj])



for participant in s.index.values:
	train_temp_resp= np.array( [ s['resp_category'][participant][-(800-i)] for i in range(len(istest)) if istrain[-(800-i)] ] ) 
	test_temp_resp = np.array( [ s['resp_category'][participant][-(800-i)] for i in range(len(istest)) if istest[-(800-i)] ] ) 
	all_temp_resp = np.array( [ s['resp_category'][participant][-(800-i)] for i in range(len(istest)) ] ) 
	train_temp = np.array( [ s['resp_correct_prob'][participant][-(800-i)] for i in range(len(istest)) if istrain[-(800-i)] ] ) 
	test_temp = np.array( [ s['resp_correct_prob'][participant][-(800-i)] for i in range(len(istest)) if istest[-(800-i)] ] ) 
	all_temp = np.array( [ s['resp_correct_prob'][participant][-(800-i)] for i in range(len(istest)) ] ) 
	train_temp = train_temp.astype('float'); train_temp[train_temp_resp==None]=0 # you can either code these as np.nan or as 0.5
	test_temp = test_temp.astype('float'); test_temp[test_temp_resp==None]=0
	all_temp = all_temp.astype('float'); all_temp[all_temp_resp==None]=0
	train_temp_RT= np.array( [ s['resp_reactiontime'][participant][-(800-i)] for i in range(len(istest)) if istrain[-(800-i)] ] ) 
	test_temp_RT = np.array( [ s['resp_reactiontime'][participant][-(800-i)] for i in range(len(istest)) if istest[-(800-i)] ] ) 

	train_RT_ts.append(train_temp_resp)
	test_RT_ts.append(test_temp_RT)
	train_ts.append(train_temp)
	test_ts.append(test_temp)
	all_ts.append(all_temp)
	train_ts_bin10.append(np.nanmean(train_temp.reshape(-1,10),axis=1))
	test_ts_bin10.append(np.nanmean(test_temp.reshape(-1,10),axis=1))
	train_ts_bin25.append(np.nanmean(train_temp.reshape(-1,25),axis=1))
	train_ts_bin5.append(np.nanmean(train_temp.reshape(-1,5),axis=1))

	train_temp_r 	= np.array( [ s['resp_reactiontime'][participant][i] for i in range(len(istest)) if istrain[i] ] ) 
	test_temp_r 	= np.array( [ s['resp_reactiontime'][participant][i] for i in range(len(istest)) if istest[i] ] ) 
	test_rs_bin10.append(np.mean(test_temp_r.reshape(-1,10),axis=1))	
	train_rs_bin25.append(np.mean(train_temp_r.reshape(-1,25),axis=1))
	# test_ts_bin20.append(np.mean(test_temp.reshape(-1,20),axis=1))
	train_ts_bin50.append(np.mean(train_temp.reshape(-1,50),axis=1))
	test_ts_bin50.append(np.mean(test_temp.reshape(-1,50),axis=1))
	train_RT_ts_bin50.append(np.mean(train_temp_RT.reshape(-1,50),axis=1))
	test_RT_ts_bin50.append(np.mean(test_temp_RT.reshape(-1,50),axis=1))

make_DF = True 
if make_DF:
	DF = pd.DataFrame()
	blocks = np.ravel(np.matlib.repmat(np.arange(1,17),50,1).T)

	subj_array = [];
	cond_array = [];
	bl_array = [];
	trial_nr_array = [];
	param_rel_array = [];
	correct_array = [];
	resp_category_array = [];
	resp_trial_nr_array = [];
	rewarded_outcome_array = []
	blocks_array = [];
	istest_array = []; 
	istrain_array = [];
	RT_array = [];
	point1_array, point2_array, point3_array, point4_array, point6_array, point7_array, point8_array, point9_array, nr_stims_array = np.zeros(len(edata_group['expt_subject'])*800),np.zeros(len(edata_group['expt_subject'])*800),np.zeros(len(edata_group['expt_subject'])*800),np.zeros(len(edata_group['expt_subject'])*800),np.zeros(len(edata_group['expt_subject'])*800),np.zeros(len(edata_group['expt_subject'])*800),np.zeros(len(edata_group['expt_subject'])*800),np.zeros(len(edata_group['expt_subject'])*800),[];

	for subj in edata_group['expt_subject'].index: 		# ls -> true bs -> false bt-> true lt -> true, but should be false?
		#print(parameters_group['probDict'][subj])
		probDict = parameters_group['probDict'][subj] 
		for trial in range(800): 
			subj_array.append(subj)
			cond_array.append(parameters_group['cond'][subj])
			bl_array.append(parameters_group['imageCond'][subj])
			trial_nr_array.append(trial)
			stims_tmp=[]; 
			# add parameters..
			if istest[trial]: # i.e. test trial
				temp_trial_idx = np.count_nonzero(istest[0:trial])
				param_rel_array.append(parameters_group['probTest'][subj][str(temp_trial_idx)])
				stims_tmp = parameters_group['stimTest'][subj][str(temp_trial_idx)]
			else: # i.e. train trial
				temp_trial_idx = np.count_nonzero(istrain[0:trial])
				if len(parameters_group['probTrainCurr'][subj]):
					param_rel_array.append(parameters_group['probTrainCurr'][subj][str(temp_trial_idx)])
					stims_tmp = parameters_group['stimTrainCurr'][subj][str(temp_trial_idx)]
				else:
					param_rel_array.append(parameters_group['probTrainParal'][subj][str(temp_trial_idx)])
					stims_tmp = parameters_group['stimTrainParal'][subj][str(temp_trial_idx)]
			resp_trial_nr_array.append(temp_trial_idx)
			correct_array.append(sdata_group['resp_correct_prob'][subj][-(800-trial)])
			resp_category_array.append(sdata_group['resp_category'][subj][-(800-trial)]) # essentially XOR, 0=left/no, 1=right/yes
			RT_array.append(sdata_group['resp_reactiontime'][subj][-(800-trial)]) # essentially XOR, 0=left/no, 1=right/yes
			rewarded_outcome_array.append(sdata_group['vbxi_category'][subj][-(800-trial)])

			for i in range(4):
				try:
					if probDict[str(stims_tmp[str(i)])]==.1:
						point1_array[subj*800+trial]+=1
					if probDict[str(stims_tmp[str(i)])]==.2:
						point2_array[subj*800+trial]+=1
					if probDict[str(stims_tmp[str(i)])]==.3:
						point3_array[subj*800+trial]+=1
					if probDict[str(stims_tmp[str(i)])]==.4:
						point4_array[subj*800+trial]+=1
					if probDict[str(stims_tmp[str(i)])]==.6:
						point6_array[subj*800+trial]+=1
					if probDict[str(stims_tmp[str(i)])]==.7:
						point7_array[subj*800+trial]+=1
					if probDict[str(stims_tmp[str(i)])]==.8:
						point8_array[subj*800+trial]+=1
					if probDict[str(stims_tmp[str(i)])]==.9:
						point9_array[subj*800+trial]+=1
				except: # <4 stims on a trial
					pass 
			#nr_stims_array.append(point1_array[subj*800+trial]+point2_array[subj*800+trial]+point3_array[subj*800+trial]+point4_array[subj*800+trial]+point6_array[subj*800+trial]+point7_array[subj*800+trial]+point8_array[subj*800+trial]+point9_array[subj*800+trial])

		blocks_array.append(blocks)
		istest_array.append(istest)
		istrain_array.append(istrain)

		if subj%25==0:
			print("%d/%d participants done"%(subj,len(edata_group['expt_subject'])))
	
	DF['subject'] = subj_array
	DF['condition'] = cond_array
	DF['relevant dimension'] = bl_array
	DF['trial number'] = trial_nr_array
	DF['resp trial number'] = resp_trial_nr_array
	DF['parameter (relevant)'] = param_rel_array
	DF['block'] = np.ravel(blocks_array)
	DF['correct'] = correct_array
	DF['response category'] = resp_category_array
	DF['rewarded outcome'] = rewarded_outcome_array
	DF['istest'] = np.ravel(istest_array)
	DF['istrain'] = np.ravel(istrain_array)
	DF['.1'] = point1_array
	DF['.2'] = point2_array
	DF['.3'] = point3_array
	DF['.4'] = point4_array
	DF['.6'] = point6_array
	DF['.7'] = point7_array
	DF['.8'] = point8_array
	DF['.9'] = point9_array
	#DF['number of stimuli'] = nr_stims_array


	DF=DF.dropna() # remove nans

	pd.DataFrame.from_dict(DF).to_hdf(os.path.join(hdf5_path,"DF.h5"), "DF")

else:
	DF = pd.read_hdf(os.path.join(hdf5_path,"DF.h5"))

test_resp_binned_all = []; train_resp_binned_all = []
# [test_resp_binned_all.append( np.array( [[test_resp_all[i][np.where((test_params_all[i][dim]>binmin)&(test_params_all[i][dim]<binmin+.1))[0]] for binmin in np.linspace(0,.9,10)] for dim in range(test_params_all[i].shape[0])] )  ) for i in range(len(test_resp_all)) ]
# [test_resp_binned_all.append( np.array( [np.nanmean(DF[DF.subject==i][DF.istest==1][DF['parameter (relevant)']>=binmin][DF['parameter (relevant)']<binmin+.125]['response category']) for binmin in np.linspace(0,.875,8)]) ) for i in range(len(p)) ]
# [train_resp_binned_all.append( np.array( [np.nanmean(DF[DF.subject==i][DF.istrain==1][DF['parameter (relevant)']>=binmin][DF['parameter (relevant)']<binmin+.125]['response category']) for binmin in np.linspace(0,.875,8)] )  ) for i in range(len(p)) ]

print(DF)
for col in DF.columns:
	print(col)

N = 79

#sigmoid function fitting: first individual fits and then average 
test_resp_binned_all = []; train_resp_binned_all = []
# [test_resp_binned_all.append( np.array( [[test_resp_all[i][np.where((test_params_all[i][dim]>binmin)&(test_params_all[i][dim]<binmin+.1))[0]] for binmin in np.linspace(0,.9,10)] for dim in range(test_params_all[i].shape[0])] )  ) for i in range(len(test_resp_all)) ]
[test_resp_binned_all.append( np.array( [np.nanmean(DF[DF.subject==i][DF.istest==1][DF['parameter (relevant)']>=binmin][DF['parameter (relevant)']<binmin+.125]['response category']) for binmin in np.linspace(0,.875,8)]) ) for i in range(N) ]
[train_resp_binned_all.append( np.array( [np.nanmean(DF[DF.subject==i][DF.istrain==1][DF['parameter (relevant)']>=binmin][DF['parameter (relevant)']<binmin+.125]['response category']) for binmin in np.linspace(0,.875,8)] )  ) for i in range(N) ]


x_fit=np.arange(0,1,1./1000)
test_y_fit_sigmoid4k_c0 = []; train_y_fit_sigmoid4k_c0 = [];
test_y_fit_sigmoid4k_c1 = []; train_y_fit_sigmoid4k_c1 = [];

# def func(x, a, b, c, d):
# 	return a/(b+np.exp(-(c+d*x)))

def sigmoid(x, a, b):
	return 1/(1+np.exp(-(a+b*x)))

# sigmoid = lambda x, a, b, c, d: a/(b+np.exp(-(c+d*x))) 

xdata=np.array(DF[DF.istrain==1][DF.condition==0]['parameter (relevant)'])
ydata=np.array(DF[DF.istrain==1][DF.condition==0]['response category'])

[a,b]=sp.optimize.curve_fit(sigmoid, xdata, ydata)
train_y_fit_sigmoid4k_c0 = sigmoid(x_fit, a[0], a[1]) 
[a,b]=sp.optimize.curve_fit(sigmoid, np.array(DF[DF.istest==1][DF.condition==0]['parameter (relevant)']), np.array(DF[DF.istest==1][DF.condition==0]['response category']))
test_y_fit_sigmoid4k_c0 = sigmoid(x_fit,a[0], a[1])
[a,b]=sp.optimize.curve_fit(sigmoid, np.array(DF[DF.istrain==1][DF.condition==1]['parameter (relevant)']), np.array(DF[DF.istrain==1][DF.condition==1]['response category']))
train_y_fit_sigmoid4k_c1 = sigmoid(x_fit,a[0], a[1])
[a,b]=sp.optimize.curve_fit(sigmoid, np.array(DF[DF.istest==1][DF.condition==1]['parameter (relevant)']), np.array(DF[DF.istest==1][DF.condition==1]['response category']))
test_y_fit_sigmoid4k_c1 = sigmoid(x_fit,a[0], a[1])



#individual sigmoid fits 
train_y_fit_rel_4k_all_s = []; train_y_a_rel_4k_all_s = []; train_y_b_rel_4k_all_s = []; train_y_c_rel_4k_all_s = []; train_y_d_rel_4k_all_s = [];
for subj in range(N): 
	DF_slice = DF[DF['istrain'] == 1][DF['subject']==subj]
	print(subj)
	[a,b] = sp.optimize.curve_fit(sigmoid, np.array(DF_slice['parameter (relevant)']), np.array(DF_slice['response category']), maxfev = 100000)
	train_y_fit_rel_4k_all_s.append(sigmoid(x_fit,a[0], a[1]))
	train_y_a_rel_4k_all_s.append(a[0])
	train_y_b_rel_4k_all_s.append(a[1])
	# train_y_c_rel_4k_all_s.append(c)
	# train_y_d_rel_4k_all_s.append(d)		 


test_y_fit_rel_4k_all_s = []; test_y_a_rel_4k_all_s = []; test_y_b_rel_4k_all_s = []; test_y_c_rel_4k_all_s = []; test_y_d_rel_4k_all_s = [];
for subj in range(N): 
	DF_slice = DF[DF['istest'] == 1][DF['subject']==subj]
	[a,b] = sp.optimize.curve_fit(sigmoid, np.array(DF_slice['parameter (relevant)']), np.array(DF_slice['response category']), maxfev = 100000)
	test_y_fit_rel_4k_all_s.append(sigmoid(x_fit, a[0], a[1]))
	test_y_a_rel_4k_all_s.append(a[0])
	test_y_b_rel_4k_all_s.append(a[1])
	# test_y_c_rel_4k_all_s.append(c)
	# test_y_d_rel_4k_all_s.append(d)		

### plot 






#regression analysis 
from sklearn.linear_model import LogisticRegression 

cue_reg_intercept_train, cue_reg_intercept_test = [], []
cue_reg_weights_train, cue_reg_weights_test = [], [] 
for subj in range(N): 
	DF_slice = DF[DF.subject==subj][DF.istrain==1]
	X = np.array(DF_slice[['.1', '.2', '.3', '.4', '.6', '.7', '.8', '.9']])
	y = np.array(DF_slice['response category'])
	#print(subj)
	clf = LogisticRegression(penalty = 'l2', solver = 'lbfgs').fit(X,y)
	#print(clf.intercept_, clf.coef_)
	cue_reg_intercept_train.append(clf.intercept_)
	cue_reg_weights_train.append(clf.coef_)

	DF_slice = DF[DF.subject==subj][DF.istest==1]
	X = np.array(DF_slice[['.1', '.2', '.3', '.4', '.6', '.7', '.8', '.9']])
	y = np.array(DF_slice['response category'])
	#print(subj)
	clf = LogisticRegression(penalty = 'l2', solver = 'lbfgs').fit(X,y)
	#print(clf.intercept_, clf.coef_)
	cue_reg_intercept_test.append(clf.intercept_)
	cue_reg_weights_test.append(clf.coef_)

c = []
for i in range(N):
	tmp = np.unique(DF[DF.subject == i]['condition'].values)
	if  tmp[0] == 0:
		for k in range(18):
			c.append('Curriculum')
	else:
		for k in range(18):
			c.append('Parallel')


DF_reg = pd.DataFrame() 
DF_tmp = pd.DataFrame()
DF_tmp['cue'] = ['intercept','.1','.2','.3','.4','.6','.7','.8','.9']
count = -1 
for subj in range(N): 
	DF_tmp['weights']=np.hstack((cue_reg_intercept_train[subj], np.squeeze(cue_reg_weights_train)[subj]))
	DF_tmp['subject']=subj
	# DF_tmp['condition']=parameters_group['cond'][subj]
	DF_tmp['istest']=0
	DF_reg=DF_reg.append(DF_tmp)
	DF_tmp['weights']=np.hstack((cue_reg_intercept_test[subj], np.squeeze(cue_reg_weights_test)[subj]))
	DF_tmp['istest']=1
	DF_reg=DF_reg.append(DF_tmp)
DF_reg['condition'] = c



#############################
# PLOTS
#############################
import pylab as plt

SMALLEST_SIZE = 16
SMALL_SIZE = 18
MEDIUM_SIZE = 18
BIGGER_SIZE = 22
FONT_SIZE = 20

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALLEST_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALLEST_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALLEST_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

col1 = '#aa8866'
col2 = '#79ad5f'
col3 = '#727272'

condcols = {0: "#AA3939",1: "#226666", 2:"#333333"} 
er_style = 'ci_band'
condnames = {0: "Curriculum",1: "Parallel"} 


# Train + Test plots
sns.set_style('white')
fig1 = plt.figure(figsize=(6,4))
ax = fig1.add_subplot(1,1,1)
fig1.suptitle("Accuracy per Block")
ax.set_xlabel('Block')
ax.set_ylabel('Accuracy')
sns.lineplot(data = DF, x = 'block', y = 'correct', hue = 'condition', legend = False, palette=['#226666','#AA3939'], ci = 68)
plt.legend(title = 'Condition', loc = 'upper right', labels = ['Curriculum', 'Parallel'])
plt.axvline(x=8, alpha =.5, c="black", dashes=[10,10])
plt.ylim(.5, 1)
plt.tight_layout()
fig1.savefig('train_test_curves'); 
plt.close()


# Performance barplot
nems = ['Train', 'Test'];
for i in range(2):
	b1 = plt.figure(figsize = (16,9))
	tmp = DF[DF['istest']==i].groupby('subject').mean()
	[t,p] = sp.stats.ttest_ind(tmp[tmp.condition==0]['correct'], tmp[tmp.condition==1]['correct'])
	tmp['condition'] = ['Parallel' if i else 'Curriculum' for i in tmp['condition']]
	sns.barplot(data=tmp, x='condition', y='correct', palette=['#226666','#AA3939'], ci=68)
	plt.suptitle('Overall accuracy - '+nems[i]+' - t=%.2f, p=%.2f'%(t,p))
	plt.ylim([0,.8])
	b1.savefig(plotting_dir+nems[i]+"_accs_bar.png")
	plt.close()


##test bar plot
fig = plt.figure()
sns.set_style('white')
tmp = pd.DataFrame()
for i in range(2):
	tmp = tmp.append(DF[DF['istest']==i].groupby('subject').mean())
tmp['Condition'] = ['Parallel' if i else 'Curriculum' for i in tmp['condition']]
tmp['istest'] = ['Test' if i else 'Train' for i in tmp['istest']]
sns.barplot(data=tmp, capsize = .05, x='istest', y='correct', hue = 'Condition', palette=['#226666','#AA3939'], ci = 68)
plt.suptitle('Train & Test Overall Accuracy')
plt.xlabel('')
plt.ylabel('Accuracy')
plt.legend(loc = 'upper right')
sns.despine()

# plt.show()
plt.savefig('Train & Test Overall Accuracy')


c0idx = np.unique(DF[DF.condition == 0]['subject'].values)
c1idx = np.unique(DF[DF.condition == 1]['subject'].values)


#sigmoid function fitting plot per-condition visualization 
f = plt.figure(figsize = (12,10))
sns.set(font_scale = 1.5)
sns.set_style('white')
f.suptitle("Sigmoids: P(right) (y) ~ integrated probability (x)") 
s1 = f.add_subplot(2,2,1)
sns.lineplot( x_fit, np.array(train_y_fit_sigmoid4k_c0),color='#226666', alpha = .7)
sns.regplot(np.linspace(.0625,.9375,8), np.nanmean(np.array(train_resp_binned_all)[c0idx],axis=0), fit_reg=False, scatter_kws = {'lw':5, 'alpha':0.4}, color='black')
s1.axis([0,1,0,1])
s2 = f.add_subplot(2,2,2)
sns.lineplot(x_fit,np.array(test_y_fit_sigmoid4k_c0),  color='#226666', alpha = .7)
sns.regplot(np.linspace(.0625,.9375,8), np.nanmean(np.array(test_resp_binned_all)[c0idx],axis=0), fit_reg=False, scatter_kws = {'lw':5, 'alpha':0.4}, color="black")
s2.axis([0,1,0,1])
s3 = f.add_subplot(2,2,3)
sns.lineplot(x_fit,np.array(train_y_fit_sigmoid4k_c1), color='#AA3939', alpha = .4)
sns.regplot(np.linspace(.0625,.9375,8), np.nanmean(np.array(train_resp_binned_all)[c1idx],axis=0), fit_reg=False, scatter_kws = {'lw':5, 'alpha':0.4}, color="black")
s3.axis([0,1,0,1])
s4 = f.add_subplot(2,2,4)
sns.lineplot(x_fit, np.array(test_y_fit_sigmoid4k_c1), color='#AA3939', alpha = .4)
sns.regplot(np.linspace(.0625,.9375,8), np.nanmean(np.array(test_resp_binned_all)[c1idx],axis=0), fit_reg=False, scatter_kws = {'lw':5, 'alpha':0.4}, color='black')
s4.axis([0,1,0,1])
s1.set_title("Training")
s1.set_ylabel("Curriculum")
s2.set_title("Test")
s3.set_ylabel("Parallel")
plt.tight_layout()
f.savefig(plotting_dir+"sigmoids_percond.png"); plt.close()

# per-subject average
f = plt.figure(figsize = (12,10))
f.suptitle("Sigmoids: P(good weather) (y) ~ integrated probability (x)") 
s1 = f.add_subplot(2,2,1)
sns.lineplot(x_fit, np.mean(np.array(train_y_fit_rel_4k_all_s)[c0idx], axis = 0),  color='#226666', alpha = .7)
sns.regplot(np.linspace(.0625,.9375,8), np.nanmean(np.array(train_resp_binned_all)[c0idx],axis=0), fit_reg=False, scatter_kws = {'lw':5, 'alpha':0.4}, color='black')
s1.axis([0,1,0,1])
s2 = f.add_subplot(2,2,2)
sns.lineplot(x_fit, np.mean(np.array(test_y_fit_rel_4k_all_s)[c0idx], axis = 0), color='#226666', alpha = .7)
sns.regplot(np.linspace(.0625,.9375,8), np.nanmean(np.array(test_resp_binned_all)[c0idx],axis=0), fit_reg=False, scatter_kws = {'lw':5, 'alpha':0.4}, color="black")
s2.axis([0,1,0,1])
s3 = f.add_subplot(2,2,3)
sns.lineplot(x_fit, np.mean(np.array(train_y_fit_rel_4k_all_s)[c1idx], axis = 0), color='#AA3939', alpha = .4)
sns.regplot(np.linspace(.0625,.9375,8), np.nanmean(np.array(train_resp_binned_all)[c1idx],axis=0), fit_reg=False, scatter_kws = {'lw':5, 'alpha':0.4}, color="black")
s3.axis([0,1,0,1])
s4 = f.add_subplot(2,2,4)
sns.lineplot(x_fit, np.mean(np.array(test_y_fit_rel_4k_all_s)[c1idx], axis = 0), color='#AA3939', alpha = .4)
sns.regplot(np.linspace(.0625,.9375,8), np.nanmean(np.array(test_resp_binned_all)[c1idx],axis=0), fit_reg=False, scatter_kws = {'lw':5, 'alpha':0.4}, color='black')
s4.axis([0,1,0,1])
s1.set_title("Training")
s1.set_ylabel("Curriculum")
s2.set_title("Test")
s3.set_ylabel("Parallel")
f.savefig(plotting_dir+"sigmoids.png")
plt.close()

# Regression weights 
nems = ['Train', 'Test'];
for i in range(2):
	br = plt.figure(figsize = (16,9))
	sns.barplot(data=DF_reg[DF_reg.istest==i], x='cue', y='weights', hue='condition',palette=['#226666','#AA3939'], ci=95)
	plt.suptitle('Cue regression weights - '+nems[i])
	plt.xlabel('cue, ci = 95')
	br.savefig(plotting_dir+nems[i]+"_cue_weights_bar.png")
	plt.close()


#two way anova of regressed weights 
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.graphics.factorplots import interaction_plot

logOdds = lambda x: np.log(x/(1-x)) # lambda indicates a simple function. this line is equivalent to: "def logOdds(x): <br><space> return np.log(x/(1-x))"
p = [.5, .1, .2, .3, .4, .6, .7, .8, .9]
w = logOdds(np.array(p)); w = np.abs(w); w = np.around(w, decimals = 3)
abs_woe = []
for i in range(int(len(DF_reg)/9)):
	for k in range(len(w)):
		abs_woe.append(w[k])
	
DF_reg['abs_woe'] = abs_woe
DF_reg['abs_weights'] = np.abs(DF_reg['weights'].values)

DF_reg_istest = DF_reg[DF_reg.istest == 1]

DF_tmp = DF_reg_istest[DF_reg_istest.cue != 'intercept']

model = ols('abs_weights ~ C(abs_woe) + C(condition) + C(abs_woe):C(condition)', data=DF_tmp).fit()
sm.stats.anova_lm(model, typ=2) 


##post hoc
from statsmodels.stats.multicomp import pairwise_tukeyhsd

pairwise_tukeyhsd(endog=DF_tmp[DF_tmp.condition == 'Curriculum']['abs_weights'], groups=DF_tmp[DF_tmp.condition == 'Curriculum']['abs_woe'], alpha=0.05)


#mixed anova
import pingouin as pg

aov = pg.mixed_anova(dv='abs_weights', within='abs_woe', between='condition', subject='subject', data=DF_tmp)
# Pretty printing of ANOVA summary
pg.print_table(aov)

posthocs = pg.pairwise_ttests(dv='abs_weights', within='abs_woe', between='condition', subject='subject', data=DF_tmp, padjust = 'holm', effsize = 'eta-square')
pg.print_table(posthocs)






fig = plt.figure(figsize = (8, 6))
sns.barplot(data = DF_reg_istest, capsize = .05, x = 'abs_woe', y = 'abs_weights', hue = 'condition', ci = 68, palette=['#226666','#AA3939'])
plt.xlabel('Absolute Ground Truth WOE')
plt.ylabel('Absolute Regressed Weight')
plt.suptitle('Regressed Weights by Absolute WOE')
plt.legend(title = 'Condition')
plt.xticks(ticks = np.arange(5), labels = ['bias', '.41', '.85', '1.39', '2.20'])
plt.ylim(0, 1.8)
sns.despine()
# plt.show()
plt.savefig('regression cue weights by absolute values')


##plot of regression of real responses for participants best fitted bt cue sign 
hdf5_path = '/Users/lucyzhang/Desktop/analysis/model_fitting'

DF_AIC_best = pd.read_hdf(os.path.join(hdf5_path,'AIC_best.h5'), key = 'best fit')

fit = []
for i in range(N):
	for k in range(9):
		fit.append(DF_AIC_best[DF_AIC_best.Subjects == i]['Best Fit'].values[0])
DF_reg_istest['Best_Fit'] = fit 

fig = plt.figure(figsize = (8, 6))
# sns.set(font_scale = .8)
sns.set_style('white')
sns.barplot(data = DF_reg_istest[DF_reg_istest.Best_Fit == 'Cue Sign'], hue_order = ['Curriculum', 'Parallel'], capsize = .05, x = 'abs_woe', y = 'abs_weights', hue = 'condition', ci = 68, palette=['#226666','#AA3939'])
plt.xlabel('Absolute Ground Truth WOE')
plt.ylabel('Absolute Regressed Weight')
plt.suptitle('Regressed Weights by Absolute WOE of Participants Best Fitted by Cue Sign (N = 30, Real data)')
plt.legend(title = 'Condition')
plt.xticks(ticks = np.arange(5), labels = ['bias', '.41', '.85', '1.39', '2.20'])
# plt.ylim(0, 1.8)
sns.despine()
# plt.tight_layout()
# plt.show()
plt.savefig('regression cue weights by absolute values participants best fitted by cue real data')


# per-subject 
# f = pl.figure(figsize = (12,10))
# f.suptitle("Sigmoids: P(good weather) (y) ~ integrated probability (x)") 
f = plt.figure(figsize = (12,9))
for i in range(min(np.sum(parameters_group.cond==0),32)):
	f.add_subplot(8,4,i+1)
	sns.lineplot(x_fit, np.array(test_y_fit_rel_4k_all_s)[c0idx[i]], color='#226666', alpha = .9)
	sns.regplot(np.linspace(.0625,.9375,8), np.array(test_resp_binned_all)[c0idx[i]], fit_reg=False, scatter_kws = {'lw':5, 'alpha':0.4}, color='black')
	plt.axis([0,1,0,1])
	plt.xticks([], [])
	plt.yticks([], [])
f.savefig(plotting_dir+"persubj_Curr_sigmoids_test.png")

plt.close()
f = plt.figure(figsize = (12,9))
for i in range(min(np.sum(parameters_group.cond==1),32)):
	f.add_subplot(8,4,i+1)
	sns.lineplot(x_fit, np.array(test_y_fit_rel_4k_all_s)[c1idx[i]], color='#AA3939', alpha = .9)
	sns.regplot(np.linspace(.0625,.9375,8), np.array(test_resp_binned_all)[c1idx[i]], fit_reg=False, scatter_kws = {'lw':5, 'alpha':0.4}, color='black')
	plt.axis([0,1,0,1])
	plt.xticks([], [])
	plt.yticks([], [])
f.savefig(plotting_dir+"persubj_Parallel_sigmoids_test.png")
plt.close()

