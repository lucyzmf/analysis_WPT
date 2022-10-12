import numpy as np
from IPython import embed as shell
import os, sys, datetime, pickle
import scipy as sp
import numpy as np
import numpy.matlib
import matplotlib.pylab as plt
from matplotlib.colors import hex2color, rgb2hex
import pandas as pd
import os
import h5py
from scipy import stats, polyval, polyfit
import seaborn as sns
import math

hdf5_path = '/Users/lucyzhang/Desktop/analysis/ready'
DF = pd.read_hdf(os.path.join(hdf5_path,"DF.h5"))

softmax = lambda y1, y0, b: np.exp(b*y1)/(np.exp(b*y0)+np.exp(b*y1))
logOdds = lambda x: np.log(x/(1-x)) # lambda indicates a simple function. this line is equivalent to: "def logOdds(x): <br><space> return np.log(x/(1-x))"
sigmoid = lambda x: 1/(1+np.exp(-x))

#slicing dataphrame to take the stimuli schedule 
def all_sim(subject, para):
	xs = DF[DF.subject==subject][['.1', '.2', '.3', '.4', '.6', '.7', '.8', '.9']].to_numpy()
	xs = np.insert(xs, 0, 1, axis = 1)
	response_subject = DF[DF.subject==subject]['response category'].to_numpy()
	rewarded_outcome = DF[DF.subject==subject]['rewarded outcome'].to_numpy()
	pA1 = DF[DF.subject==subject]['parameter (relevant)'].to_numpy()
	correct_array = (pA1>.5).astype('int')


	#calculate outcome and reward schedule 
	#A1 refers to the right option, A0 the left option, coding (1 right, 0 left)
	p = [.1, .2, .3, .4, .6, .7, .8, .9]

	alpha = para[0]
	beta = para[1]
	k = para[2]

	#pA1_outcome = set_up(xs, p)

	#training
	qs = np.zeros((2,9)) #qs[0] for A1(choosing right), qs[1] for A0(choosing left)
	p_choice_A1_array = []
	A1_chosen_array = []; 
	correct = []
	Choice_prob_chosen = []

	#pA1_outcome = set_up(xs, p)

	#training
	for trial in range(len(xs)):
		x = xs[trial, :]
		#search for k most predictive cues, return an array of the index of used cues
		present_cues = list(np.nonzero(x)[0]); #print(x_temp)
		present_qs = [qs[0][i] for i in present_cues]
		abs_qs = list(np.abs(present_qs))
		used_cues = []

		no_cue = range(math.ceil(k)) if len(present_cues)-1 > k else range(len(present_cues)-1)

		for i in no_cue:
			cue = np.argmax(abs_qs[1:])+1; 
			used_cues.append(present_cues[cue]); 
			present_cues.pop(cue)
			abs_qs.pop(cue); 
		
		#print(index)
		#only use k%1 of the next cue  
		x_new = np.zeros(9); x_new[0] = 1
		done = []; 
		for i in used_cues:
			x_new[i] = x[i]
			done.append(int(i))
			
			if np.sum(x_new)-1 == math.ceil(k): 
				x_new[done[len(done)-1]] = x_new[done[len(done)-1]]-(1-k%1)
				break 
			elif np.sum(x_new)-1 > math.ceil(k): 
				m = np.sum(x_new)-1
				x_new[done[len(done)-1]] = x_new[done[len(done)-1]]-(1-k%1)-(m-math.ceil(k))
				break
		#print(x)
		y_temp = np.matmul(qs, x_new) #output decision variable
		y_sum = np.matmul(qs, x)	#choice prob of A1 stochastic decision rule
		p_choice_right = softmax(y_temp[0], y_temp[1], beta)
		p_choice_A1_array.append(p_choice_right)
		#A1 chosen based on p pchoice 
		choice = int(p_choice_right > np.random.uniform(0, 1))
		A1_chosen_array.append(choice)
		#whether the chosen outcome is correct or not 
		right = 1 if pA1[trial]>.5 else 0
		correct.append(1 if A1_chosen_array[trial] == right else 0) 
		#whether that option is rewarded or not 
		r = rewarded_outcome[trial] 
		#update
		qs[1,:] += alpha*((1-r) - y_sum[1])*x
		qs[0,:] += alpha*(r - y_sum[0])*x 


	return (A1_chosen_array, correct) 



# LR = np.arange(0, 0.3, .01)
# beta = 3
# cue = np.arange(0, 4, .1)
# error = []
# for i in LR:
# 	row =[]
# 	for k in cue:
# 		print(i,k)
# 		row.append(TTB_all_fit([i, beta, k]))
# 	error.append(row)

# ax = sns.heatmap(np.array(error), vmin = 350, vmax = 800)
# plt.show()
