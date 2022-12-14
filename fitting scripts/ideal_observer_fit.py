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

hdf5_path = '/Users/lucyzhang/Desktop/analysis/ready'
DF = pd.read_hdf(os.path.join(hdf5_path,"DF.h5"))

softmax = lambda y1, y0, b: np.exp(b*y1)/(np.exp(b*y0)+np.exp(b*y1))
logOdds = lambda x: np.log(x/(1-x)) # lambda indicates a simple function. this line is equivalent to: "def logOdds(x): <br><space> return np.log(x/(1-x))"
sigmoid = lambda x: 1/(1+np.exp(-x))

#slicing dataphrame to take the stimuli schedule 
def ideal_fit_data(subject):
	xs = DF[DF.subject==subject][['.1', '.2', '.3', '.4', '.6', '.7', '.8', '.9']].to_numpy()
	xs = np.insert(xs, 0, 1, axis = 1)
	response_subject = DF[DF.subject==subject]['response category'].to_numpy()
	rewarded_outcome = DF[DF.subject==subject]['rewarded outcome'].to_numpy()
	pA1 = DF[DF.subject==subject]['parameter (relevant)'].to_numpy()
	correct_array = (pA1>.5).astype('int')


	#calculate outcome and reward schedule 
	#A1 refers to the right option, A0 the left option, coding (1 right, 0 left)
	p = [.1, .2, .3, .4, .6, .7, .8, .9]

	# def set_up (xs, p):
	# 	woes = logOdds(np.array(p)); woes = np.insert(woes, 0, 0)
	# 	stim_to_woe = woes*xs
	# 	woe_A1 = np.nansum(stim_to_woe, axis=1)
	# 	p_A1 = sigmoid(woe_A1)
	# 	outcome = (np.random.uniform(0,1,len(xs)) < p_A1).astype('int')
	# 	pA1_outcome_sche = np.hstack((p_A1[:,None], outcome[:, None]))

	# 	return (pA1_outcome_sche)



	def ideal_fit(para, symm_updates=1, verbose=0):
		alpha = para[0]
		beta = para[1]
		qs = np.zeros((2,9)) #qs[0] for A1(choosing right), qs[1] for A0(choosing left)
		p_choice_A1_array = []; 
		A1_chosen_array = []; 
		correct = []
		Choice_prob_chosen = []
		
		#pA1_outcome = set_up(xs, p)

		################################
		#curriculum
		################################
		#training
		for trial in range(len(xs)):
			x = xs[trial, :]
			y_sum = np.matmul(qs, x) #output decision variable
			#choice prob of A1 stochastic decision rule
			p_choice_right = softmax(y_sum[0], y_sum[1], beta)
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
			if symm_updates:
				qs[1,:] += alpha*((1-r) - y_sum[1])*x
				qs[0,:] += alpha*(r - y_sum[0])*x 
			else:
				y_tmp = p_choice_right if choice else 1-p_choice_right
				qs[choice,:] += alpha * (correct[trial] - y_tmp) * x


		for i in range(len(response_subject)):
			if response_subject[i]:
				Choice_prob_chosen.append(p_choice_A1_array[i])
			else:
				Choice_prob_chosen.append(1 - p_choice_A1_array[i])

		#print(np.log(np.array(Choice_prob_chosen)))
		error = -np.sum(np.log(np.array(Choice_prob_chosen)))

		if verbose:
			abs_error = np.mean((1-np.array(Choice_prob_chosen)))
			print("mean fit error is %.2f"%abs_error)

		return error


	##finding best fit 
	x0 = [.05, 1]
	bnds = ((0,.2), (0, 10))
	res = sp.optimize.dual_annealing(ideal_fit, bounds = bnds) 
	print(res)
	# para = res.x 
	# print(para)
	# alpha = para[0]; beta = para[1]
	error = res.fun
	para = res.x

	return (para[0], para[1], error) 


# LR = np.arange(0, 0.3, .01)
# beta = np.arange(.4, 8, .1)
# cue = 2
# error = []
# for i in LR:
# 	row =[]
# 	for k in beta:
# 		print(i,k)
# 		row.append(ideal_fit([i, k]))
# 	print(row)	
# 	error.append(row)

# ax = sns.heatmap(np.array(error), vmin=300, vmax=800)
# plt.show()


x = np.linspace(0, 1, 1000)
y = logOdds(x)

fig, ax = plt.subplots()
sns.lineplot(x, y, ax = ax)
ax.scatter(p, w, marker = 'X', color = 'red', label = 'Cues')
ax.legend()
plt.xlabel('P(right)')
plt.ylabel('WOE')
# plt.show()
plt.savefig('illustration of p to woe')




