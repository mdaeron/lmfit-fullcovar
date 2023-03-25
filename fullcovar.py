#! /usr/bin/env python3

import asteval
import numpy as np

def fullcovar(minresult, epsilon = 0.01):
	'''
	Construct full covariance matrix in the case of constrained parameters
	'''
	def f(var_names):
		interp = asteval.Interpreter()
		for n,v in zip(minresult.var_names, var_names):
			interp(f'{n} = {v}')
		for q in minresult.params:
			if minresult.params[q].expr:
				interp(f'{q} = {minresult.params[q].expr}')
		return np.array([interp.symtable[q] for q in minresult.params])

	# construct Jacobian
	J = np.zeros((minresult.nvarys, len(minresult.params)))
	X = array([minresult.params[p].value for p in minresult.var_names])
	sX = array([minresult.params[p].stderr for p in minresult.var_names])

	for j in range(minresult.nvarys):
		x1 = [_ for _ in X]
		x1[j] += epsilon * sX[j]
		x2 = [_ for _ in X]
		x2[j] -= epsilon * sX[j]
		J[j,:] = (f(x1) - f(x2)) / (2 * epsilon * sX[j])

	_covar = J.T @ minresult.covar @ J
	_se = np.diag(_covar)**.5
	_correl = _covar / np.expand_dims(_se, 0) / np.expand_dims(_se, 1)

	return _covar, _se, _correl


if __name__ == '__main__':

	from pylab import *
	import lmfit

	x = linspace(0,10)
	y = sin(x)
	y[::3] += .1
	y[1::3] -= .1

	def residuals(p, x=x, y=y):
		return y - p['a']*sin(p['b']*x + p['c']) - p['d']

	params = lmfit.Parameters()
	params.add('a', value = 1)
	params.add('b', value = 1)
	params.add('c', value = 0, expr = '(b-1)/2')
	params.add('d', value = 0, expr = 'c*4')

	out = lmfit.minimize(residuals, params)	
	
	print(lmfit.fit_report(out))
	
	for _ in fullcovar(out):
		print()
		print(_)

	# EXPECTED OUTPUT:
	# 
	# [[Fit Statistics]]
	# 	# fitting method   = leastsq
	# 	# function evals   = 7
	# 	# data points      = 50
	# 	# variables        = 2
	# 	chi-square         = 0.33993009
	# 	reduced chi-square = 0.00708188
	# 	Akaike info crit   = -245.551916
	# 	Bayesian info crit = -241.727870
	# [[Variables]]
	# 	a:  1.00056797 +/- 0.01742448 (1.74%) (init = 1)
	# 	b:  1.00022527 +/- 0.00249682 (0.25%) (init = 1)
	# 	c:  1.1264e-04 +/- 0.00124841 (1108.35%) == '(b-1)/2'
	# 	d:  4.5055e-04 +/- 0.00000000 (0.00%) == 'c*4'
	# 
	# [[ 3.03612501e-04 -4.19684608e-06 -2.09842304e-06 -8.39369216e-06]
	#  [-4.19684608e-06  6.23411148e-06  3.11705574e-06  1.24682230e-05]
	#  [-2.09842304e-06  3.11705574e-06  1.55852787e-06  6.23411148e-06]
	#  [-8.39369216e-06  1.24682230e-05  6.23411148e-06  2.49364459e-05]]
	# 
	# [0.01742448 0.00249682 0.00124841 0.00499364]
	# 
	# [[ 1.         -0.09646637 -0.09646637 -0.09646637]
	#  [-0.09646637  1.          1.          1.        ]
	#  [-0.09646637  1.          1.          1.        ]
	#  [-0.09646637  1.          1.          1.        ]]
