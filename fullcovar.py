#! /usr/bin/env python3

import asteval
import numpy as np

def fullcovar(minresult, epsilon = 0.01, named = True):
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

	if named:
		_covar = {i: {j:_covar[i,j] for j in minresult.params} for i in minresult.params}
		_se = {i: _se[i] for i in minresult.params}
		_correl = {i: {j:_correl[i,j] for j in minresult.params} for i in minresult.params}

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
	params.add('d', value = 0, expr = 'c*4+a-1')

	out = lmfit.minimize(residuals, params)	
	
	print(lmfit.fit_report(out))
	
	for _ in fullcovar(out, named = False):
		print()
		print(_)

	# EXPECTED OUTPUT:
	# 
	# [[Fit Statistics]]
	#     # fitting method   = leastsq
	#     # function evals   = 7
	#     # data points      = 50
	#     # variables        = 2
	#     chi-square         = 0.33993761
	#     reduced chi-square = 0.00708203
	#     Akaike info crit   = -245.550809
	#     Bayesian info crit = -241.726763
	# [[Variables]]
	#     a:  0.99999450 +/- 0.00907082 (0.91%) (init = 1)
	#     b:  1.00023346 +/- 0.00255585 (0.26%) (init = 1)
	#     c:  1.1673e-04 +/- 0.00127793 (1094.79%) == '(b-1)/2'
	#     d:  4.6141e-04 +/- 0.00907082 (1965.88%) == 'c*4+a-1'
	# [[Correlations]] (unreported correlations are < 0.100)
	#     C(a, b) = -0.231
	# 
	# [[ 8.22798322e-05 -5.36437388e-06 -2.68218694e-06  7.15510844e-05]
	#  [-5.36437388e-06  6.53237710e-06  3.26618855e-06  7.70038031e-06]
	#  [-2.68218694e-06  3.26618855e-06  1.63309427e-06  3.85019016e-06]
	#  [ 7.15510844e-05  7.70038031e-06  3.85019016e-06  8.69518451e-05]]
	# 
	# [0.00907082 0.00255585 0.00127793 0.0093248 ]
	# 
	# [[ 1.         -0.23138581 -0.23138581  0.84592161]
	#  [-0.23138581  1.          1.          0.32310014]
	#  [-0.23138581  1.          1.          0.32310014]
	#  [ 0.84592161  0.32310014  0.32310014  1.        ]]
