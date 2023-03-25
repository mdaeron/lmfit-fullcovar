# Compute full covariance for `lmfit` parameters

[LMfit](https://github.com/lmfit/lmfit-py) provides a Least-Squares Minimization routine and class with a simple, flexible approach to parameterizing a model for fitting to data. Among other features, it allows adding arbitrary constraints to the fit parameters. For example, to specify that parameters `a` and `b` should be equal, one might write:

```py
from lmfit import Parameters

params = Parameters()
params['a'] = Parameter(value = 1.0)
params['b'] = Parameter(value = 1.0, expr = 'a')
```

After minimization, LMfit will provide an estimate of the variance-covariance matrix for best-fit values of the parameters, but this matrix only includes parameters without an `expr` specification.

The function `fullcovar()` defined here computes the full variance-covariance matrix for all parameters, along with the corresponding standard errors and correlation matrix:

```py
from pylab import *
from fullcovar import fullcovar
import lmfit

x = linspace(0,10)
y = sin(x)

# add some noise
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
```

The expected output is:

```
[[Fit Statistics]]
	# fitting method   = leastsq
	# function evals   = 7
	# data points      = 50
	# variables        = 2
	chi-square         = 0.33993009
	reduced chi-square = 0.00708188
	Akaike info crit   = -245.551916
	Bayesian info crit = -241.727870
[[Variables]]
	a:  1.00056797 +/- 0.01742448 (1.74%) (init = 1)
	b:  1.00022527 +/- 0.00249682 (0.25%) (init = 1)
	c:  1.1264e-04 +/- 0.00124841 (1108.35%) == '(b-1)/2'
	d:  4.5055e-04 +/- 0.00000000 (0.00%) == 'c*4'

[[ 3.03612501e-04 -4.19684608e-06 -2.09842304e-06 -8.39369216e-06]
 [-4.19684608e-06  6.23411148e-06  3.11705574e-06  1.24682230e-05]
 [-2.09842304e-06  3.11705574e-06  1.55852787e-06  6.23411148e-06]
 [-8.39369216e-06  1.24682230e-05  6.23411148e-06  2.49364459e-05]]

[0.01742448 0.00249682 0.00124841 0.00499364]

[[ 1.         -0.09646637 -0.09646637 -0.09646637]
 [-0.09646637  1.          1.          1.        ]
 [-0.09646637  1.          1.          1.        ]
 [-0.09646637  1.          1.          1.        ]]
```