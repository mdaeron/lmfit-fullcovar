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
params.add('d', value = 0, expr = 'c*4+a-1')

out = lmfit.minimize(residuals, params)	
	
print(lmfit.fit_report(out))
	
for _ in fullcovar(out, named = False):
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
    chi-square         = 0.33993761
    reduced chi-square = 0.00708203
    Akaike info crit   = -245.550809
    Bayesian info crit = -241.726763
[[Variables]]
    a:  0.99999450 +/- 0.00907082 (0.91%) (init = 1)
    b:  1.00023346 +/- 0.00255585 (0.26%) (init = 1)
    c:  1.1673e-04 +/- 0.00127793 (1094.79%) == '(b-1)/2'
    d:  4.6141e-04 +/- 0.00907082 (1965.88%) == 'c*4+a-1'
[[Correlations]] (unreported correlations are < 0.100)
    C(a, b) = -0.231

[[ 8.22798322e-05 -5.36437388e-06 -2.68218694e-06  7.15510844e-05]
 [-5.36437388e-06  6.53237710e-06  3.26618855e-06  7.70038031e-06]
 [-2.68218694e-06  3.26618855e-06  1.63309427e-06  3.85019016e-06]
 [ 7.15510844e-05  7.70038031e-06  3.85019016e-06  8.69518451e-05]]

[0.00907082 0.00255585 0.00127793 0.0093248 ]

[[ 1.         -0.23138581 -0.23138581  0.84592161]
 [-0.23138581  1.          1.          0.32310014]
 [-0.23138581  1.          1.          0.32310014]
 [ 0.84592161  0.32310014  0.32310014  1.        ]]
```