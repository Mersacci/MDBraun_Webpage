# Hello world!

Welcome to my webpage!
<img src="C:\Users\razor\OneDrive\Pictures\Wallpapers\evol_mandelbrot.jpeg" width="400" height="600" alt="hi" class="inline"/>

## Masaccio Braun

#### Education
**The College of William and Mary**
*Bachelor of Science*
Data Science Major, Data Applications Concentration
Philosophy Major

#### Interests
- Machine Learning
- Reformed Theology
- Natural Philosophy

### Mandelbrot Set

<img src="C:\Users\razor\OneDrive\Pictures\Wallpapers\mandelbrot_set.jpeg" width="400" height="600" alt="hi" class="inline"/>

The Mandelbrot set is a set of complex numbers which does not diverge to infinity when iterated from $z=0$.  It is represented by the function:

$\begin{aligned}z_{n+1}=z_{n}^2 + c\end{aligned}$

### Python Code for LOESS Model

Here we will perform a locally weighted linear regression on the data from the mtcars.csv dataset

<<engine='python', engine.path='python3'>>=
@
```Python
# Hi-rez images
%matplotlib inline
%config InlineBackend.figure_format = 'retina'
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 120

# Import libraries and data
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
import scipy.stats as stats
cars = pd.read_csv(r"C:\Users\razor\OneDrive\Documents\DATA 310\mtcars.csv")

# Create locally weighted linear regression model
loess = sm.nonparametric.lowess

# Define predictor and target variables
x = cars["wt"].values
y = cars["mpg"].values
y_sm = loess(y,x,frac=1/3,it=10, return_sorted = False)

# Generate graph of model
plt.figure(figsize=(8,5))
plt.scatter(x,y, facecolors = 'none', edgecolor = 'darkblue', label = 'data')
plt.plot(x,y_sm,color = 'red', label = 'LOESS model')
plt.xlabel('Weight',fontsize=14)
plt.ylabel('Miles Per Gallon',fontsize=14)
plt.legend()
plt.title('LOESS Regression')
plt.show()
```
### Graph of LOESS Model
This is the output when the above code is run.
<img src="C:\Users\razor\OneDrive\Documents\DATA 410\loess_reg.png" width="400" height="600" alt="hi" class="inline"/>
