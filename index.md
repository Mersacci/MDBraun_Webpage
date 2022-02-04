# Hello world!

Welcome to my webpage!

![Alt text](https://c4.wallpaperflare.com/wallpaper/320/157/310/fractal-mandelbrot-set-wallpaper-preview.jpg)

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

![Alt text](https://wallpapercave.com/wp/BLwLejR.jpg)

The Mandelbrot set is a set of complex numbers which does not diverge to infinity when iterated from z=0.  It is represented by the function:

<img src="https://render.githubusercontent.com/render/math?math=z_%7Bn%2B1%7D%3Dz_n%5E2%2Bc" width="600">

### Python Code for LOESS Model

Here we will perform a locally weighted linear regression on the data from the mtcars.csv dataset

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


![loess_graph](https://user-images.githubusercontent.com/85206050/152606204-bf0f03d6-8793-48df-bf6f-8c6fb3da7717.png)
