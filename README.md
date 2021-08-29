# medical cost predictions

optimal way to predict medical cost of patients based on their historical data such as age,gender children, smoking
habits and their region of live. of course this is regression problem so linear, non-linear and ensemble methods are
used to choose best model to make prediction with less variance and with higher accuracy possible.

## Data quality

before apply any modeling algorithm or before do any changes to data examine basic assumption with 4-plots, which are:

- **Fixed Location:**
  If the fixed location assumption holds, then the run sequence plot will be flat and non-drifting.

- **Fixed Variation:**
  If the fixed variation assumption holds, then the vertical spread in the run sequence plot will be approximately the same over the entire horizontal axis.

- **Randomness:**
  If the randomness assumption holds, then the lag plot will be structureless and random.

- **Fixed Distribution:**
  If the fixed distribution assumption holds, in particular if the fixed normal distribution holds, then the histogram
  will be bell-shaped, and the normal probability plot will be linear.

![4-plot example](demo/EDA/charges_4plot.png)

after prove data fulfill underlying assumption then exploratory data analysis can be begun. first summery about
location statistics and variability of data then distribution and correlation statistics are calculated. four estimators
used to estimate mean(not robust), trimmed mean, winzorized mean and median. winsorized mean has the lowest 
standard error among three mean estimators. median is lower than mean in every feature that is a hint on right skewed
distribution.

![central tendency estimators](demo/EDA/central tendancy.jpg)
![standard error of estimator](demo/EDA/std error of central tendancy.jpg)

## Dataset

dataset use in here is medical cost prediction dataset from **kaggle**. you can find it
from <a href="https://www.kaggle.com/mirichoi0218/insurance">here</a>.

## models
