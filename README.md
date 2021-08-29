# medical cost predictions

optimal way to predict medical cost of patients based on their historical data such as age,gender children, smoking
habits and their region of live. of course this is regression problem so linear, non-linear and ensemble methods are
used to choose best model to make prediction with less variance and with higher accuracy possible.

## Data quality

before apply any modeling algorithm or before do any changes to data examine basic assumption with 4-plots, which are:

- **Fixed Location:**
  If the fixed location assumption holds, then the run sequence plot will be flat and non-drifting.

- **Fixed Variation:**
  If the fixed variation assumption holds, then the vertical spread in the run sequence plot will be approximately the
  same over the entire horizontal axis.

- **Randomness:**
  If the randomness assumption holds, then the lag plot will be structureless and random.

- **Fixed Distribution:**
  If the fixed distribution assumption holds, in particular if the fixed normal distribution holds, then the histogram
  will be bell-shaped, and the normal probability plot will be linear.

<img src="https://github.com/ashen007/Medical_Cost_prediction/blob/master/demo/EDA/charges_4plot.png" alt="4-plot example">

***central tendency***

after prove data fulfill underlying assumption then exploratory data analysis can be begun. first summery about location
statistics and variability of data then distribution and correlation statistics are calculated. four estimators used to
estimate mean(not robust), trimmed mean, winzorized mean and median. winsorized mean has the lowest standard error among
three mean estimators. median is lower than mean in every feature that is a hint on right skewed distribution.

<img src="https://github.com/ashen007/Medical_Cost_prediction/blob/master/demo/EDA/central tendancy.jpg" alt="central tendency estimators" width="400px">
<img src="https://github.com/ashen007/Medical_Cost_prediction/blob/master/demo/EDA/std error of central tendancy.jpg" alt="standard error of estimator" width="400px">

***distribution analysis***

chi-squared goodness of fit test done on every discrete and continuess feature to estimate best distribution to match
with data.

<img src="https://github.com/ashen007/Medical_Cost_prediction/blob/master/demo/EDA/children_hist.png" alt="distribution (histogram)">
<img src="https://github.com/ashen007/Medical_Cost_prediction/blob/master/demo/EDA/continues_hist.png" alt="distribution (histogram)">
<img src="https://github.com/ashen007/Medical_Cost_prediction/blob/master/demo/EDA/kde.png" alt="distribution (histogram)">

theoretical distributions with underline distribution of data :

<img src="https://github.com/ashen007/Medical_Cost_prediction/blob/master/demo/EDA/age_comp_plot.png" width="400px" alt="distribution (histogram)">
<img src="https://github.com/ashen007/Medical_Cost_prediction/blob/master/demo/EDA/bmi_comp_plot.png" width="400px" alt="distribution (histogram)">
<img src="https://github.com/ashen007/Medical_Cost_prediction/blob/master/demo/EDA/charges_comp_plot.png" width="400px" alt="distribution (histogram)">

estimated distribution based on chi-squared test data :

<img src="https://github.com/ashen007/Medical_Cost_prediction/blob/master/demo/EDA/age_dist_score.png" width="400px" alt="estimated distribution (kde)">
<img src="https://github.com/ashen007/Medical_Cost_prediction/blob/master/demo/EDA/age_estimate_dist.png" width="400px" alt="estimated distribution (kde)">
<img src="https://github.com/ashen007/Medical_Cost_prediction/blob/master/demo/EDA/bmi_dist_score.png" width="400px" alt="estimated distribution (kde)">
<img src="https://github.com/ashen007/Medical_Cost_prediction/blob/master/demo/EDA/bmi_estimate_dist.png" width="400px" alt="estimated distribution (kde)">
<img src="https://github.com/ashen007/Medical_Cost_prediction/blob/master/demo/EDA/charges_dist_score.png" width="400px" alt="estimated distribution (kde)">
<img src="https://github.com/ashen007/Medical_Cost_prediction/blob/master/demo/EDA/charges_estimate_dist.png" width="400px" alt="estimated distribution (kde)">

## Dataset

dataset use in here is medical cost prediction dataset from **kaggle**. you can find it
from <a href="https://www.kaggle.com/mirichoi0218/insurance">here</a>.

## models
