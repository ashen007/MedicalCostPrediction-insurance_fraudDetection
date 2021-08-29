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

<img src="https://github.com/ashen007/Medical_Cost_prediction/blob/master/demo/EDA/central tendancy.jpg" alt="central tendency estimators" width="400px"> <img src="https://github.com/ashen007/Medical_Cost_prediction/blob/master/demo/EDA/std error of central tendancy.jpg" alt="standard error of estimator" width="400px">

***distribution analysis***

chi-squared goodness of fit test done on every discrete and continuess feature to estimate best distribution to match
with data.

<img src="https://github.com/ashen007/Medical_Cost_prediction/blob/master/demo/EDA/continues_hist.png" alt="distribution (histogram)" width="400px"> <img src="https://github.com/ashen007/Medical_Cost_prediction/blob/master/demo/EDA/kde.png" alt="distribution (histogram)" width="400px">

theoretical distributions with underline distribution of data :

<img src="https://github.com/ashen007/Medical_Cost_prediction/blob/master/demo/age_comp_plot.png" width="400px" alt="distribution (histogram)"> <img src="https://github.com/ashen007/Medical_Cost_prediction/blob/master/demo/bmi_comp_plot.png" width="400px" alt="distribution (histogram)">
<img src="https://github.com/ashen007/Medical_Cost_prediction/blob/master/demo/charges_comp_plot.png" width="400px" alt="distribution (histogram)">

estimated distribution based on **_chi-squared_** test data :

<img src="https://github.com/ashen007/Medical_Cost_prediction/blob/master/demo/age_dist_score.png" width="400px" alt="estimated distribution"> <img src="https://github.com/ashen007/Medical_Cost_prediction/blob/master/demo/age_estimate_dist.png" width="400px" alt="estimated distribution">
<img src="https://github.com/ashen007/Medical_Cost_prediction/blob/master/demo/bmi_dist_score.png" width="400px" alt="estimated distribution"> <img src="https://github.com/ashen007/Medical_Cost_prediction/blob/master/demo/bmi_estimate_dist.png" width="400px" alt="estimated distribution">
<img src="https://github.com/ashen007/Medical_Cost_prediction/blob/master/demo/charges_dist_score.png" width="400px" alt="estimated distribution"> <img src="https://github.com/ashen007/Medical_Cost_prediction/blob/master/demo/charges_estimate_dist.png" width="400px" alt="estimated distribution">

***normality test***

three types of testing methods used to test normalirity of distribution. qq-plots and anderson darling, wilk Shapiro.

- **H0** = The null hypothesis assumes no difference between the observed and theoretical distribution
- **Ha** = The alternative hypothesis assumes there is a difference between the observed and theoretical distribution

<img src="https://github.com/ashen007/Medical_Cost_prediction/blob/doc/demo/EDA/age_norm_test.jpg" width="400px" alt="qq plot"> <img src="https://github.com/ashen007/Medical_Cost_prediction/blob/doc/demo/EDA/bmi_norm_test.jpg" width="400px" alt="qq plot">
<img src="https://github.com/ashen007/Medical_Cost_prediction/blob/doc/demo/EDA/children_norm_test.jpg" width="400px" alt="qq plot"> <img src="https://github.com/ashen007/Medical_Cost_prediction/blob/doc/demo/EDA/charges_norm_test.jpg" width="400px" alt="qq plot">

<table>
    <tbody>
        <tr>
            <td style="background-color: darkgray; text-align: center;" colspan="4">
                <h4>anderson darling test results</h4>
            </td>
        </tr>
        <tr>
            <td><h4>feature</h4></td>
            <td><h4>statics</h4></td>
            <td><h4>significant level</h4></td>
            <td><h4>critical value</h4></td>
        </tr>
        <tr>
            <td rowspan="5">age</td>
            <td rowspan="5">18.7887</td>
            <td>0.574</td>
            <td>15</td>
        </tr>
        <tr>
            <td>0.654</td>
            <td>10</td>
        </tr>
        <tr>
            <td>0.785</td>
            <td>5</td>
        </tr>
        <tr>
            <td>0.915</td>
            <td>2.5</td>
        </tr>
        <tr>
            <td>1.089</td>
            <td>1</td>
        </tr>
        <tr>
            <td rowspan="5">bmi</td>
            <td rowspan="5">1.2355</td>
            <td>0.574</td>
            <td>15</td>
        </tr>
        </tr>
        <tr>
            <td>0.654</td>
            <td>10</td>
        </tr>
        <tr>
            <td>0.785</td>
            <td>5</td>
        </tr>
        <tr>
            <td>0.915</td>
            <td>2.5</td>
        </tr>
        <tr>
            <td>1.089</td>
            <td>1</td>
        </tr>
        <tr>
            <td rowspan="5">children</td>
            <td rowspan="5">87.6711</td>
            <td>0.574</td>
            <td>15</td>
        </tr>
        </tr>
        <tr>
            <td>0.654</td>
            <td>10</td>
        </tr>
        <tr>
            <td>0.785</td>
            <td>5</td>
        </tr>
        <tr>
            <td>0.915</td>
            <td>2.5</td>
        </tr>
        <tr>
            <td>1.089</td>
            <td>1</td>
        </tr>
        <tr>
            <td rowspan="5">charges</td>
            <td rowspan="5">85.1285</td>
            <td>0.574</td>
            <td>15</td>
        </tr>
        </tr>
        <tr>
            <td>0.654</td>
            <td>10</td>
        </tr>
        <tr>
            <td>0.785</td>
            <td>5</td>
        </tr>
        <tr>
            <td>0.915</td>
            <td>2.5</td>
        </tr>
        <tr>
            <td>1.089</td>
            <td>1</td>
        </tr>
    </tbody>
</table>

<table>
    <tbody>
        <tr>
            <td style="background-color: darkgray; text-align: center;" colspan="4">
                <h4>wilk Shapiro test results</h4>
            </td>
        </tr>
        <tr>
            <td><h4>feature</h4></td>
            <td><h4>statics</h4></td>
            <td><h4>p value</h4></td>
            <td><h4>null hypothesis</h4></td>
        </tr>
        <tr>
            <td>age</td>
            <td>0.9446</td>
            <td>5.6874e-22</td>
            <td>reject</td>
        </tr>
        <tr>
            <td>bmi</td>
            <td>0.9938</td>
            <td>2.6098e-05</td>
            <td>reject</td>
        </tr>
        <tr>
            <td>children</td>
            <td>0.8231</td>
            <td>5.0663e-36</td>
            <td>reject</td>
        </tr>
        <tr>
            <td>charges</td>
            <td>0.8146</td>
            <td>1.1504e-36</td>
            <td>reject</td>
        </tr>
    </tbody>
</table>

***outliers***

box plots used to identify outliers in the data.

<img src="https://github.com/ashen007/Medical_Cost_prediction/blob/doc/demo/EDA/boxplot.png" alt="outliers">

## feature standardization, transformation and selection

three methods used to standardized features which are -

- quantile transformation
- box-cox transformation
- yoe-Johnson transformation

scaling done by robust scaling method because there are outliers in data.

![](demo/Model%20Creation/age_box_cox.png)
![](demo/Model%20Creation/bmi_yj_trans.png)
![](demo/Model%20Creation/charges_power_(yj)_trans.png)

## models

coefficient of determination or R2, It measures the amount of variance of the prediction which is explained by the
dataset. R2 values close to 1 mean an almost-perfect regression, while values close to 0 (or negative) imply a bad
model.

![](demo/Model%20Creation/all%20models%20evaluation.png)

- linear models - ordinary least square, lasso (L1), ridge (L2), elastic net, ransac, huber

![](demo/Model%20Creation/compare real_regularization_robust.png)

<table>
<tbody>
<tr style="background-color: darkgray; text-align: center;">
<td><h4>model</h4></td>
<td><h4>score in train data</h4></td>
<td><h4>score in test data</h4></td>
</tr>
<tr>
<td>ols</td>
<td>0.761332</td>
<td>0.791844</td>
</tr>
<tr>
<td>lasso</td>
<td>0.362672</td>
<td>0.362733</td>
</tr>
<tr>
<td>ridge</td>
<td>0.761330</td>
<td>0.791807</td>
</tr>
<tr>
<td>elasticnet</td>
<td>0.559900</td>
<td>0.569312</td>
</tr>
<tr>
<td>ransac</td>
<td>0.595668</td>
<td>0.623618</td>
</tr>
<tr>
<td>huber</td>
<td>0.754969</td>
<td>0.788107</td>
</tr>
</tbody>
</table>

- non-linear models - polynomial ridge, k nearest neighbors, decision tree

![](demo/Model%20Creation/the best tree compare the best knn.png)
![](demo/Model%20Creation/compare%20degree%20of%20polynomial%20function.png)

<table>
<tbody>
<tr style="background-color: darkgray; text-align: center;">
<td><h4>model</h4></td>
<td><h4>score in train data</h4></td>
<td><h4>score in test data</h4></td>
</tr>
<tr>
<td>ridge poly</td>
<td>0.829277</td>
<td>0.856282</td>
</tr>
<tr>
<td>knn</td>
<td>0.822168</td>
<td>0.840297</td>
</tr>
<tr>
<td>tree</td>
<td>0.842030</td>
<td>0.851783</td>
</tr>
</tbody>
</table>

- ensemble model - bagging, random forest, adaptive boost, stacking

![](demo/Model%20Creation/ensemble%20models%20performance.png)

<table>
<tbody>
<tr style="background-color: darkgray; text-align: center;">
<td><h4>model</h4></td>
<td><h4>score in train data</h4></td>
<td><h4>score in test data</h4></td>
</tr>
<tr>
<td>bagging</td>
<td>0.843557</td>
<td>0.852594</td>
</tr>
<tr>
<td>adaboost</td>
<td>0.759680</td>
<td>0.791620</td>
</tr>
<tr>
<td>random forest</td>
<td>0.838272</td>
<td>0.854912</td>
</tr>
<tr>
<td>stacking</td>
<td>0.831071</td>
<td>0.851779</td>
</tr>
</tbody>
</table>

## Dataset

dataset use in here is medical cost prediction dataset from **kaggle**. you can find it
from <a href="https://www.kaggle.com/mirichoi0218/insurance">here</a>.