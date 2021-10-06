# medical cost predictions & insurance fraud detection

The optimal way to predict the medical cost of patients is based on their historical data such as age, gender children, smoking
habits and the region they live in. of course, this is a regression problem so linear, non-linear and ensemble methods are
used to choose the best model to make predictions with less variance and with higher accuracy possible. fraud detection decided
to parts based on either patient admit to hospital or OPD. fraud detection is a binary classification task.

## Data quality

before applying any modelling algorithm or before do any data changes examine basic assumptions with 4-plots, which are:
(almost every feature of fraud detection data are categorical 4-plots are using on numerical data)

- **Fixed Location:**
  If the fixed location assumption holds, then the run sequence plot will be flat and non-drifting.

- **Fixed Variation:**
  If the fixed variation assumption holds, then the vertical spread in the run sequence plot will be approximately the
  same over the entire horizontal axis.

- **Randomness:**
  If the randomness assumption holds, then the lag plot will be structureless and random.

- **Fixed Distribution:**
  If the fixed distribution assumption holds, in particular, if the fixed normal distribution holds, then the histogram
  will be bell-shaped, and the normal probability plot will be linear.

<img src="https://github.com/ashen007/Medical_Cost_prediction/blob/master/demo/EDA/charges_4plot.png" alt="4-plot example">

***central tendency***

after proving the data fulfil the underlying assumption then exploratory data analysis can be begun. The first summary is about location
statistics and variability of data then distribution and correlation statistics are calculated. four estimators were used to
estimate mean(not robust), trimmed mean, winterized mean and median. winsorized mean has the lowest standard error among
The three mean estimators. median is lower than mean in every feature that is a hint on right-skewed distribution.

<img src="https://github.com/ashen007/Medical_Cost_prediction/blob/master/demo/EDA/central tendancy.jpg" alt="central tendency estimators" width="400px"> <img src="https://github.com/ashen007/Medical_Cost_prediction/blob/master/demo/EDA/std error of central tendancy.jpg" alt="standard error of estimator" width="400px">

***distribution analysis***

chi-squared goodness of fit test done on every discrete and continuous feature to estimate the best distribution to match
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

three types of testing methods are used to test the normality of distribution. Q-Q-plots and Anderson darling, wilk Shapiro.

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

box plots used to identify outliers in the data. in fraud, detection outliers are used to identify unusual behaviours
so that purpose outliers should be kept in the dataset.

<img src="https://github.com/ashen007/Medical_Cost_prediction/blob/doc/demo/EDA/boxplot.png" alt="outliers">

## feature standardization, transformation and selection

three methods used to standardized features which are -

- quantile transformation
- box-cox transformation
- yoe-Johnson transformation

scaling is done by the robust scaling method because there are outliers in data.

![](demo/Model%20Creation/age_box_cox.png)
![](demo/Model%20Creation/bmi_yj_trans.png)
![](demo/Model%20Creation/charges_power_(yj)_trans.png)

scaling in fraud detection dataset:

![](demo/Fraud%20Detection/scaling_methods_comp.png)

## diamentional reduction

fraud detection data has two separate data on patients which are medical information and beneficiary information about
A certain insurance claims. these two combine using claim id. in that case it has 53 features and 40474 records. Random forest
classifier used with Recursive feature elimination cross-validation to identify optimized feature count. k best method use
to get the best k features for models.

![](demo/Fraud%20Detection/rfecv_feture_selection.png)
![](demo/Fraud%20Detection/kbest_feture_selection.png)

## models

- medical cost prediction

coefficient of determination or R2, It measures the amount of variance of the prediction which is explained by the
dataset. R2 values close to 1 mean an almost-perfect regression, while values close to 0 (or negative) imply a bad
model.

![](demo/Model%20Creation/all%20models%20evaluation.png)

- linear models - ordinary least square, lasso (L1), ridge (L2), elastic net, RANSAC, Huber

![](demo/Model%20Creation/compare%20real_regularization_robust.png)

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

- non-linear models - polynomial ridge, k nearest neighbours, decision tree

![](demo/Model%20Creation/the%20best%20tree%20compare%20the%20best%20knn.png)
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

- insurance fraud detection

The first section of the classifier pipeline is feature selection using select best scoring by `mutual_info_classif`
to choose 10 out of 48 features. The next section is the grid search for tuning hyper-parameters of the underlying model.
it can be linear, tree, polynomial or ensemble. scoring methods used in grid search was `precision`, `recall`,
`f1` and `auc`. to re-fit the model `f1` and `auc` used.

#### tested models

  - `LogisticRegression`

![](demo/Fraud%20Detection/compare%20splits%20scores.png)

  - `SGDClassifier`

![](demo/Fraud%20Detection/compare%20splits%20scores%20model_2.png)

  - `LogisticRegression` with `Polynomial` function

![](demo/Fraud%20Detection/compare%20splits%20scores%20model_3.png)

  - `SVC`

![](demo/Fraud%20Detection/compare%20splits%20scores%20model_4.png)

  - `KNeighborsClassifier`

![](demo/Fraud%20Detection/compare%20splits%20scores%20model_5.png)

  - `DecisionTreeClassifier`

![](demo/Fraud%20Detection/compare%20splits%20scores%20model_6.png)

  - `ExtraTreeClassifier`

![](demo/Fraud%20Detection/compare%20splits%20scores%20model_7.png)

  - `RandomForestClassifier`

![](demo/Fraud%20Detection/compare%20splits%20scores%20model_8.png)

  - `AdaBoostClassifier`

![](demo/Fraud%20Detection/compare%20splits%20scores%20model_9.png)

  - `BaggingClassifier`

![](demo/Fraud%20Detection/compare%20splits%20scores%20model_10.png)

  - `VotingClassifier`

![](demo/Fraud%20Detection/compare%20splits%20scores%20model_11.png)

<table>
<tbody>
<tr style="background-color: darkgray; text-align: center;">
<td><h4>model</h4></td>
<td><h4>score in train data</h4></td>
<td><h4>score in test data</h4></td>
</tr>
<tr>
<td>LogisticRegression</td>
<td>0.73</td>
<td>0.73</td>
</tr>
<tr>
<td>SGDClassifier</td>
<td>0.68</td>
<td>0.68</td>
</tr>
<tr>
<td>LogisticRegression with Polynomial function</td>
<td>0.73</td>
<td>0.73</td>
</tr>
<tr>
<td>SVC</td>
<td>0.73</td>
<td>0.73</td>
</tr>
<tr>
<td>KNeighborsClassifier</td>
<td>0.73</td>
<td>0.82</td>
</tr>
<tr>
<td>DecisionTreeClassifier</td>
<td>0.91</td>
<td>0.92</td>
</tr>
<tr>
<td>ExtraTreeClassifier</td>
<td>0.73</td>
<td>0.73</td>
</tr>
<tr>
<td>RandomForestClassifier</td>
<td>0.87</td>
<td>0.93</td>
</tr>
<tr>
<td>AdaBoostClassifier</td>
<td>0.97</td>
<td>1</td>
</tr>
<tr>
<td>BaggingClassifier</td>
<td>0.91</td>
<td>0.92</td>
</tr>
<tr>
<td>VotingClassifier</td>
<td>0.95</td>
<td>0.97</td>
</tr>
</tbody>
</table>

## Dataset

dataset use in here is medical cost prediction dataset from **kaggle**. you can find it
from <a href="https://www.kaggle.com/mirichoi0218/insurance">here</a>.

## how to contribute
- you can improve models
- modify readme file to more clear look
- feature engineer data
