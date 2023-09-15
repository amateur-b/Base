---
description: Kaggle Learning Outcomes(Intermediate Level)
---

# Intermediate ML

`pd.read_csv('..../path/to/file.csv',index_col='ID')`\
The index\_col is included in every subset of the data frame

#### Importance of random\_state

Here’s an example of how to use `train_test_split`:

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
```

In the example above, `X` and `y` are the dataset to be split, and `test_size` is the proportion of the dataset to be allocated to the testing set. The remaining data is used for training.

Once the data is split, you can use the subsets to train and evaluate your model. However, the results you obtain may differ each time you run the code. This is where “random\_state” comes in.

“random\_state” is a parameter in `train_test_split` that controls the random number generator used to shuffle the data before splitting it. In other words, it ensures that the same randomisation is used each time you run the code, resulting in the same splits of the data.

Let’s look at an example to demonstrate this. Suppose you have a dataset of 1000 samples, and you want to split it into a training set of 700 samples and a testing set of 300 samples. Here’s how to do it:

```python
from sklearn.model_selection import train_test_split
import numpy as np

data = np.arange(1000)
X_train, X_test, y_train, y_test = train_test_split(data, data, test_size=0.3, random_state=42)
```

In the example above, we set “random\_state” to 42. This means that each time we run the code, the data will be shuffled in the same way, resulting in the same training and testing subsets.

#### n\_estimator Parameter&#x20;

`model1=RandomForestRegressor(n_estimators=100,random_state=0)`

the **n\_estimators** parameter decides the number of trees in the forest. The default is 100.

{% hint style="info" %}
In the given Python code snippet:

```python
print("Model %d MAE: %d" % (i+1, mae))
```

The code is using string formatting with the `%` operator to create a formatted string for printing. Let's break down the syntax step by step:

1. `"Model %d MAE: %d"`: This is a string containing placeholders marked by `%d`. These placeholders are used to indicate where values should be inserted into the string. `%d` is a placeholder for integer values.
2. `% (i+1, mae)`: After the string, there is a `%` operator followed by a tuple `(i+1, mae)`. This tuple contains the values that will be inserted into the placeholders in the string. In this case, `(i+1, mae)` provides two values to replace the two `%d` placeholders in the string.
   * `i+1` is an expression that calculates the value of `i+1`. It appears that `i` is some variable, and you are adding 1 to it.
   * `mae` is another variable, which is presumably an integer representing the Mean Absolute Error.
3. `print()`: This is the Python built-in function used to display text or values in the console.

So, when you execute this line of code, it will print a formatted string where `%d` placeholders are replaced with the values of `(i+1)` and `mae`. For example, if `i` is 2 and `mae` is 10, the output will be something like:

```
Model 3 MAE: 10
```

The `%d` placeholders are replaced with the values `3` (result of `i+1`) and `10` (value of `mae`).
{% endhint %}

#### Saving Output to External File

```python
// Saving Output

# Fit the model to the training data
my_model.fit(X, y)

# Generate test predictions
preds_test = my_model.predict(X_test)

# Save predictions in format used for competition scoring
output = pd.DataFrame({'Id': X_test.index,
                       'SalePrice': preds_test})
output.to_csv('submission.csv', index=False)
```

## Missing Values

1. Drop the columns with missing values
2. Imputation
3. Extended Imputation

### Drop Columns

```python
// 
# Get names of columns with missing values
cols_with_missing = [col for col in X_train.columns
                     if X_train[col].isnull().any()]

# Drop columns in training and validation data
reduced_X_train = X_train.drop(cols_with_missing, axis=1)
reduced_X_valid = X_valid.drop(cols_with_missing, axis=1)

print("MAE from Approach 1 (Drop columns with missing values):")
print(score_dataset(reduced_X_train, reduced_X_valid, y_train, y_valid))
```

### Imputation

> Imputation: A method of inserting values into missing cells in the dataset.

```python
// Imputation
from sklearn.impute import SimpleImputer
#Create Instance
my_imputer=SimpleImputer()
imputed_train_X=pd.DataFrame(my_imputer.fit_transform(train_X))
imputer_val_X=pd.DataFrame(my_imputer.fit_transform(val_X))

#Imputation Removes Column Names.Put them back
imputed_train_X.columns=train_X.columns
imputed_val_X.columns=val_X.columns

```

### Extended Imputation

```python
// 
# Make a copy of original data
train_X_plus=train_X.copy()
val_X_plus=val_X.copy()
#Make new columns indicating what will be imputed
for col in cols_with_missing:
    train_X_plus[col + '_was_missing']=train_X_plus[col].isnul()
    val_X_plus[col + '_was_missing']=val_X_plus[col].isnul()
#Imputation
imputed_train_X_plus=pd.DataFrame(my_imputer.fit_transform(train_X_plus))
imputed_val_X_plus=pd.DataFrame(my_imputer.fit_transform(val_X_plus))

#Imputation removes column names. Add the columns headings
imputed_train_X_plus.columns=train_X_plus.columns
```

{% hint style="info" %}
Since there are relatively few missing entries in the data (the column with the greatest percentage of missing values is missing less than 20% of its entries), we can expect that dropping columns is unlikely to yield good results. This is because we'd be throwing away a lot of valuable data, and so imputation will likely perform better.
{% endhint %}

{% hint style="info" %}
In scikit-learn's `SimpleImputer` class, both the `fit_transform` and `transform` methods are used for imputing missing values in a dataset. However, they are used in slightly different ways:

1.  `fit_transform` method:

    * The `fit_transform` method is used to both fit the imputer to the data and perform the imputation in a single step.
    * When you call `fit_transform`, the imputer computes the imputation strategy (e.g., mean, median, most frequent) based on the data passed to it and then replaces missing values with the computed values.
    * This method is typically used on the training dataset because it allows the imputer to learn the imputation strategy from the training data and apply it to fill missing values.

    Example:

    ```python
    imputed_data = imputer.fit_transform(X_train)
    ```
2.  `transform` method:

    * The `transform` method, on the other hand, is used after the imputer has been fitted to the training data.
    * It applies the previously learned imputation strategy to fill missing values in a new dataset without re-computing the strategy.
    * This method is typically used on validation and test datasets to ensure that the same imputation strategy is applied consistently.

    Example:

    ```python
    imputed_validation_data = imputer.transform(X_valid)
    ```

In summary:

* `fit_transform` is used when you want to both fit the imputer to your training data and perform the imputation in one step. It's commonly used on the training dataset.
* `transform` is used when you have already fitted the imputer to the training data and want to apply the same imputation strategy to other datasets, like validation or test datasets.

Using `fit_transform` on the training data and `transform` on other datasets ensures that the imputation is consistent across all datasets, which is important for machine learning models to make accurate predictions on new, unseen data.
{% endhint %}

## Categorical Variables

### 1. Drop Categorical variables

### 2. Ordinal Encoding

<figure><img src=".gitbook/assets/image.png" alt=""><figcaption></figcaption></figure>

assigning each uniques value to an integer in a way that explains the hierarchy of categorical values. "Every Day">"Most Days">Rarely">"Never"

Not all categorical variables have a clear ordering in the values, but we refer to those that do as **ordinal variables**. For tree-based models (like decision trees and random forests), you can expect ordinal encoding to work well with ordinal variables.

### 3. One-hot Encoding

**One-hot encoding** creates new columns indicating the presence (or absence) of each possible value in the original data.

<figure><img src=".gitbook/assets/image (1).png" alt=""><figcaption></figcaption></figure>

In contrast to ordinal encoding, one-hot encoding _does not_ assume an ordering of the categories. Thus, you can expect this approach to work particularly well if there is no clear ordering in the categorical data . Such variables are called **Nominal Variables**

```python
//
#Get list of categorical variables
s = (X_train.dtypes == 'object')
object_cols = list(s[s].index)

print("Categorical variables:")
print(object_cols)
```

`from sklearn.preprocessing import OrdinalEncoder`

{% hint style="info" %}


```
# "Cardinality" means the number of unique values in a column
```
{% endhint %}
