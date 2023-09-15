---
description: Lessons in the intro course to ML at Kaggle
---

# Kaggle Intro to ML

### Preparing the Data

1. Import relvant class/modules\
   `import pandas as pd`
2. locate data file\
   `filePath=path/to/the/file`
3. Read the file\
   `priceData=pd.read_csv(filePath)`

{% code fullWidth="false" %}
```python
// Data Inference
priceData.describe() #Describes the price data
priceData.head() #Shows the first 5 rows of the data frame
priceData.dropna() #Drops the rows with missing values
//
```
{% endcode %}

4. `y=priceData.Price` \
   <mark style="background-color:yellow;">Here Price is a specific column in priceData data frame found out using the .describe function</mark>
5. `features=[location,age,area,condition]`\
   <mark style="background-color:yellow;">These are the particular columns in the priceData data frame revealed using .describe function</mark>.&#x20;
6. `X=priceData[features]`
7. **Splitting Data into train\_X,train\_y,val\_X,val\_y**
   1. `from sklearn.model_selection import train_test_split`
   2. `train_X,train_y,val_X,val_y=train_test_split(X)`



### Working with Model

1. Import relevant class containing the model\
   `from sklearn.tree import DecisionTreeRegressor`\
   `from sklearn.ensemble import RandomForestAgressor`
2.  Create an instance of the model. Eg: `rf_model=RandomForestAgressor(random_state=1)`

    <mark style="background-color:yellow;">whenever defining the model set parameter random\_state to some non-zero value</mark>
3. Train the model \
   `rf_model.fit(train_X,train_y)`
4. Predict Values\
   `values_predicted=re_model.predict(val_X)`
5. Check Accuracy\
   `from sklearn.metrics import mean_absolute_error`\
   `mae=mean_absolute_error(val_y,values_predicted)`

### `Trend Between Fitting and Accuracy`

![](.gitbook/assets/Screenshot\_20230913\_120131.png)\


&#x20;

