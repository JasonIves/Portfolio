# DATA*6100 FINAL PROJECT
## A neural network classifier for the Kaggle "What's Cooking?" competition

### Jason Ives

# 1. Data Pre-Processing

Before beginning data processing, I will configure the project environment, define key functions, and load the data.  Once that is complete I will construct a pipeline to manage data cleaning and encoding.

## 1.1. System Configuration

System configuration includes package imports and definition of key global constants.


```python
##REFERENCES
##SOURCE: General python syntax assistance provided by Google search's AI Overview and VSCode's Python extension code completion suggestions.
##SOURCE: https://www.markdownguide.org/cheat-sheet/

##IMPORT PACKAGES - STANDARD
import os
from datetime import date
import gc

##IMPORT PACKAGES - THIRD PARTY
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from itertools import compress


pd.options.mode.chained_assignment = None ##SOURCE: https://stackoverflow.com/questions/20625582/how-to-deal-with-settingwithcopywarning-in-pandas - use df.loc[...] instead of direct cell assignment to ensure data is updated

##GLOBAL CONSTANTS
##DEFINE DATA FILE LOCATIONS
dataDirectory = "data_files"
trainData = "train.json"
testData = "test.json"

##COLUMN NAMES LIST, WILL BE UPDATED WITH TRAINING COLUMNS AFTER FIT FOR USE IN TEST PREDICTIONS
trainingColumns = []

##HYPERPARAMETER STORAGE VARIABLES
lowFreqCutoff = 5

##DEFINE NUMBER OF CORES TO MAKE AVAILABLE FOR PROCESSING
numCores = max(1, os.cpu_count() // 2)

```

## 1.2. Functions

In this section I will define key functions, broken down into 4 main categories.

- Convenience functions
- Data processing functions
- Analysis functions
- Plot functions

### 1.2.1. Convenience functions


```python
##CONVENIENCE FUNCTIONS
##WRITE TEST DATA TO FILE-------------------------------------------------------------
def WriteTestData(data, revision, byDate = True, type = "csv"):
    folderLvl1 = "submissions"
    folderLvl2 = date.today().strftime("%Y%m%d") if byDate else "Latest"
    folderLvl3 = revision

    path = os.path.join(os.getcwd(), folderLvl1, folderLvl2, folderLvl3)
    os.makedirs(path, exist_ok=True)

    ##DEFINE FILE NAME
    if type == "6100_kaggle_csv":
        fileName = "test_predictions.csv"
        writeData = f"""id,cuisine\n"""
        writeData += "\n".join([f"{row['id']},{row['cuisine']}" for index, row in data.iterrows()])
    elif type == "calculator":
        fileName = "calculator.py"
        writeData = f"""def my_answer_list():
    return({data})"""

    filePath = os.path.join(path, fileName)

    with open(filePath, 'w') as file:
        file.write(writeData)
```

### 1.2.2. Data processing functions


```python
##RUN DATAFRAME THROUGH PIPELINE AND RETURN FORMATTED DATAFRAME-------------------------------------------------------------
def DfFromPipeline(data, steps, idToIndex = True):
    pipeline = Pipeline(steps)
    processedData = pipeline.fit_transform(data)

    if idToIndex:
        dataDf = pd.DataFrame(processedData)
    else:
        dataDf = pd.DataFrame(processedData).reset_index(drop = False)

    return(dataDf)

##SET UNSEEN INGREDIENTS TO OTHER, AND ENSURE ALL INGREDIENT COLUMNS ARE PRESENT-------------------------------------------------------------
def OtherizeUnseenIngredients(df, pipeline, keyCol, targetCol, knownIngredients):
    
    ##APPLY PIPELINE TRANFORMATION TO DATAFRAME
    df = DfFromPipeline(df, pipeline)

    ##RECODE UNSEEN INGREDIENTS AS 'OTHER'
    df.loc[~df[targetCol].isin(knownIngredients), targetCol] = "other"

    ##REAGGREGATE TO SINGLE COLUMN LIST
    df = pd.DataFrame(df.groupby(keyCol, sort = False)[targetCol].apply(list).reset_index(False))
    
    ##ADD DUMMY RECIPE TO ENSURE ALL INGREDIENT COLUMNS ARE PRESENT
    df.loc[len(df)] = [-1, knownIngredients + ['other']]
        

    return(df)
```

### 1.2.3. Analysis functions


```python
##ANALYSIS FUNCTIONS
##SINGLE OBSERVATION DETAILS, IN A VERTICAL DISPLAY-------------------------------------------------------------
def SingleObservationDetails(obs):
    if len(obs) != 1:
        print("Only single observations are supported.")
        return
    cols = obs.columns.tolist()
    vals = obs.iloc[0].tolist()
    vertObs = pd.DataFrame({'Parameter': cols, 'Value': vals})
    display(vertObs)

##QUICK COLUMN DETAILS-------------------------------------------------------------
def QuickDetails(col):
    print("========================")
    print(f"Column: {col.name}")
    print("")
    print("Unique values:")
    [print(f"{x}") for x in col.unique()[:10]]
    if len(col.unique()) > 10:
        print("...")
    if col.dtype in ['int64', 'float64']:
        display(col.describe())
    else:
        display(col.value_counts())
    print(f"Number of nulls: {col.isnull().sum()}")
    print("========================")


##TWO COLUMN COMPARISON-------------------------------------------------------------
def TwoColCompare(df, col1, col2):
    display(df[col1].notnull().corr(df[col2] > 0))
    display(df.groupby([col1, col2]).size())

##COUNT NULLS BY COLUMN-------------------------------------------------------------
def NullCountByColumn(df):
    paramsWithNull = [(df.columns[i], df[df.columns[i]].isnull().sum()) for i in range(len(df.columns)) if (df[df.columns[i]].isnull().sum() > 0 and df.columns[i] != 'SalePrice')]
    return(pd.DataFrame(paramsWithNull, columns = ('parameter', 'nullCount')).sort_values(by='nullCount', ascending=False))
```

### 1.2.4. Plot functions


```python
##PLOT FUNCTIONS
def QuickHist(col, bins = 20):
    plt.hist(col, bins=bins)
    plt.title(f"Histogram of {col.name}")
    plt.xlabel(col.name)
    plt.ylabel("Frequency")
    plt.show()


##CUSTOM BOXPLOT PLOT WITH FILTERING AND NORMALIZATION FOR NUMERIC-------------------------------------------------------------
def PlotNormalizedBoxplot(df, filter):
    ##FILTER VALID VALUES: 'numeric'
    if filter == 'numeric':
        scaler = StandardScaler()
        filterCols = df.dtypes[(df.dtypes == 'int64') | (df.dtypes == 'float64')].index.tolist()
        plt.boxplot(scaler.fit_transform(df[filterCols]), tick_labels = filterCols, vert = False)

    else:
        filterCols = df.columns.tolist()
        plt.boxplot(df[filterCols], tick_labels = filterCols, vert = False)

    plt.title("Standardized Numeric Parameter Ranges")
    plt.show()


##CUSTOM HEATMAP PLOT WITH FILTERING AND FOCUS TYPE-------------------------------------------------------------
def PlotHeatmap(df, filter, pType = 'nulls'):
    ##PTYPE VALID VALUES: 'nulls'
    ##FILTER VALID VALUES: 'numeric'
    if filter == 'numeric':
        filterCols = df.dtypes[(df.dtypes == 'int64') | (df.dtypes == 'float64')].index.tolist()
    else:
        filterCols = df.columns.tolist()

    if pType == 'nulls':
        plt.imshow(df[filterCols].isnull(), origin = 'lower', aspect = 'auto', interpolation = 'nearest', cmap = 'gray_r')
        plt.title("Null Value Heatmap")
        plt.show()
    elif pType == 'zeros':
        plt.imshow(df[filterCols] == 0, origin = 'lower', aspect = 'auto', interpolation = 'nearest', cmap = 'gray_r')
        plt.title("Zero Value Heatmap")
        plt.show()
    else:
        print("Invalid plot type")
```


```python
rawTrain = pd.read_json(os.path.join(dataDirectory, trainData))
rawTest = pd.read_json(os.path.join(dataDirectory, testData))
```

## 1.3. Creating lists to capture transformation steps

Since I will be using a sciki-learn pipeline to manage my data preprocessing, I will create lists to capture the data transformation steps as I create them.  The lists can then be added to the pipeline after preprocessing is complete.


```python
##EMPTY EMPTY STEP LISTS TO PRIME THE PIPELINES
cleaningSteps = []
encodingSteps = []
```

## 1.4. Data load

Loading the json data to training and test data frames.


```python
##READ IN RAW TRAINING AND TEST DATA
rawTrain = pd.read_json(os.path.join(dataDirectory, trainData))
rawTest = pd.read_json(os.path.join(dataDirectory, testData))
```

## 1.5. Data exploration

I will explore the data structure and ranges, and ensure the outcome doesn't have any null values in the training data.

### 1.5.1. Examining the training dataframe

A quick look at the head of the training dataframe will allow me to familiarize myself with the data structure.

---

*Examining the training data reveals that the target variable is **cuisine***

*It is accompanied by a single column of ingredient lists, and an ID column*


```python
display(rawTrain.head())
QuickDetails(rawTrain['id'])
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>cuisine</th>
      <th>ingredients</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10259</td>
      <td>greek</td>
      <td>[romaine lettuce, black olives, grape tomatoes...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>25693</td>
      <td>southern_us</td>
      <td>[plain flour, ground pepper, salt, tomatoes, g...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>20130</td>
      <td>filipino</td>
      <td>[eggs, pepper, salt, mayonaise, cooking oil, g...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>22213</td>
      <td>indian</td>
      <td>[water, vegetable oil, wheat, salt]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>13162</td>
      <td>indian</td>
      <td>[black pepper, shallots, cornflour, cayenne pe...</td>
    </tr>
  </tbody>
</table>
</div>


    ========================
    Column: id
    
    Unique values:
    10259
    25693
    20130
    22213
    13162
    6602
    42779
    3735
    16903
    12734
    ...
    


    count    39774.000000
    mean     24849.536959
    std      14360.035505
    min          0.000000
    25%      12398.250000
    50%      24887.000000
    75%      37328.500000
    max      49717.000000
    Name: id, dtype: float64


    Number of nulls: 0
    ========================
    

### 1.5.2. Checking the variables for nulls and preparing the data for cleaning

I will check to see if there is any null data.  To ensure all the cases in the training data are valid, I will also check for nulls in the target / outcome variable.

---

*There are no null values in either of the columns, so data pre-processing can move forward and focus on the ingredients columns and prepare the predictor and outcome data for modeling.*


```python
NullCountByColumn(rawTrain)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>parameter</th>
      <th>nullCount</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>



## 1.6. Data cleaning

Data cleaning for this project will focus on formatting and preparing the ingredients column for the modeling process.

### 1.6.1. Restructuring the data for data cleaning

The data needs to be broken out into separate predictor and outcome data sets.  Then the predictor data can be restructured into a long form  that will allow effective standardization of the cleaning process.


```python
##PREPARE PREDICTOR DATA FOR PIPELINE MODELING
xTrain = rawTrain.drop(columns = ['cuisine'])
xTest = rawTest


##PREPARE OUTCOME DATA FOR PIPELINE MODELING
yEncoder = LabelEncoder()
yTrainEncoded = yEncoder.fit_transform(rawTrain['cuisine'])
```


```python
##EXPAND INGREDIENTS LIST TO LONG FORMAT
def ExpandListCol(df, colName):
    longDf = df.explode(colName).reset_index(False)
    return longDf

##ADD LONG EXPANSION TO PIPELINE AND REVIEW
transformDfLong = FunctionTransformer(ExpandListCol, kw_args={'colName': 'ingredients'})

cleaningSteps.append(('expandIngredients', transformDfLong))

cleaningDf = DfFromPipeline(xTrain, cleaningSteps)

display(cleaningDf.head(10))

```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>index</th>
      <th>id</th>
      <th>ingredients</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>10259</td>
      <td>romaine lettuce</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>10259</td>
      <td>black olives</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>10259</td>
      <td>grape tomatoes</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>10259</td>
      <td>garlic</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>10259</td>
      <td>pepper</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0</td>
      <td>10259</td>
      <td>purple onion</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0</td>
      <td>10259</td>
      <td>seasoning</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0</td>
      <td>10259</td>
      <td>garbanzo beans</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0</td>
      <td>10259</td>
      <td>feta cheese crumbles</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1</td>
      <td>25693</td>
      <td>plain flour</td>
    </tr>
  </tbody>
</table>
</div>


### 1.6.2. Assessing the ingredients values

I will extract and review the ingredients values to check for inconsistencies, spelling issues, and variability in formatting.

---

*The ingredients column appears to be fairly clean but has a very broad range of values, 6714 total.*

*Several key things will need to be addressed:*
- *Punctuation*
- *Plural vs singular*
- *Multi-word ingredient handling*
- *Low-frequency cardinanlity*


```python
##REVIEW COUNTS OF INGREDIENTS
ingredientVals = cleaningDf['ingredients'].value_counts()
display(ingredientVals.head(10))
display(ingredientVals.tail(10))

len(ingredientVals)

##CHECK HISTOGRAM OF INGREDIENT COUNTS
QuickHist(ingredientVals, bins = 50)
```


    ingredients
    salt                   18049
    olive oil               7972
    onions                  7972
    water                   7457
    garlic                  7380
    sugar                   6434
    garlic cloves           6237
    butter                  4848
    ground black pepper     4785
    all-purpose flour       4632
    Name: count, dtype: int64



    ingredients
    cumberland sausage                                   1
    cocktail pumpernickel bread                          1
    chunky tomatoes                                      1
    Colman's Mustard Powder                              1
    manouri                                              1
    cherry vanilla ice cream                             1
    bone-in ribeye steak                                 1
    frozen lemonade concentrate, thawed and undiluted    1
    flowering chinese chives                             1
    Kraft Slim Cut Mozzarella Cheese Slices              1
    Name: count, dtype: int64



    
![png](IvesJ_DATA6100_F25_FinalProject_files/IvesJ_DATA6100_F25_FinalProject_28_2.png)
    


### 1.6.3. Addressing punctuation and non-standard characters in ingredients

The ingredients column has a number of elements that use punctuation.  A review of these cases will provide context, as I look to standardize the values.


```python
##CHECK FOR USAGE OF NON-STANDARD CHARACTERS IN INGREDIENT NAMES
nonStandardChars = ingredientVals[ingredientVals.index.str.contains(r'[^a-zA-Z0-9\s\-]')]
display(nonStandardChars)
display(len(nonStandardChars))
```


    ingredients
    half & half                                            337
    tomato purée                                           217
    1% low-fat milk                                        193
    crème fraîche                                          145
    cream cheese, soften                                   143
                                                          ... 
    Lipton® Iced Tea Brew Family Size Tea Bags               1
    Hidden Valley® Greek Yogurt Original Ranch® Dip Mix      1
    Honeysuckle White® Hot Italian Turkey Sausage Links      1
    Colman's Mustard Powder                                  1
    frozen lemonade concentrate, thawed and undiluted        1
    Name: count, Length: 225, dtype: int64



    225


#### 1.6.3.1. Targeted replacement of punctuation

255 ingredient values use punctuation or other non-standard characters.  Some should be removed or replaced, while others seem key to model interpretation.  In general language specific characters will be left in place, while structural characters will be replaced or removed.


```python
##TARGETED CLEANUP OF PUNCTUATION IN INGREDIENT NAMES
def PunctuationCleanup(df, colName):
    df.loc[:, colName] = df[colName].str.replace(r'&', ' and ', regex = False)
    df.loc[:, colName] = df[colName].str.replace(r'®', '', regex = False)
    df.loc[:, colName] = df[colName].str.replace(r'%', ' percent ', regex = False)
    df.loc[:, colName] = df[colName].str.replace(r'\'', '', regex = True)
    df.loc[:, colName] = df[colName].str.replace(r'™', '', regex = False)
    df.loc[:, colName] = df[colName].str.replace(r',', ' ', regex = False)
    df.loc[:, colName] = df[colName].str.replace(r'\s+', ' ', regex = True)
    df.loc[:, colName] = df[colName].str.replace(r'-', ' ', regex = False)
    df.loc[:, colName] = df[colName].str.strip()

    return(df)

##ADD PUNCTUATION CLEANUP TO PIPELINE
transformPunctuation = FunctionTransformer(PunctuationCleanup, kw_args={'colName': 'ingredients'})

cleaningSteps.append(('punctuationCleanup', transformPunctuation))

#cleaningDf = DfFromPipeline(xTrain, cleaningSteps)

```

### 1.6.4. Standardizing letter case

Uneven capitalization could cause otherwise equivalent values to be treated as different by the model


```python
##SET ALL STRINGS TO LOWERCASE
def LowercaseIngredients(df, colName):
    df.loc[:, colName] = df[colName].str.lower()
    return(df)

##ADD LOWERCASE TRANSFORMATION TO PIPELINE
transformLowercase = FunctionTransformer(LowercaseIngredients, kw_args={'colName': 'ingredients'})

cleaningSteps.append(('setLowercase', transformLowercase))

#cleaningDf = DfFromPipeline(xTrain, cleaningSteps)
```

### 1.6.5. Removing plurality

Inconsistent pluarlization can also promote uneven encoding in the ingredients, so common pluarlization characters will be removed or recoded.


```python
##REMOVE AND RECODE PLURAL AND RELATED VARAIATIONS TO STANDARDIZE
def RemovePlural(df, colName):
    df.loc[:, colName] = df[colName].str.rstrip(r'es')
    df.loc[:, colName] = df[colName].str.rstrip(r's')
    df.loc[:, colName] = df[colName].str.replace(r'es\s', ' ', regex = True)
    df.loc[:, colName] = df[colName].str.replace(r's\s', ' ', regex = True)
    df.loc[:, colName] = df[colName].str.replace(r'ie\s', 'y ', regex = True)
    df.loc[:, colName] = df[colName].str.replace(r'i\s', 'y ', regex = True)
    df.loc[:, colName] = df[colName].str.replace(r'ie$', 'y', regex = True)
    df.loc[:, colName] = df[colName].str.replace(r'i$', 'y', regex = True)
    
    return(df)

##ADD SPACE REMOVAL TO PIPELINE
transformPlural = FunctionTransformer(RemovePlural, kw_args={'colName': 'ingredients'})

cleaningSteps.append(('removePlural', transformPlural))

# cleaningDf = DfFromPipeline(xTrain, cleaningSteps)
```

### 1.6.6. Compress ingredient strings

To finalize the ingredient transformations, spaces will be removed from the ingredient strings.  This will further reduce variability among equivalent entries.


```python
##COMPRESS INGREDIENT STRINGS TO REMOVE SPACES
def RemoveSpaces(df, colName):
    df.loc[:, colName] = df[colName].str.replace(r'\s+', '', regex = True).str.strip()
    return(df)

##ADD SPACE REMOVAL TO PIPELINE
transformSpacing = FunctionTransformer(RemoveSpaces, kw_args={'colName': 'ingredients'})

cleaningSteps.append(('removeSpaces', transformSpacing))

# cleaningDf = DfFromPipeline(xTrain, cleaningSteps)
```

## 1.7. Encoding

### 1.7.1. Recoding low frequency ingredients

Many of the ingredients occur with very low frequency.  These low frequency items can create overfitting and add noise to the model.  They also have performance overhead implications that need to be considered.  A multi-stage encoding process will work to address this.

1. Low frequency items will be identified, based on a % of total based cutoff which is a cross-validated hyperparameter.
2. Low frequency strings will be scanned for a shared ingredient phrase with the high frequency strings.  If one is found, the low frequency string will be recoded to the high frequency string.
3. If no match among the high frequency strings is found, the low frequency string will be recoded to "other".

---

*5% was found to be an optimal cutoff value during cross-validation*


```python
##RECODE LOW FREQUENCY ITEMS TO OTHER OR HIGH FREQUENCY SUBSTRING MATCH
def RecodeLowFreq(df, countCol, keyCol, cutoffPct = 1):
    freqCounts = pd.DataFrame(df[countCol].value_counts().sort_values(ascending = False))
    ##ONLY RECODE FOR TRAINING DATA, DATA FOR PREDICTIONS WILL BE "OTHERIZED" SEPARATELY
    if -1 not in df[keyCol].values:
        freqCounts['cumulativePct'] = (freqCounts['count'].cumsum() / freqCounts['count'].sum()) * 100
        lowFreqIngredients = freqCounts[freqCounts['cumulativePct'] >= (100 - cutoffPct)].index.tolist()
        highFreqIngredients = freqCounts[freqCounts['cumulativePct'] < (100 - cutoffPct)].index.tolist()
        matchedHighFreqIngredients = list(compress(highFreqIngredients, [any(hf in lf for lf in lowFreqIngredients) for hf in highFreqIngredients]))
        
        ##FOR LOW FREQUENCY INGREDIENTS THAT CONTAIN A HIGH FREQUENCY SUBSTRING, RECODE TO SUBSTRING
        for ingred in matchedHighFreqIngredients:
            df.loc[df[countCol].isin(lowFreqIngredients) & df[countCol].str.contains(ingred, regex = False), countCol] = ingred

        ##ENCODE REMAINING LOW FREQUENCY INGREDIENTS AS 'OTHER'
        df.loc[df[countCol].isin(lowFreqIngredients), countCol] = "other"

    return(df)

##ADD LOW FREQUENCY DROP TO PIPELINE
transformDropLowFreq = FunctionTransformer(RecodeLowFreq, kw_args={'countCol': 'ingredients', 'keyCol': 'id', 'cutoffPct': lowFreqCutoff})

encodingSteps.append(('dropLowFrequencyIngredients', transformDropLowFreq))

# pipelineSteps = cleaningSteps + encodingSteps
# cleaningDf = DfFromPipeline(xTrain, pipelineSteps)
```

### 1.7.2. Encode the long-form data into wide-form

The long-form ingredient data must be converted to a wide format for modeling.  In this phase the ingredients column will be one-hot encoded to create a sparse long and wide table, one with the encodings for each recipe spread out among a number of lines.  In the next step this table will be compressed into a one-row-per-observation format.

---

*WARNING: Do not attmpt to view this data frame outside of the pipeline, it is very large and can cause crashing and other performance issues.*


```python
##ONE HOT ENCODE THE INGREDEIENTS.  NOT YET COMPRESSED INTO ONE ROW PER RECIPE
def OneHotEncoderWrapper(dfLong, targetCol, keyCol):
    ohEncoder = OneHotEncoder(handle_unknown = 'ignore', drop = None)
    encodedIngredients = ohEncoder.fit_transform(dfLong[[targetCol]])

    dfLongWide = pd.DataFrame(encodedIngredients.toarray(), columns = ohEncoder.get_feature_names_out([targetCol]))
    dfLongWide.insert(loc = 0, column = keyCol, value = dfLong[keyCol].values)

    return(dfLongWide)

oneHotIngredients = FunctionTransformer(OneHotEncoderWrapper, kw_args={'targetCol': 'ingredients', 'keyCol': 'id'})

encodingSteps.append(('oneHotIngredients', oneHotIngredients))
```

### 1.7.3. Compress long data into a single row per observation

The long and wide format can now be compressed to reduce the length to 1 row per observation.


```python
##COMPRESS ROWS TO SINGLE ROW PER RECIPE
def CompressRows(df, keyCol):
    df = df.groupby(df[keyCol], sort = False).max()

    return(df)

compressRows = FunctionTransformer(CompressRows, kw_args={'keyCol': 'id'})

encodingSteps.append(('compressRows', compressRows))

# pipelineSteps = cleaningSteps + encodingSteps
# cleaningDf = DfFromPipeline(xTrain, pipelineSteps)
```

### 1.7.4. Creating a neural network autoencoder for dimension reduction

The training data set with 5% of values recoded has over 1700 features.  This can affecting hardware performance, and I'm concerned it could be an issue for the model as well.  To do some testing with dimension reduction, I will create a prototype neural network autoencoder.  This will be a single layer neural network that transforms the features to a reduced dimension hidden layer, then decodes them back to the size of the original feature set.  The resulting output features are scored to assess how closely they resemble the inupts, using the binary cross entropy score (BCE).  The neural network then executes a training loop, working to improve that BCE score.  Once the training loop is complete, the weights and biases used to create the final hidden layer are extracted, and used to transform the data set into a lower dimension.

This autoencoder will be added to the modeling pipeline so the hyperparameters can be tuned and it's best fit can be stored by the pipeline.  My hope is that the resulting data set will have comparable modeling performance, and run more quickly and while consuming less hardware resources.

---

*The autoencoder was succesfully intergrated into the pipeline for parameter validation, but the cross entropy loss was significantly worse than the full feature set model across all hyperparameter tuning runs.  The best validation score came from a model with a very high number of features (little dimension reduction).  It had a negative CEL of **-1.287**, compared to a negative CEL of a bit over **1.0** for equivalently configured validation models with no dimension reduction, more than 20% worse than the baseline.*

*In addition, despite attempts at optimization, the cross-validation phase with the full training data set could not be executed with the autoencoder in the pipeine.  The operating system could not allocate sufficient reasources.  As such, the autoencoder has been removed from the modeling pipelines.  It can still be tested using step 1.7.4.1.*


```python
##AMPLE CREDIT TO GOOGLE GEMINI [FAST] FOR STEP BY STEP INSTRUCTIONAL

##TAKING A SHOT AT A NEURAL NETWORK AUTOENCODER CLASS FOR DIMENSION REDUCTION
class AutoencoderPrototype(BaseEstimator, TransformerMixin):
    def __init__(self, nHiddenLayer = 50, eta = 5e-8, epochs = 25):
        if epochs > 50:
            print("Warning: More than 50 epochs may lead to excess sparsity.  Limiting to 50.")
            epochs = 50

        self.nHiddenLayer = nHiddenLayer
        self.eta = eta
        self.epochs = epochs
        
        ##INITALIZE WEIGHTS AND BIASES IN INIT SO THEY PERSIST FOR LATER TRANSFORMS
        self.encodeWeights = None
        self.encodeBiases = None
        self.decodeWeights = None
        self.decodeBiases = None

        ##INITIALIZE LOSS DATAFRAME
        self.lossData = pd.DataFrame(columns = ['iteration', 'bceLoss'])

    ##FIT METHOD TO ESTABLISH WEIGHTS AND BIASES
    def fit(self, df, y = None):
        inputMatrix = df.values if isinstance(df, pd.DataFrame) else df
        if self.nHiddenLayer > inputMatrix.shape[1]:
            self.nHiddenLayer = inputMatrix.shape[1] - 1
        
        nFeatures = inputMatrix.shape[1]
        nObservations = inputMatrix.shape[0]

        ##INITIALIZE WEIGHTS AND BIASES
        self.encodeWeights = np.random.rand(nFeatures, self.nHiddenLayer) * .01
        self.encodeBiases = np.zeros((1, self.nHiddenLayer))
        self.decodeWeights = np.random.rand(self.nHiddenLayer, nFeatures) * .01
        self.decodeBiases = np.zeros((1, nFeatures))

        for iteration in range(self.epochs):
            ##-=FORWARD PASS=-
            ##CREATE HIDDEN LAYER WITHOUT RELU TRANFORMATION
            hiddenLayer = np.dot(inputMatrix, self.encodeWeights) + self.encodeBiases
            
            ##APPLY RELU ACTIVATION FUNCTION
            hiddenLayerActivated = self._relu(hiddenLayer)

            ##CREATE OUTPUT LAYER WITHOUT SIGMOID TRANFORMATION
            outputLayer = np.dot(hiddenLayerActivated, self.decodeWeights) + self.decodeBiases

            ##ACTIVATE OUTPUT LAYER (SIGMOID)
            outputLayerActivated = 1 / (1 + np.exp(-outputLayer))



            ##-=BACKWARD PASS=-
            ##CALCULATE ERROR (BCE LOSS FUNCTION)
            outputError = outputLayerActivated - inputMatrix

            ##CALCULATE DECODER GRADIENTS
            dDecodeWeights = np.dot(hiddenLayerActivated.T, outputError)
            dDecodeBiases = np.sum(outputError, axis = 0, keepdims = True)

            ##CALCULATE ERROR PASSED OUTPUT TO HIDDEN LAYER
            decoderToHiddenError = np.dot(outputError, self.decodeWeights.T)

            ##DEFINE RELU GRADIENT
            reluGradient = np.where(hiddenLayer > 0, 1, 0)

            ##CALCULATE ERROR PASSED FROM HIDDEN LAYER TO ENCODER WEIGHTS AND BIASES
            hiddenToEncoderError = decoderToHiddenError * reluGradient

            ##CALCULATE ENCODER GRADIENTS
            dEncodeWeights = np.dot(inputMatrix.T, hiddenToEncoderError)
            dEncodeBiases = np.sum(hiddenToEncoderError, axis = 0, keepdims = True)

            ##UPDATE WEIGHTS AND BIASES USING GRADIENT DESCENT
            self.encodeWeights -= self.eta * dEncodeWeights
            self.encodeBiases -= self.eta * dEncodeBiases
            self.decodeWeights -= self.eta * dDecodeWeights
            self.decodeBiases -= self.eta * dDecodeBiases

            ##CALCULATE AND PRINT BCE LOSS EVERY 10 ITERATIONS
            bceLoss = -np.mean(inputMatrix * np.log(outputLayerActivated + 1e-10) + (1 - inputMatrix) * np.log(1 - outputLayerActivated + 1e-10))
            self.lossData.loc[self.lossData.shape[0]] = [iteration + 1, bceLoss]

        return self

    ##TRANSFORM METHOD TO CREATE UPDATED DATAFRAME
    def transform(self, df):
        ##TRANSFORM DF USING TRAINED WEIGHTS AND BIASES
        transformationMatrix = df.values if isinstance(df, pd.DataFrame) else df
        reducedDim = self._relu(np.dot(transformationMatrix, self.encodeWeights) + self.encodeBiases)
        reducedDim = pd.DataFrame(reducedDim)
        reducedDim.columns = [f"dim_{i+1}" for i in range(reducedDim.shape[1])]
        reducedDim.index = df.index
        return(reducedDim)
    
    ##OUTPUT BCE LOSS INFO AND PLOT LOSS OVER TIME
    def metrics(self):
        plt.plot(self.lossData['iteration'], self.lossData['bceLoss'])
        plt.title("BCE Loss Over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("BCE Loss")
        plt.show()
    
    ##RELU HELPER METHOD
    def _relu(self, z):
        return np.maximum(0, z)
```

#### 1.7.4.1. Tuning the eta step size and epoch parameters

To ensure that the autoencoder is able to improve the model over all epochs without overstepping a minimum point and being set to 0 by the ReLU function, I will use the BCE loss score of each epoch to assess the threshold for overstepping - indicated by a sharp decline in BCE loss.  I will then adjuste the eta and number of epochs to balance performance, sparsity, and accuracy.

---

*After testing the following values were set:*

- *eta: 5e-8*
- *epochs: capped at 50*


```python
##TEST / DEMO AUTOENCODER ETA AND EPOCH TUNING
pipelineSteps = cleaningSteps + encodingSteps
cleaningDf = DfFromPipeline(xTrain, pipelineSteps)

dimReducer = AutoencoderPrototype(nHiddenLayer = 200, eta = 5e-8, epochs = 51)
test = dimReducer.fit_transform(cleaningDf.iloc[0:cleaningDf.shape[0]//20])

dimReducer.metrics()

##FREE UP MEMORY
del cleaningDf
gc.collect()
```

    Warning: More than 50 epochs may lead to excess sparsity.  Limiting to 50.
    


    
![png](IvesJ_DATA6100_F25_FinalProject_files/IvesJ_DATA6100_F25_FinalProject_49_1.png)
    





    16149



### 1.7.5. Finalize the preprocessing pipeline steps

To prepare for modeling the preprocessing pipeline steps can be combined.


```python
##COMBINE CLEANING AND ENCODING STEPS INTO A SINGLE PIPELINE
pipelineSteps = cleaningSteps + encodingSteps
```

# 2. Modelling

## 2.1. Create a neural network classifier

My primary model will be a neural network classifier.  I will use a 30% stratified sample to execute a grid cross-validated version of the model pipeline for hyperparameter tuning.  The optimized parameter values will then be passed to a final modeling pipeline.  That pipeline will first be used to model and predict 5-fold cross-validated data sets, for review of the modeling metrics.  Then the full training set will be used to fit the final modeling pipeline and make predictions based on the test data.

### 2.1.1. Process the training data

The training data will be processed using the cleaning and encoding pipeline.


```python
##RUN TRAINING DATA THROUGH CLEANING / ENCODING PIPELINE
xTrainFormatted = DfFromPipeline(xTrain, pipelineSteps, idToIndex = True)
```

### 2.1.2. Establish the modeling parameters

Before the modeling phase I will create a grid cross-validated model to refine the hyperparameters associated with the mlp classifier.  To avoid performance issues I will use a subset of the data to find the best combination of the hyperparameters, and use those values for the full modeling process.

#### 2.1.2.1. Subset the training data for model parameterization

The pre-processed training data will then be split using 70% / 30%.  The 70% sample will be discarded and the 30% sample will be used for grid cross-validated parameter tuning.


```python
##TAKE A 10% SAMPLE FOR PARAMETER TUNING
xDummy, xTrainParam, yDummy, yTrainParam = train_test_split(xTrainFormatted, yTrainEncoded, test_size = 0.3, stratify = yTrainEncoded, random_state = 9)

##DELETE UNEEDED DFS
del xDummy
del yDummy
gc.collect()

```




    2041



#### 2.1.2.2. Instantiate the neural network classifier

The multilayer perceptron classification model for the parameterization pipeline is defined.


```python
##INSTANTIATE MLP MODEL
paramModel = MLPClassifier(hidden_layer_sizes=(200,), 
                            activation='relu', 
                            max_iter=500, 
                            random_state=1, 
                            early_stopping=True, 
                            n_iter_no_change=10)
```

#### 2.1.2.3. Create hyperparmeter tuning pipeline 

The parameterization pipeline is created using the previously instantiated classifier.


```python
##CREATE PARAMETERIZATION PIPELINE
parameterizationPipeline = Pipeline([
    ('paramClassifier', paramModel)
])
```

#### 2.1.2.4. Define the hyperparameter tuning cross-validation grid

Values for the size and number of hidden layers, as well as the alpha smoothing parameter, are defined here, for assessment during cross-validation.

---

*A broader range of values was tested over several testing cycles.  The values below represent a narrowed range of optimized values for the final cross-validation cycle to assess.*


```python
mlpParameterGrid = {
    ##MLP CLASSIFIER PARAMETERS
    ##REDUCED ONCE OPTIMAL PARAMETERS WERE IDENTIFIED
    # 'paramClassifier__hidden_layer_sizes': [(400,), (500,), (600,), (700,), (800,), (900,), (1000,), (1100,), (1200,), (1300,)],
    # 'paramClassifier__alpha': [0.0001, 0.001, 0.01]
    'paramClassifier__hidden_layer_sizes': [(500,), (600,), (700,)], 
    'paramClassifier__alpha': [0.0001, 0.001]
    }
```

#### 2.1.2.5. Execute cross-validation for hyperparameter tuning

With the parameters and pipeline in place, the cross-validation cycle can begin.


```python
stratKFold = StratifiedKFold(n_splits=5, shuffle=True, random_state=99)

paramGrid = GridSearchCV(
    estimator = parameterizationPipeline, 
    param_grid = mlpParameterGrid, 
    cv = stratKFold, 
    scoring = ['neg_log_loss', 'accuracy'],
    refit = 'neg_log_loss', 
    n_jobs = numCores
)

paramGrid.fit(xTrainParam, yTrainParam)
```

#### 2.1.2.6. Review the results of the hyperparameter tuning cycle

Once the cross-validation process is complete, the results of the tuning cycle can be reviewed.  Reviewing this in aggregate can be informative and help in identifying patterns of strong or weak factors across the different validation runs.  These patterns of strength or weakness can help guide the set of hyperparamter values to be used in subsequent cycles.

---

*After the full tuning pass, the range of cross-validation parameters was narrowed for performance reasons.  The full results plot is loaded from file here:*
![Full range cross-validation plot](images/mlp_cv_results_01.png)


```python
##DISPLAY RESULTS DF AND BEST CE SCORE
paramCvResults = pd.DataFrame(paramGrid.cv_results_)
print(f"Best Mean Cross Entropy Loss: {paramCvResults['mean_test_neg_log_loss'][paramCvResults['rank_test_neg_log_loss'] == 1].values[0]}")
display(paramCvResults.sort_values(by='rank_test_neg_log_loss').head(10))

##DISPLAY A PLOT OF MEAN TEST LOG LOSS BY HIDDEN LAYER SIZE, BROKEN OUT BY ALPHA
for alpha in mlpParameterGrid['paramClassifier__alpha']:
    subsetResults = paramCvResults[paramCvResults['param_paramClassifier__alpha'] == alpha]
    plt.plot(subsetResults['param_paramClassifier__hidden_layer_sizes'].apply(lambda x: x[0]), subsetResults['mean_test_neg_log_loss'], label = f'Alpha: {alpha}')
plt.title("Mean Validation Cross-Entropy Loss - by Hidden Layer Size and Alpha")
plt.xlabel("Hidden Layer Size")
plt.ylabel("Cross-Entropy Loss (Negative, Higher is Better)")
plt.legend()
plt.show()

```

    Best Mean Cross Entropy Loss: -0.8625825636231529
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mean_fit_time</th>
      <th>std_fit_time</th>
      <th>mean_score_time</th>
      <th>std_score_time</th>
      <th>param_paramClassifier__alpha</th>
      <th>param_paramClassifier__hidden_layer_sizes</th>
      <th>params</th>
      <th>split0_test_neg_log_loss</th>
      <th>split1_test_neg_log_loss</th>
      <th>split2_test_neg_log_loss</th>
      <th>...</th>
      <th>std_test_neg_log_loss</th>
      <th>rank_test_neg_log_loss</th>
      <th>split0_test_accuracy</th>
      <th>split1_test_accuracy</th>
      <th>split2_test_accuracy</th>
      <th>split3_test_accuracy</th>
      <th>split4_test_accuracy</th>
      <th>mean_test_accuracy</th>
      <th>std_test_accuracy</th>
      <th>rank_test_accuracy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4</th>
      <td>140.096494</td>
      <td>5.758295</td>
      <td>0.521187</td>
      <td>0.046393</td>
      <td>0.0010</td>
      <td>(600,)</td>
      <td>{'paramClassifier__alpha': 0.001, 'paramClassi...</td>
      <td>-0.866261</td>
      <td>-0.840681</td>
      <td>-0.863283</td>
      <td>...</td>
      <td>0.025946</td>
      <td>1</td>
      <td>0.747801</td>
      <td>0.757017</td>
      <td>0.734395</td>
      <td>0.747276</td>
      <td>0.753562</td>
      <td>0.748010</td>
      <td>0.007719</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>122.917276</td>
      <td>8.892743</td>
      <td>0.471339</td>
      <td>0.012432</td>
      <td>0.0010</td>
      <td>(500,)</td>
      <td>{'paramClassifier__alpha': 0.001, 'paramClassi...</td>
      <td>-0.873963</td>
      <td>-0.843775</td>
      <td>-0.880402</td>
      <td>...</td>
      <td>0.030166</td>
      <td>2</td>
      <td>0.754085</td>
      <td>0.749895</td>
      <td>0.736908</td>
      <td>0.742246</td>
      <td>0.748533</td>
      <td>0.746334</td>
      <td>0.006052</td>
      <td>4</td>
    </tr>
    <tr>
      <th>0</th>
      <td>123.416832</td>
      <td>8.698970</td>
      <td>0.438667</td>
      <td>0.028130</td>
      <td>0.0001</td>
      <td>(500,)</td>
      <td>{'paramClassifier__alpha': 0.0001, 'paramClass...</td>
      <td>-0.873724</td>
      <td>-0.843419</td>
      <td>-0.880807</td>
      <td>...</td>
      <td>0.030496</td>
      <td>3</td>
      <td>0.754085</td>
      <td>0.749057</td>
      <td>0.736908</td>
      <td>0.742666</td>
      <td>0.747276</td>
      <td>0.745998</td>
      <td>0.005833</td>
      <td>6</td>
    </tr>
    <tr>
      <th>1</th>
      <td>138.457693</td>
      <td>6.994077</td>
      <td>0.523172</td>
      <td>0.019858</td>
      <td>0.0001</td>
      <td>(600,)</td>
      <td>{'paramClassifier__alpha': 0.0001, 'paramClass...</td>
      <td>-0.865819</td>
      <td>-0.888762</td>
      <td>-0.862897</td>
      <td>...</td>
      <td>0.024954</td>
      <td>4</td>
      <td>0.748220</td>
      <td>0.752828</td>
      <td>0.735233</td>
      <td>0.746857</td>
      <td>0.751886</td>
      <td>0.747005</td>
      <td>0.006290</td>
      <td>2</td>
    </tr>
    <tr>
      <th>5</th>
      <td>118.817941</td>
      <td>15.252375</td>
      <td>0.304084</td>
      <td>0.079246</td>
      <td>0.0010</td>
      <td>(700,)</td>
      <td>{'paramClassifier__alpha': 0.001, 'paramClassi...</td>
      <td>-0.913266</td>
      <td>-0.832950</td>
      <td>-0.859410</td>
      <td>...</td>
      <td>0.030317</td>
      <td>5</td>
      <td>0.749895</td>
      <td>0.758274</td>
      <td>0.737746</td>
      <td>0.743085</td>
      <td>0.745599</td>
      <td>0.746920</td>
      <td>0.006905</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>168.843865</td>
      <td>16.699910</td>
      <td>0.589206</td>
      <td>0.031498</td>
      <td>0.0001</td>
      <td>(700,)</td>
      <td>{'paramClassifier__alpha': 0.0001, 'paramClass...</td>
      <td>-0.913992</td>
      <td>-0.832747</td>
      <td>-0.892845</td>
      <td>...</td>
      <td>0.030309</td>
      <td>6</td>
      <td>0.750314</td>
      <td>0.758274</td>
      <td>0.732719</td>
      <td>0.743923</td>
      <td>0.745599</td>
      <td>0.746166</td>
      <td>0.008366</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
<p>6 rows × 23 columns</p>
</div>



    
![png](IvesJ_DATA6100_F25_FinalProject_files/IvesJ_DATA6100_F25_FinalProject_68_2.png)
    


#### 2.1.2.7. Extract parameters from parameterization model

The best hyperparameter values can finally be extracted from the cross-validation pipeline, and passed to the final modeling pipeline for use in modeling the full training data set.


```python
##PARAMETER TRANSFER FROM PARAMETERIZATION GRID TO FINAL MODEL
finalHiddenLayers = paramGrid.best_params_['paramClassifier__hidden_layer_sizes']
finalAlpha = paramGrid.best_params_['paramClassifier__alpha']

##CLEAN UP PARAMETERIZATION OBJECTS
del paramGrid
del parameterizationPipeline
del paramModel
del xTrainParam
del yTrainParam
gc.collect()
```




    4021



### 2.1.3. Defining the final model

The final model and pipeline can now be defined, using the optimal parameters established in the previous steps.  This pipeline will be used for both the final cross-validation steps, and full final model fitting.


```python
##INSTANTIATE THE FINAL MODEL
finalModel = MLPClassifier(activation='relu', 
                           max_iter=500, 
                           random_state=1, 
                           early_stopping=True, 
                           n_iter_no_change=10, 
                           hidden_layer_sizes=finalHiddenLayers, 
                           alpha=finalAlpha)

##DEFINE THE FINAL MODEL PIPELINE
finalModelPipeline = Pipeline([
    ('finalClassifier', finalModel)
    ])
```

### 2.1.4. cross-validating the final model

To get a sense of the performance of the final model using the optimized parameters and the full set of training data, one final cross-validation will be run.  This will provide cross entropy loss and accuracy scores for validation data, and is a good way to get a sense of how the model will perform on unseen data.


```python
##cross-valIDATE THE FINAL MODEL TO ASSESS OVERALL PERFORMANCE
finalCvScores = cross_validate(
    estimator = finalModelPipeline, 
    X = xTrainFormatted, 
    y = yTrainEncoded, 
    cv = stratKFold, 
    scoring = ['neg_log_loss', 'accuracy'],
    n_jobs = numCores)
```

#### 2.1.4.1. Review the cross-validation results

The cross-entropy loss and accuracy of the final cross-validation model are checked, to confirm final model performance on unseen data.

---




```python
##DISPLAY cross-validation METRICS
print(f"Validation cross-entropy loss: {np.mean(finalCvScores['test_neg_log_loss'])}")
print(f"Validation accuracy: {np.mean(finalCvScores['test_accuracy'])}")
```

    Validation cross-entropy loss: -0.7680586842725612
    Validation accuracy: 0.7742747858640989
    

### 2.1.5. Train the full-data model

After confirming model performance using cross-validation, the final model can be fit using the full training data set.


```python
##FIT THE FINAL FULL TRAINING DATA MODEL
finalModelPipeline.fit(xTrainFormatted, yTrainEncoded)
```




<style>#sk-container-id-2 {
  /* Definition of color scheme common for light and dark mode */
  --sklearn-color-text: #000;
  --sklearn-color-text-muted: #666;
  --sklearn-color-line: gray;
  /* Definition of color scheme for unfitted estimators */
  --sklearn-color-unfitted-level-0: #fff5e6;
  --sklearn-color-unfitted-level-1: #f6e4d2;
  --sklearn-color-unfitted-level-2: #ffe0b3;
  --sklearn-color-unfitted-level-3: chocolate;
  /* Definition of color scheme for fitted estimators */
  --sklearn-color-fitted-level-0: #f0f8ff;
  --sklearn-color-fitted-level-1: #d4ebff;
  --sklearn-color-fitted-level-2: #b3dbfd;
  --sklearn-color-fitted-level-3: cornflowerblue;

  /* Specific color for light theme */
  --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));
  --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-icon: #696969;

  @media (prefers-color-scheme: dark) {
    /* Redefinition of color scheme for dark theme */
    --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));
    --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-icon: #878787;
  }
}

#sk-container-id-2 {
  color: var(--sklearn-color-text);
}

#sk-container-id-2 pre {
  padding: 0;
}

#sk-container-id-2 input.sk-hidden--visually {
  border: 0;
  clip: rect(1px 1px 1px 1px);
  clip: rect(1px, 1px, 1px, 1px);
  height: 1px;
  margin: -1px;
  overflow: hidden;
  padding: 0;
  position: absolute;
  width: 1px;
}

#sk-container-id-2 div.sk-dashed-wrapped {
  border: 1px dashed var(--sklearn-color-line);
  margin: 0 0.4em 0.5em 0.4em;
  box-sizing: border-box;
  padding-bottom: 0.4em;
  background-color: var(--sklearn-color-background);
}

#sk-container-id-2 div.sk-container {
  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
     but bootstrap.min.css set `[hidden] { display: none !important; }`
     so we also need the `!important` here to be able to override the
     default hidden behavior on the sphinx rendered scikit-learn.org.
     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
  display: inline-block !important;
  position: relative;
}

#sk-container-id-2 div.sk-text-repr-fallback {
  display: none;
}

div.sk-parallel-item,
div.sk-serial,
div.sk-item {
  /* draw centered vertical line to link estimators */
  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));
  background-size: 2px 100%;
  background-repeat: no-repeat;
  background-position: center center;
}

/* Parallel-specific style estimator block */

#sk-container-id-2 div.sk-parallel-item::after {
  content: "";
  width: 100%;
  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
  flex-grow: 1;
}

#sk-container-id-2 div.sk-parallel {
  display: flex;
  align-items: stretch;
  justify-content: center;
  background-color: var(--sklearn-color-background);
  position: relative;
}

#sk-container-id-2 div.sk-parallel-item {
  display: flex;
  flex-direction: column;
}

#sk-container-id-2 div.sk-parallel-item:first-child::after {
  align-self: flex-end;
  width: 50%;
}

#sk-container-id-2 div.sk-parallel-item:last-child::after {
  align-self: flex-start;
  width: 50%;
}

#sk-container-id-2 div.sk-parallel-item:only-child::after {
  width: 0;
}

/* Serial-specific style estimator block */

#sk-container-id-2 div.sk-serial {
  display: flex;
  flex-direction: column;
  align-items: center;
  background-color: var(--sklearn-color-background);
  padding-right: 1em;
  padding-left: 1em;
}


/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is
clickable and can be expanded/collapsed.
- Pipeline and ColumnTransformer use this feature and define the default style
- Estimators will overwrite some part of the style using the `sk-estimator` class
*/

/* Pipeline and ColumnTransformer style (default) */

#sk-container-id-2 div.sk-toggleable {
  /* Default theme specific background. It is overwritten whether we have a
  specific estimator or a Pipeline/ColumnTransformer */
  background-color: var(--sklearn-color-background);
}

/* Toggleable label */
#sk-container-id-2 label.sk-toggleable__label {
  cursor: pointer;
  display: flex;
  width: 100%;
  margin-bottom: 0;
  padding: 0.5em;
  box-sizing: border-box;
  text-align: center;
  align-items: start;
  justify-content: space-between;
  gap: 0.5em;
}

#sk-container-id-2 label.sk-toggleable__label .caption {
  font-size: 0.6rem;
  font-weight: lighter;
  color: var(--sklearn-color-text-muted);
}

#sk-container-id-2 label.sk-toggleable__label-arrow:before {
  /* Arrow on the left of the label */
  content: "▸";
  float: left;
  margin-right: 0.25em;
  color: var(--sklearn-color-icon);
}

#sk-container-id-2 label.sk-toggleable__label-arrow:hover:before {
  color: var(--sklearn-color-text);
}

/* Toggleable content - dropdown */

#sk-container-id-2 div.sk-toggleable__content {
  display: none;
  text-align: left;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-2 div.sk-toggleable__content.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-2 div.sk-toggleable__content pre {
  margin: 0.2em;
  border-radius: 0.25em;
  color: var(--sklearn-color-text);
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-2 div.sk-toggleable__content.fitted pre {
  /* unfitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-2 input.sk-toggleable__control:checked~div.sk-toggleable__content {
  /* Expand drop-down */
  display: block;
  width: 100%;
  overflow: visible;
}

#sk-container-id-2 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
  content: "▾";
}

/* Pipeline/ColumnTransformer-specific style */

#sk-container-id-2 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-2 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator-specific style */

/* Colorize estimator box */
#sk-container-id-2 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-2 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

#sk-container-id-2 div.sk-label label.sk-toggleable__label,
#sk-container-id-2 div.sk-label label {
  /* The background is the default theme color */
  color: var(--sklearn-color-text-on-default-background);
}

/* On hover, darken the color of the background */
#sk-container-id-2 div.sk-label:hover label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

/* Label box, darken color on hover, fitted */
#sk-container-id-2 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator label */

#sk-container-id-2 div.sk-label label {
  font-family: monospace;
  font-weight: bold;
  display: inline-block;
  line-height: 1.2em;
}

#sk-container-id-2 div.sk-label-container {
  text-align: center;
}

/* Estimator-specific */
#sk-container-id-2 div.sk-estimator {
  font-family: monospace;
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: 0.25em;
  box-sizing: border-box;
  margin-bottom: 0.5em;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-2 div.sk-estimator.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

/* on hover */
#sk-container-id-2 div.sk-estimator:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-2 div.sk-estimator.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Specification for estimator info (e.g. "i" and "?") */

/* Common style for "i" and "?" */

.sk-estimator-doc-link,
a:link.sk-estimator-doc-link,
a:visited.sk-estimator-doc-link {
  float: right;
  font-size: smaller;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1em;
  height: 1em;
  width: 1em;
  text-decoration: none !important;
  margin-left: 0.5em;
  text-align: center;
  /* unfitted */
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
  color: var(--sklearn-color-unfitted-level-1);
}

.sk-estimator-doc-link.fitted,
a:link.sk-estimator-doc-link.fitted,
a:visited.sk-estimator-doc-link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
div.sk-estimator:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover,
div.sk-label-container:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover,
div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

/* Span, style for the box shown on hovering the info icon */
.sk-estimator-doc-link span {
  display: none;
  z-index: 9999;
  position: relative;
  font-weight: normal;
  right: .2ex;
  padding: .5ex;
  margin: .5ex;
  width: min-content;
  min-width: 20ex;
  max-width: 50ex;
  color: var(--sklearn-color-text);
  box-shadow: 2pt 2pt 4pt #999;
  /* unfitted */
  background: var(--sklearn-color-unfitted-level-0);
  border: .5pt solid var(--sklearn-color-unfitted-level-3);
}

.sk-estimator-doc-link.fitted span {
  /* fitted */
  background: var(--sklearn-color-fitted-level-0);
  border: var(--sklearn-color-fitted-level-3);
}

.sk-estimator-doc-link:hover span {
  display: block;
}

/* "?"-specific style due to the `<a>` HTML tag */

#sk-container-id-2 a.estimator_doc_link {
  float: right;
  font-size: 1rem;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1rem;
  height: 1rem;
  width: 1rem;
  text-decoration: none;
  /* unfitted */
  color: var(--sklearn-color-unfitted-level-1);
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
}

#sk-container-id-2 a.estimator_doc_link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
#sk-container-id-2 a.estimator_doc_link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

#sk-container-id-2 a.estimator_doc_link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
}

.estimator-table summary {
    padding: .5rem;
    font-family: monospace;
    cursor: pointer;
}

.estimator-table details[open] {
    padding-left: 0.1rem;
    padding-right: 0.1rem;
    padding-bottom: 0.3rem;
}

.estimator-table .parameters-table {
    margin-left: auto !important;
    margin-right: auto !important;
}

.estimator-table .parameters-table tr:nth-child(odd) {
    background-color: #fff;
}

.estimator-table .parameters-table tr:nth-child(even) {
    background-color: #f6f6f6;
}

.estimator-table .parameters-table tr:hover {
    background-color: #e0e0e0;
}

.estimator-table table td {
    border: 1px solid rgba(106, 105, 104, 0.232);
}

.user-set td {
    color:rgb(255, 94, 0);
    text-align: left;
}

.user-set td.value pre {
    color:rgb(255, 94, 0) !important;
    background-color: transparent !important;
}

.default td {
    color: black;
    text-align: left;
}

.user-set td i,
.default td i {
    color: black;
}

.copy-paste-icon {
    background-image: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCA0NDggNTEyIj48IS0tIUZvbnQgQXdlc29tZSBGcmVlIDYuNy4yIGJ5IEBmb250YXdlc29tZSAtIGh0dHBzOi8vZm9udGF3ZXNvbWUuY29tIExpY2Vuc2UgLSBodHRwczovL2ZvbnRhd2Vzb21lLmNvbS9saWNlbnNlL2ZyZWUgQ29weXJpZ2h0IDIwMjUgRm9udGljb25zLCBJbmMuLS0+PHBhdGggZD0iTTIwOCAwTDMzMi4xIDBjMTIuNyAwIDI0LjkgNS4xIDMzLjkgMTQuMWw2Ny45IDY3LjljOSA5IDE0LjEgMjEuMiAxNC4xIDMzLjlMNDQ4IDMzNmMwIDI2LjUtMjEuNSA0OC00OCA0OGwtMTkyIDBjLTI2LjUgMC00OC0yMS41LTQ4LTQ4bDAtMjg4YzAtMjYuNSAyMS41LTQ4IDQ4LTQ4ek00OCAxMjhsODAgMCAwIDY0LTY0IDAgMCAyNTYgMTkyIDAgMC0zMiA2NCAwIDAgNDhjMCAyNi41LTIxLjUgNDgtNDggNDhMNDggNTEyYy0yNi41IDAtNDgtMjEuNS00OC00OEwwIDE3NmMwLTI2LjUgMjEuNS00OCA0OC00OHoiLz48L3N2Zz4=);
    background-repeat: no-repeat;
    background-size: 14px 14px;
    background-position: 0;
    display: inline-block;
    width: 14px;
    height: 14px;
    cursor: pointer;
}
</style><body><div id="sk-container-id-2" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>Pipeline(steps=[(&#x27;finalClassifier&#x27;,
                 MLPClassifier(alpha=0.001, early_stopping=True,
                               hidden_layer_sizes=(600,), max_iter=500,
                               random_state=1))])</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-4" type="checkbox" ><label for="sk-estimator-id-4" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>Pipeline</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.7/modules/generated/sklearn.pipeline.Pipeline.html">?<span>Documentation for Pipeline</span></a><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></div></label><div class="sk-toggleable__content fitted" data-param-prefix="">
        <div class="estimator-table">
            <details>
                <summary>Parameters</summary>
                <table class="parameters-table">
                  <tbody>

        <tr class="user-set">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('steps',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">steps&nbsp;</td>
            <td class="value">[(&#x27;finalClassifier&#x27;, ...)]</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('transform_input',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">transform_input&nbsp;</td>
            <td class="value">None</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('memory',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">memory&nbsp;</td>
            <td class="value">None</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('verbose',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">verbose&nbsp;</td>
            <td class="value">False</td>
        </tr>

                  </tbody>
                </table>
            </details>
        </div>
    </div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-5" type="checkbox" ><label for="sk-estimator-id-5" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>MLPClassifier</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.7/modules/generated/sklearn.neural_network.MLPClassifier.html">?<span>Documentation for MLPClassifier</span></a></div></label><div class="sk-toggleable__content fitted" data-param-prefix="finalClassifier__">
        <div class="estimator-table">
            <details>
                <summary>Parameters</summary>
                <table class="parameters-table">
                  <tbody>

        <tr class="user-set">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('hidden_layer_sizes',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">hidden_layer_sizes&nbsp;</td>
            <td class="value">(600,)</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('activation',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">activation&nbsp;</td>
            <td class="value">&#x27;relu&#x27;</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('solver',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">solver&nbsp;</td>
            <td class="value">&#x27;adam&#x27;</td>
        </tr>


        <tr class="user-set">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('alpha',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">alpha&nbsp;</td>
            <td class="value">0.001</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('batch_size',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">batch_size&nbsp;</td>
            <td class="value">&#x27;auto&#x27;</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('learning_rate',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">learning_rate&nbsp;</td>
            <td class="value">&#x27;constant&#x27;</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('learning_rate_init',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">learning_rate_init&nbsp;</td>
            <td class="value">0.001</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('power_t',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">power_t&nbsp;</td>
            <td class="value">0.5</td>
        </tr>


        <tr class="user-set">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('max_iter',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">max_iter&nbsp;</td>
            <td class="value">500</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('shuffle',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">shuffle&nbsp;</td>
            <td class="value">True</td>
        </tr>


        <tr class="user-set">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('random_state',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">random_state&nbsp;</td>
            <td class="value">1</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('tol',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">tol&nbsp;</td>
            <td class="value">0.0001</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('verbose',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">verbose&nbsp;</td>
            <td class="value">False</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('warm_start',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">warm_start&nbsp;</td>
            <td class="value">False</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('momentum',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">momentum&nbsp;</td>
            <td class="value">0.9</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('nesterovs_momentum',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">nesterovs_momentum&nbsp;</td>
            <td class="value">True</td>
        </tr>


        <tr class="user-set">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('early_stopping',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">early_stopping&nbsp;</td>
            <td class="value">True</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('validation_fraction',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">validation_fraction&nbsp;</td>
            <td class="value">0.1</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('beta_1',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">beta_1&nbsp;</td>
            <td class="value">0.9</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('beta_2',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">beta_2&nbsp;</td>
            <td class="value">0.999</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('epsilon',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">epsilon&nbsp;</td>
            <td class="value">1e-08</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('n_iter_no_change',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">n_iter_no_change&nbsp;</td>
            <td class="value">10</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('max_fun',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">max_fun&nbsp;</td>
            <td class="value">15000</td>
        </tr>

                  </tbody>
                </table>
            </details>
        </div>
    </div></div></div></div></div></div></div><script>function copyToClipboard(text, element) {
    // Get the parameter prefix from the closest toggleable content
    const toggleableContent = element.closest('.sk-toggleable__content');
    const paramPrefix = toggleableContent ? toggleableContent.dataset.paramPrefix : '';
    const fullParamName = paramPrefix ? `${paramPrefix}${text}` : text;

    const originalStyle = element.style;
    const computedStyle = window.getComputedStyle(element);
    const originalWidth = computedStyle.width;
    const originalHTML = element.innerHTML.replace('Copied!', '');

    navigator.clipboard.writeText(fullParamName)
        .then(() => {
            element.style.width = originalWidth;
            element.style.color = 'green';
            element.innerHTML = "Copied!";

            setTimeout(() => {
                element.innerHTML = originalHTML;
                element.style = originalStyle;
            }, 2000);
        })
        .catch(err => {
            console.error('Failed to copy:', err);
            element.style.color = 'red';
            element.innerHTML = "Failed!";
            setTimeout(() => {
                element.innerHTML = originalHTML;
                element.style = originalStyle;
            }, 2000);
        });
    return false;
}

document.querySelectorAll('.fa-regular.fa-copy').forEach(function(element) {
    const toggleableContent = element.closest('.sk-toggleable__content');
    const paramPrefix = toggleableContent ? toggleableContent.dataset.paramPrefix : '';
    const paramName = element.parentElement.nextElementSibling.textContent.trim();
    const fullParamName = paramPrefix ? `${paramPrefix}${paramName}` : paramName;

    element.setAttribute('title', fullParamName);
});
</script></body>



# 3. Final test predictions and wrap-up

With model validation and fitting complete, the training data can now be processed by the model and predictions can be generated.

## 3.1. Extract feature names for test data configuration

To ensure a uniform structure between the training and test data sets, the column names from the training data must be transferred to the test data.


```python
trainingColumns = xTrainFormatted.columns.tolist()

trainingColumns = [item.replace('ingredients_', '') for item in trainingColumns if 'ingredients_' in item]
```

## 3.2. Final predictions

The test data must be pre-processed using the cleaning and encoding pipeline.  Once that is done the final model pipeline can be used to generate predictions based on the model.  Those predictions will be label encoded, so they will need to be inverse transformed to represent the cuisine categories.

Once the formatting is done the final test predictions can be output to file for submission to Kaggle.

---

*Final kaggle submission score: **.77353***

*The leaderboard is locked, but if it were active and in the same state it is in now, leaderboard position would be **778**.*


```python
###FORMAT TEST DATA (OTHERIZE TO MATCH OHE COLUMNS, THEN RUN PIPELINE)
xTestFormatted = DfFromPipeline(OtherizeUnseenIngredients(xTest, cleaningSteps, 'id', 'ingredients', trainingColumns), pipelineSteps, idToIndex = True)

##RUN AND DECODE TEST PREDICTIONS
testPredsEncoded = finalModelPipeline.predict(xTestFormatted)
xTestFormatted['cuisine'] = yEncoder.inverse_transform(testPredsEncoded)

#REMOVE DUMMY RECIPE PREDICTION
xTestFormatted = xTestFormatted.reset_index(drop = False)
xTestFormatted = xTestFormatted.loc[xTestFormatted['id'] != -1]

##OUTPUT TEST PREDICTIONS TO FILE
WriteTestData(xTestFormatted[['id', 'cuisine']], revision = "Rev1", byDate = True, type = "6100_kaggle_csv")
```

