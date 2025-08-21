## Model Details
This model was developed by Candy Pipes using Udacity's Machine Learning DevOps course.  The model date is 8/17/2025 and the version is v1.
This model is a RandomForestClassifier.  
The included libraries are Python 3.10, sklearn, os, pickle, numpy, and fastapi.
The model hyperparameter of n_estimators is set at the default of 100, the random state was set as 42,  and n_jobs set at -1 for faster performance and reproduceability.
The data source that was used is Census Income (census.csv) which can be found at https://archive.ics.uci.edu/dataset/20/census+income

## Intended Use
This model was designed to predict if an individual earns more than $50k each year based on the census data provided. The intended users of this model include Udacity course graders.  The primary intended use is for educational purposes. Out of scope uses include anything outside of educational uses.

## Training Data
The data set that was used was Census Income provided by UC Irvine Macine Learnin Repository.  This dataset is used to predict if income exceeds $50k a year based on census data.  The dataset is dated April 30, 1996, which is now outdated.  The feature types are Categorical and Integer. It contains the 15 features (age, workclass, fnlwgt, education, education-num, marital-status, occupation, relationship, race, sex, capital-gain, capital-loss, hours-per-week, native-country, and income) and 48,842 rows.  Due to the changes in demographics of race, sex, and marital status, this information may be considered biased or out of date. Training data is set to 80% of the dataset.  The target label is "salary", which is binarized into greater than or equal to 50K and less than 50K.

## Evaluation Data
The data set used was Census Income which matches the training data set.  The same preprocessing streps were used.  Evaluation data is set to 20% of the dataset.  

## Metrics
The model was evaluated using precision, recall, and F1 score.  The results were Precision: 0.7419 | Recall: 0.6384 | F1: 0.6863.  This model predicts if an individual earns at least 50K, 74% of the time.  The recall score of 63% suggests that 37% of true positives were missed.  The F1 score of 69% indicates that the model performs well, but there is room for further improvement.

## Ethical Considerations
This dataset is dated 1996 which reflects different demographics than are expressed now.  This means the dataset could be based on biased categories.  This model should not be used to make financial decisions based on outdated, potentially biased results.  The metrics show that further improvement is needed on this model before it can be used to more accurately predict an individuals earning level.

## Caveats and Recommendations
Features such as workclass, education, marital-status, relationship, race, and sex could be expanded to meet more up-to-date categories.
This model is best served for educational purposes and not as a tool used for real-life applications.