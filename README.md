
Titanic Survival Prediction

Overview

This project uses machine learning algorithms to predict the survival of passengers on the Titanic based on their demographic and travel information.

Dataset

The dataset used for this project is the Titanic Dataset, which contains information on 891 passengers, including:

- Demographic information: age, sex, class, etc.
- Travel information: embarkation point, destination, etc.
- Survival information: survived or died

Project Goals

1. Exploratory Data Analysis: Explore the Titanic Dataset to understand the distribution of the variables and the relationships between them.
2. Feature Engineering: Select the most relevant features for predicting survival.
3. Model Selection: Train and evaluate several machine learning models to predict survival.
4. Hyperparameter Tuning: Optimize the hyperparameters of the selected model to improve its performance.

Methodology

1. Data Preprocessing: Clean and preprocess the dataset by handling missing values and scaling the features.
2. Feature Selection: Select the most relevant features using techniques such as correlation analysis and recursive feature elimination.
3. Model Training: Train several machine learning models, including logistic regression, decision trees, random forests, and support vector machines.
4. Model Evaluation: Evaluate the performance of each model using metrics such as accuracy, precision, recall, and F1-score.
5. Hyperparameter Tuning: Optimize the hyperparameters of the selected model using techniques such as grid search and cross-validation.

Results

The best-performing model was a random forest classifier with an accuracy of 83.21%. The model was trained on the scaled features and optimized using grid search.

Conclusion

This project demonstrated the use of machine learning algorithms for predicting the survival of passengers on the Titanic. The results showed that a random forest classifier with optimized hyperparameters achieved the best performance.

Code

The code for this project is written in Python and uses the following libraries:

- Pandas for data manipulation and analysis
- NumPy for numerical computations
- Scikit-learn for machine learning algorithms
- Matplotlib and Seaborn for data visualization

Future Work

1. Collect more data: Collect more information on the passengers, such as their cabin numbers or travel companions.
2. Use deep learning algorithms: Use deep learning algorithms such as neural networks to improve the model's performance.
3. Use transfer learning: Use transfer learning to leverage pre-trained models and improve the model's performance.
