# California Housing Price Prediction

An end-to-end Machine Learning project to predict median housing prices in California districts using the 1990 California census data.

## Project Overview
This project follows a systematic 8-step machine learning workflow:
1. **Frame the Problem and Look at the Big Picture**: Formulate as a supervised multivariate regression task using Root Mean Squared Error (RMSE) performance metric.
2. **Get the Data**: Automated download, extraction, and loading of the California housing dataset.
3. **Explore and Visualize the Data**: Analyze numerical distributions, geographical clustering, and Pearson correlation coefficients.
4. **Prepare Data for Machine Learning Algorithms**: Handle missing values with imputation, encode categorical features, build custom feature engineering transformers, and scale numerical variables.
5. **Select and Train Models**: Train Linear Regression, Decision Tree, and Random Forest baseline models.
6. **Fine-Tune the Models**: Optimize hyperparameters using Scikit-Learn's `GridSearchCV` and evaluate via 10-fold cross-validation.
7. **Evaluate on the Test Set**: Verify final model performance on unseen stratified test data and calculate 95% confidence intervals.
8. **Document Findings & Insights**: Present feature importances and domain takeaways.

## Directory Structure
- `prediction-for-house-value-in-californian-district.ipynb`: Main Jupyter notebook containing data exploration, preprocessing pipelines, model training, cross-validation, hyperparameter tuning, and test set evaluations.
- `requirements.txt`: Python package dependencies.
- `datasets/`: Dataset cache folder.
- `images/`: Exported figures and geographical visualizations.

## Requirements & Setup
To run the notebook locally, install dependencies using Python 3.11+:
```bash
pip install -r requirements.txt
```
