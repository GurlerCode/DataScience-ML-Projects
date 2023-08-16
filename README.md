# DataScience-ML-Projects
DataScience Machine Learning Algorithm Projects

## Laptop Price Prediction Project
This project aims to predict laptop prices based on various features using machine learning techniques. The dataset used for this project is sourced from [Kaggle](https://www.kaggle.com/datasets/muhammetvarl/laptop-price), containing information about different laptop models and their specifications.

## Introduction
In this project, we aim to build a machine learning model that predicts the prices of laptops based on their specifications. The project involves data preprocessing, exploratory data analysis, model training, and evaluation.

## Dataset
The dataset used in this project can be found on Kaggle. It includes various features such as laptop specifications, including the company, product type, screen size, CPU, RAM, GPU, and more.

## Installation
To run this project, you need Python and the following libraries:
pandas
matplotlib
seaborn
scikit-learn

## Usage
1. Clone this repository to your local machine.
2. Download the dataset from the provided Kaggle link and place it in the project directory.
3. Open a Jupyter Notebook or your preferred Python environment to run the project.

## Data Preprocessing
1. Load the dataset using pandas and understand its structure.
2. Handle missing data by either removing or imputing missing values.
3. Convert data types as needed for further analysis.

## Data Visualization
Create visualizations using seaborn and matplotlib to understand the distribution of data and relationships between variables.
Visualize features such as the operating system, company, laptop type, and more.

## Model Training
1. Split the data into training and testing sets using train_test_split.
2. Create preprocessing pipelines for numerical and categorical data using ColumnTransformer.
3. Train a Random Forest Regressor model on the preprocessed data.
   
## Evaluation
1. Evaluate the model's performance using metrics such as Mean Absolute Error (MAE).
2. Use cross-validation to get an estimate of the model's generalization performance.
3. Tune hyperparameters, such as the number of estimators, to improve the model's performance.

## Future Work
Explore more advanced preprocessing techniques, such as feature scaling and dimensionality reduction.
Experiment with different machine learning algorithms and compare their performance.
Fine-tune hyperparameters further to optimize the model's accuracy.
Deploy the trained model to make price predictions for new laptops.

## Contributing
Contributions are welcome! If you find any issues or have suggestions for improvement, feel free to open an issue or submit a pull request.

## Acknowledgement
This project was done as a Datascience bootcamp final exam.
