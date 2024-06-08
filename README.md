# Machine Learning Project

## Description

This project contains the implementation of various machine learning models for analyzing data from ground vibration tests of the F-16 aircraft. The project consists of several Python scripts that enable data preparation, model training, and evaluation.

## Files

1. **`samples.py`**: 
   - Contains classes and functions for processing input data.
   - Defines the `DataSample` class, which represents a single data sample.
   - Includes functions for loading data from CSV files and converting them into `DataSample` objects.

2. **`training_validation_sets.py`**:
   - Contains functions for splitting data into training, validation, and testing sets.
   - Uses `train_test_split` to divide the data and scalers to normalize the data.

3. **`model_1.py`**:
   - Implementation of a basic neural network model with two hidden layers.
   - Uses the `Adam` optimizer and the `mean_squared_error` loss function.
   - Contains functions for training the model, visualizing results, and evaluating the model.

4. **`model_2.py`**:
   - An extended version of the model with additional Dropout layers to prevent overfitting.
   - Allows selection of various optimizers such as `Adam`, `Nadam`, `SGD`, `RMSprop`, and `Adagrad`.
   - Contains functions for training the model, visualizing results, and evaluating the model.

5. **`model_3_loss_function.py`**:
   - An extended version of the model with the option to use a custom loss function (`custom_loss`).
   - Includes an Early Stopping mechanism that automatically stops training when the model stops improving.
   - Contains functions for training the model, visualizing results, and evaluating the model.

## Instructions

1. **Data Preparation**:
   - Use the `samples.py` script to load and process the input data.
   - The input data should be in CSV format and placed in the `BenchmarkData` folder.

2. **Data Splitting**:
   - Use functions from `training_validation_sets.py` to split the data into training, validation, and testing sets.

3. **Model Training**:
   - Choose one of the models (`model_1.py`, `model_2.py`, `model_3_loss_function.py`) and run the script.
   - The script will automatically load the data, perform training, and save the results.

4. **Model Evaluation**:
   - After training the model, the script will automatically evaluate it on the test set and save the results in appropriate files.

## Requirements

- Python 3.6 or later
- TensorFlow
- scikit-learn
- matplotlib
- numpy
- pickle

## Running the Scripts

Example of running the `model_1.py` script:

```bash
python model_1.py
