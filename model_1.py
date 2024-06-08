from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import pickle
import numpy as np
import os
from training_validation_sets import get_training_validation_and_testing_sets

def build_model(input_shape):
    model = Sequential([
        Flatten(input_shape=input_shape),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(3)  # Output layer for regression task
    ])
    return model

def train_model(X_train, y_train, X_val, y_val):
    model = build_model(input_shape=X_train[0].shape)
    model.compile(optimizer='adam',
                  loss='mean_squared_error',
                  metrics=['mae'])  # Mean Absolute Error

    # Add EarlyStopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    history = model.fit(X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=200, batch_size=32,
                        callbacks=[early_stopping])
    
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('training_validation_loss.png')
    plt.close()

    # Return the trained model and its history
    return model, history

def plot_data_samples(X_train, y_train, X_val, y_val, X_test, y_test):
    def plot_samples(X, y, title, filename):
        fig, ax1 = plt.subplots()

        color = "tab:blue"
        ax1.set_xlabel("Index")
        ax1.set_ylabel("Force", color=color)
        ax1.plot(range(len(X)), [x[0][0] for x in X], color=color, label="Force")
        ax1.tick_params(axis="y", labelcolor=color)

        ax2 = ax1.twinx()
        color = "tab:red"
        ax2.set_ylabel("Voltage", color=color)
        ax2.plot(range(len(X)), [x[0][1] for x in X], color=color, label="Voltage")
        ax2.tick_params(axis="y", labelcolor=color)

        fig.legend(
            loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes
        )
        plt.title(title)
        fig.tight_layout()
        plt.savefig(filename)
        plt.close()

    plot_samples(X_train, y_train, "Training Data", "training_data.png")
    plot_samples(X_val, y_val, "Validation Data", "validation_data.png")
    plot_samples(X_test, y_test, "Testing Data", "testing_data.png")

def evaluate_model(model, X_test, y_test):
    test_loss, test_mae = model.evaluate(X_test, y_test)
    print(f'Test Loss: {test_loss}')
    print(f'Test MAE: {test_mae}')

if __name__ == "__main__":
    # Load data samples
    with open("data_samples.pkl", "rb") as file:
        data_samples = pickle.load(file)

    # Load training, validation, and test sets
    with open("sets.pkl", "rb") as file:
        sets = pickle.load(file)

    with open("scaled_sets.pkl", "rb") as file:
        sets_scaled = pickle.load(file)

    X_train, y_train, X_val, y_val, X_test, y_test = sets
    
    # Convert data to float32
    X_train = np.array(X_train).astype('float32')
    y_train = np.array(y_train).astype('float32')
    X_val = np.array(X_val).astype('float32')
    y_val = np.array(y_val).astype('float32')
    X_test = np.array(X_test).astype('float32')
    y_test = np.array(y_test).astype('float32')
    
    print(X_train.shape, y_train.shape)

    # Plot data samples
    plot_data_samples(X_train, y_train, X_val, y_val, X_test, y_test)

    # Train the model
    trained_model, training_history = train_model(X_train, y_train, X_val, y_val)

    # Save the trained model
    trained_model.save("trained_model.h5")

    # Evaluate the model on the test set
    evaluate_model(trained_model, X_test, y_test)
