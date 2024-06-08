from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam, Nadam, SGD, RMSprop, Adagrad
import matplotlib.pyplot as plt
import pickle
import numpy as np
import os

from training_validation_sets import get_training_validation_and_testing_sets

def build_model(input_shape):
    model = Sequential([
        Flatten(input_shape=input_shape),
        Dense(1024, activation='relu'), 
        Dropout(0.2), 
        Dense(512, activation='relu'),  
        Dropout(0.2),  
        Dense(256, activation='relu'), 
        Dropout(0.2),  
        Dense(128, activation='relu'),  
        Dropout(0.2),  
        Dense(3) 
    ])
    return model

def get_optimizer(optimizer_name):
    if optimizer_name == 'adam':
        return Adam(learning_rate=0.001)
    elif optimizer_name == 'nadam':
        return Nadam(learning_rate=0.001)
    elif optimizer_name == 'sgd':
        return SGD(learning_rate=0.01)
    elif optimizer_name == 'rmsprop':
        return RMSprop(learning_rate=0.001)
    elif optimizer_name == 'adagrad':
        return Adagrad(learning_rate=0.01)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

def train_model(X_train, y_train, X_val, y_val, optimizer_name='adam'):
    model = build_model(input_shape=X_train[0].shape)
    optimizer = get_optimizer(optimizer_name)
    model.compile(optimizer=optimizer,
                  loss='mean_squared_error',
                  metrics=['mae'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True)

    history = model.fit(X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=1000, batch_size=32,
                        callbacks=[early_stopping])

    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(history.history['loss'], label=f'Training MSE Loss ({optimizer_name})')
    plt.plot(history.history['val_loss'], label=f'Validation MSE Loss ({optimizer_name})')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(history.history['mae'], label=f'Training MAE ({optimizer_name})')
    plt.plot(history.history['val_mae'], label=f'Validation MAE ({optimizer_name})')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'{optimizer_name}/training_validation_metrics_{optimizer_name}.png')
    plt.close()

    with open(f'{optimizer_name}/metrics_{optimizer_name}.txt', 'w') as f:
        f.write(f"Training MSE Loss: {history.history['loss'][-1]}\n")
        f.write(f"Validation MSE Loss: {history.history['val_loss'][-1]}\n")
        f.write(f"Training MAE: {history.history['mae'][-1]}\n")
        f.write(f"Validation MAE: {history.history['val_mae'][-1]}\n")

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

def evaluate_model(model, X_test, y_test, optimizer_name):
    test_loss, test_mae = model.evaluate(X_test, y_test)
    print(f'Test MSE Loss: {test_loss}')
    print(f'Test MAE: {test_mae}')

    with open(f'{optimizer_name}/test_metrics_{optimizer_name}.txt', 'w') as f:
        f.write(f"Test MSE Loss: {test_loss}\n")
        f.write(f"Test MAE: {test_mae}\n")

    y_pred = model.predict(X_test)

    plt.figure(figsize=(14, 6))
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.plot(y_test[:, i], label='Actual')
        plt.plot(y_pred[:, i], label='Predicted')
        plt.xlabel('Sample Index')
        plt.ylabel(f'Output {i + 1}')
        plt.title(f'Actual vs Predicted for Output {i + 1} ({optimizer_name})')
        plt.legend()

    plt.tight_layout()
    plt.savefig(f'{optimizer_name}/actual_vs_predicted_{optimizer_name}.png')
    plt.close()

if __name__ == "__main__":
    with open("data_samples.pkl", "rb") as file:
        data_samples = pickle.load(file)

    with open("sets.pkl", "rb") as file:
        sets = pickle.load(file)

    with open("scaled_sets.pkl", "rb") as file:
        sets_scaled = pickle.load(file)

    X_train, y_train, X_val, y_val, X_test, y_test = sets

    X_train = np.array(X_train).astype('float32')
    y_train = np.array(y_train).astype('float32')
    X_val = np.array(X_val).astype('float32')
    y_val = np.array(y_val).astype('float32')
    X_test = np.array(X_test).astype('float32')
    y_test = np.array(y_test).astype('float32')

    print(X_train.shape, y_train.shape)

    plot_data_samples(X_train, y_train, X_val, y_val, X_test, y_test)

    optimizers = ['adam', 'nadam', 'rmsprop', 'adagrad'] 

    for optimizer_name in optimizers:
        print(f"Training with optimizer: {optimizer_name}")
        
        if not os.path.exists(optimizer_name):
            os.makedirs(optimizer_name)

        trained_model, training_history = train_model(X_train, y_train, X_val, y_val, optimizer_name=optimizer_name)

        trained_model.save(f"{optimizer_name}/trained_model_{optimizer_name}.h5")

        evaluate_model(trained_model, X_test, y_test, optimizer_name=optimizer_name)
