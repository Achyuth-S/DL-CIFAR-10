import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import numpy as np
import matplotlib.pyplot as plt
import os

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()

# Normalize pixel values
x_train, x_test = x_train / 255.0, x_test / 255.0

# Save CIFAR-10 numpy arrays to disk
os.makedirs('saved_data', exist_ok=True)
np.save('saved_data/x_train.npy', x_train)
np.save('saved_data/y_train.npy', y_train)
np.save('saved_data/x_test.npy', x_test)
np.save('saved_data/y_test.npy', y_test)
print("Saved CIFAR-10 dataset arrays to 'saved_data/' directory")

# One-hot encode labels
num_classes = 10
y_train_ohe = tf.keras.utils.to_categorical(y_train, num_classes)
y_test_ohe = tf.keras.utils.to_categorical(y_test, num_classes)

# Define CNN model architecture
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(32, 32, 3)),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3,3), activation='relu', padding='same'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3,3), activation='relu', padding='same'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(x_train, y_train_ohe, epochs=15, batch_size=64, validation_split=0.2, verbose=2)

# Save model to disk
os.makedirs('saved_model', exist_ok=True)
model.save('saved_model/cifar10_cnn_model.h5')
print("Saved trained model to 'saved_model/cifar10_cnn_model.h5'")

# Evaluate and print test accuracy
test_loss, test_acc = model.evaluate(x_test, y_test_ohe, verbose=0)
print(f"Test accuracy: {test_acc:.3f}")

# Plot training and validation accuracy and save the figure
plt.plot(history.history['accuracy'], label='train accuracy')
plt.plot(history.history['val_accuracy'], label='validation accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Model Accuracy')
plt.savefig('saved_model/training_validation_accuracy.png')
plt.show()
print("Saved accuracy plot as 'saved_model/training_validation_accuracy.png'")
