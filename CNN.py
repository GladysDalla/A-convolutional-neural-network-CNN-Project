# Import necessary libraries
from tensorflow.keras.layers import GlobalAveragePooling2D, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input, Rescaling
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import image_dataset_from_directory
from keras.models import load_model, Model
from sklearn.metrics import confusion_matrix
from tensorflow.keras.applications import ResNet50V2, EfficientNetV2S
import numpy as np

# Load the datasets
training_set = image_dataset_from_directory("Monkey Species Data/Training Data", label_mode="categorical", image_size=(100,100))
test_set = image_dataset_from_directory("Monkey Species Data/Prediction Data", label_mode="categorical", image_size=(100,100), shuffle=False)

# Task 1

# Define the first CNN architecture
model1 = Sequential([
    Input((100,100,3)),
    Rescaling(1/255),
    Conv2D(32, kernel_size=(3, 3), activation="relu"),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.2),
    Dense(10, activation="softmax")
])

# Compile the model
model1.compile(loss="categorical_crossentropy", metrics=["accuracy"])
model1.summary()

# Train the first model
epochs = 10
print("Training Model 1.")
for i in range(epochs):
    history = model1.fit(training_set, epochs=1)
    print("Epoch:", i+1, "Training Accuracy:", history.history["accuracy"])

# Evaluate the first model on test data
print("Evaluating Model 1 on test data.")
score1 = model1.evaluate(test_set)
print("Test accuracy of Model 1:", score1[1])

# Save the first model
model1.save("my_cnn1.keras")

# Define the second CNN architecture
model2 = Sequential([
    Input((100,100,3)),
    Rescaling(1/255),
    Conv2D(64, kernel_size=(3, 3), activation="relu"),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation="relu"),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(256, activation="relu"),
    Dropout(0.5),
    Dense(10, activation="softmax")
])

# Compile the model
model2.compile(loss="categorical_crossentropy", metrics=["accuracy"])
model2.summary()

# Train the second model
epochs = 10
print("Training Model 2.")
for i in range(epochs):
    history = model2.fit(training_set, epochs=1)
    print("Epoch:", i+1, "Training Accuracy:", history.history["accuracy"])

# Evaluate the second model on test data
print("Evaluating Model 2 on test data.")
score2 = model2.evaluate(test_set)
print("Test accuracy of Model 2:", score2[1])

# Save the second model 
model2.save("my_cnn2.keras")

# Choose the model with the better accuracy on test data
from keras.models import load_model
old_model = load_model("my_cnn1.keras")
old_model2 = load_model("my_cnn2.keras")

if score1[1] > score2[1]:
    better_model = old_model
else:
    better_model = old_model2

# Save the better model
better_model.save("better_model.keras")
print("Better model saved. \n")

# Get true labels and predicted classes
true_labels = []
predicted_classes = []
for images, labels in test_set:
    true_labels.extend(np.argmax(labels, axis=1))
    predictions = np.argmax(better_model.predict(images), axis=1)
    predicted_classes.extend(predictions)

# Create confusion matrix
cm = confusion_matrix(true_labels, predicted_classes)
print("Confusion Matrix:")
print(cm)

# Task 2
# Load pre-trained ResNet50V2 model without top classification layers
base_model = EfficientNetV2S(include_top=False)

# Add custom classification layers on top of the base model
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
output_layer = Dense(10, activation='softmax')(x)  

# Create the final model
model3 = Model(inputs=base_model.input, outputs=output_layer)


# Freeze the layers of the base model
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model3.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#model3.summary()

# Train the second model
epochs = 10
print("Training Model 3.")
for i in range(epochs):
    history = model3.fit(training_set, epochs=1)
    print("Epoch:", i+1, "Training Accuracy:", history.history["accuracy"])

# Evaluate the second model on test data
print("Evaluating Model 2 on test data.")
score3 = model3.evaluate(test_set)
print("Test accuracy of Model 3:", score3[1])

# Save the fine-tuned model
model3.save("fine_tuned.keras")

# Get true labels and predicted classes
true_labels3 = []
predicted_classes3 = []
for images, labels in test_set:
    true_labels3.extend(np.argmax(labels, axis=1))
    predictions3 = np.argmax(model3.predict(images), axis=1)
    predicted_classes3.extend(predictions3)

# Create confusion matrix
cm = confusion_matrix(true_labels3, predicted_classes3)
print("Confusion Matrix:")
print(cm)

# Task 3

# Initialize lists to store incorrect predictions and corresponding images
# better model
old_model3 = load_model("better_model.keras")

#old_model = load_model(model_path)

incorrect_predictions_task1 = []
incorrect_images_task1 = []

# Iterate through the test set
for images, labels in test_set:
    predictions = old_model3.predict(images)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(labels, axis=1)

    # Find incorrect predictions
    for i in range(len(images)):
        if predicted_classes[i] != true_classes[i]:
            incorrect_predictions_task1.append((predicted_classes[i], true_classes[i]))
            incorrect_images_task1.append(images[i])

    # Stop when we have collected 10 incorrect predictions
    if len(incorrect_predictions_task1) >= 10:
        break

# Display the incorrect predictions and corresponding images from Task 1
print("Incorrect Predictions and Corresponding Images from Task 1:")
for i in range(10):
    print("Incorrectly predicted class:", incorrect_predictions_task1[i][0])
    print("True class:", incorrect_predictions_task1[i][1])
    # Normalize the image data
    normalized_image = incorrect_images_task1[i] / 255.0
    plt.imshow(normalized_image)
    plt.axis('off')
    plt.show()

# Initialize lists to store predictions of Task 2 model
incorrect_predictions_task2 = []
incorrect_images_task2 = []

# Load the fine-tuned model from Task 2
#model_path_task2 = '/content/drive/MyDrive/model/fine_tuned.keras'
fine_tuned_model = load_model('fine_tuned.keras')

# Iterate through the same set of images
for image in incorrect_images_task1:
    # Make predictions using the fine-tuned model
    prediction_task2 = fine_tuned_model.predict(np.expand_dims(image, axis=0))
    predicted_class_task2 = np.argmax(prediction_task2)
    
    # Store predictions
    incorrect_predictions_task2.append(predicted_class_task2)
    incorrect_images_task2.append(image)

# Display the incorrect predictions and corresponding images from Task 2
print("Predictions from Fine-tuned Model (Task 2):")
for i in range(10):
    print("Predicted class (Task 2):", incorrect_predictions_task2[i])
    
    # Normalize the image data
    normalized_image = incorrect_images_task2[i] / 255.0
    plt.imshow(normalized_image)
    plt.axis('off')
    plt.show()

