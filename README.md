# Fault_Findy
Objective: Build an intelligent system to predict faults in manufacturing processes using deep learning.

Data Preparation: Set up ImageDataGenerator with rescaling, random transformations, and a validation split for image augmentation.

Data Loading: Load images of defective and good tires from a directory, using separate generators for training and validation datasets.

Model Selection: Use MobileNetV2 as a base model with pretrained ImageNet weights, excluding the top layers for transfer learning.

Model Freezing: Initially, freeze all layers in the base model to retain learned features.

Model Architecture: Construct a Sequential model with MobileNetV2, GlobalAveragePooling2D, Dropout, and a Dense layer with sigmoid activation for binary classification.

Compilation: Compile the model using the Adam optimizer, binary cross-entropy loss, and accuracy as the metric.

Early Stopping: Integrate early stopping to avoid overfitting by monitoring validation loss.

Initial Training: Train the model on the training data while validating with the validation set for 10 epochs.

Fine-Tuning: Unfreeze the last 20 layers of MobileNetV2 to allow for fine-tuning and compile the model with a reduced learning rate.

Fine-Tuning Training: Train the model further, starting from the previous checkpoint, for an additional 10 epochs.

Model Saving: Save the trained model for later use in classification tasks.

Evaluation: Evaluate the model on the validation set and print the final accuracy after fine-tuning.
