# A-convolutional-neural-network-CNN-Project

In this assignment I builds CNN models for image classification tasks.

Set up:

1. Doanloaded the Monkeys Species image dataset. Link; https://www.kaggle.com/datasets/utkarshsaxenadn/10-species-of-monkey-multiclass-classification
2. Obtained a pre-trained model of my choice (EfficientNetV2S).

Assignment Tasks:

Task 1: Your own CNN architectures:

Try two different CNN architectures. You can change the layers, change the number or size of filters, with and without dropout, etc. to get different architectures.
Train each model till the training accuracy does not seem to improve over epochs. Test each model on the test data.
Compare the accuracy of the two models on the test data. Obtain confusion matrix for the model with better accuracy on the test data.
Save the model which gives better accuracy on test data.

Task 2: Fine-tuning a pre-trained CNN architecture:

Fine-tune a pre-trained architecture of your choice. The layer(s) you add on the top is your design decision.
Train the model till the training accuracy does not seem to improve over epochs. Test it on the test data.
Compare the accuracy of the fine-tuned model with the better model from Task 1 on the test data. Obtain the confusion matrix for the fine-tuned model on the test data.

Task 3: Error Analysis:

Take any 10 test images on which the better model from Task 1 makes incorrect predictions.
Qualitatively see if there may be reason(s) why the model makes incorrect predictions.
See if the fine-tuned model of Task 2 improves on those 10 images or not.
