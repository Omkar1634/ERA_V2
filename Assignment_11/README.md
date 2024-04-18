# Class Activation Maps, LRs and Optimizers

This repository contains following files: <br>
`Assignment_11.ipynb`: Notebook containing the main code. <br>
`Model`: Containing all the models. <br>
`Main.py`:This Python script defines a class S_11 that encapsulates the process of training and evaluating a deep learning model for image classification using the PyTorch library. <br>
`Utils.py`: This Python script is designed for enhancing deep learning models in PyTorch, particularly focusing on image classification tasks. <br>



## Main.py : 
The class is designed to be flexible, allowing for customization of various aspects such as the model architecture, data transformations, optimization algorithm, learning rate scheduling, and more.

At its core, the `S_11` class handles the entire lifecycle of a machine learning experiment. It starts with data preparation, where it splits the dataset into training and test sets, applying specified transformations to each. This is crucial for tasks like image classification, where data augmentation can significantly impact model performance.

The class introduces an innovative approach to finding the optimal learning rate using the `LRFinder` tool, which systematically tests a range of learning rates to identify the one that offers the best compromise between training speed and convergence stability. This step is vital for achieving high performance in deep learning models.

Once the optimal learning rate is identified, `S_11` initializes the model's optimizer and learning rate scheduler based on user preferences. These components are essential for controlling the training process, dictating how the model weights are updated and how the learning rate changes over time, respectively.

The training process itself is iterative, with the model being trained for a specified number of epochs. During training, the script dynamically adjusts learning rates (if a scheduler is used), calculates losses, and updates the model weights. Additionally, it evaluates the model on a separate test dataset to monitor its generalization ability, providing insights into how well the model is expected to perform on unseen data.

Finally, `S_11` offers functionality to visualize the training and testing progress through loss and accuracy plots, providing an intuitive understanding of the model's learning trajectory. This visualization is crucial for diagnosing training issues such as overfitting or underfitting.

In summary, the `S_11` class is a comprehensive tool for conducting image classification experiments in PyTorch, offering flexibility, ease of use, and a robust methodology for optimizing model performance through systematic learning rate finding and adjustment.

## Utils.py :
It introduces several advanced features and methodologies to improve model interpretation, evaluation, and data augmentation, ensuring the development of robust and interpretable models.

The script begins with the definition of the `MisclassificationVisualizer` class, which is crafted to identify and visualize misclassified images by a trained model during its evaluation on a test dataset. This class not only aids in understanding the model's weaknesses by highlighting instances where predictions diverged from actual labels but also supports the visualization of Grad-CAM (Gradient-weighted Class Activation Mapping) outputs. Grad-CAM is a powerful technique for making convolutional neural networks (CNNs) more transparent by visualizing the areas of the input image that are important for predictions. This feature is particularly useful for identifying whether the model is focusing on relevant features of the images to make its decisions, thereby offering insights into model behavior and potential biases.

Additionally, the script introduces a custom transformation pipeline using the Albumentations library, a fast and flexible image augmentation library. This pipeline is applied to both training and testing datasets to normalize image data and apply transformations such as random cropping and coarse dropout. Such transformations are essential for making the model robust to variations and noise in the input data, ultimately improving its generalizability.

The `get_mean_and_std` function calculates the mean and standard deviation of a dataset, which are crucial statistics for normalizing images. Normalization ensures that the model's input features are on a similar scale, which can significantly speed up the learning process and lead to faster convergence.

In summary, this script encompasses a comprehensive approach towards building, evaluating, and understanding deep learning models for image classification tasks. By incorporating visualization of misclassified images, leveraging Grad-CAM for interpretability, and utilizing advanced data augmentation techniques, it addresses several critical aspects of model development and evaluation, making it a valuable tool for researchers and practitioners aiming to develop high-performing and interpretable deep learning models.

## Assignment_11.ipynb:
### Suggested LR: 5.34E+00

![image](https://github.com/Omkar1634/ERA_V2/assets/64948764/9bb0dbdb-a085-4306-a723-21c90933c404)

![image](https://github.com/Omkar1634/ERA_V2/assets/64948764/4a621ced-1a14-4b28-a7ba-7cef77013c47)

The provided image shows four line graphs detailing the progression of training and testing loss and accuracy over time for a machine learning model. 

In the top left graph, "Training Loss," we observe a significant decrease in loss over what appears to be a number of epochs. Initially, there is a steep drop, suggesting rapid learning in the early stages of training. As training progresses, the rate of decrease slows, but the trend continues downward, indicating ongoing improvement in the model’s ability to fit the training data.

The bottom left graph, "Training Accuracy," shows a complementary increase in accuracy on the training data. Starting from a lower accuracy, there's a rapid rise in the initial epochs, followed by a more gradual ascent. This trend generally indicates effective learning, as the model is becoming better at correctly classifying the training data over time.

On the top right, the "Test Loss" graph displays a somewhat volatile but generally downward trend. The fluctuations in test loss suggest variance in the model's performance on the test data from epoch to epoch, which is not unusual in training scenarios. However, the overall reduction in loss is a positive sign, pointing to the model's increasing ability to generalize from its training.

The "Test Accuracy" graph in the bottom right shows a stepwise increase in accuracy on the test set. Unlike the training accuracy, this graph isn’t as smooth, displaying some variability and even occasional dips in performance. However, the overall trajectory is upward, indicating that the model is improving at generalizing to new, unseen data.

The key takeaway from these graphs is that the model's ability both to fit the training data and to generalize to new data is improving over time. The observed fluctuations in test metrics are common, especially in the context of deep learning models, where each epoch can result in different local minima due to the stochastic nature of optimization algorithms like SGD (Stochastic Gradient Descent). The graphs suggest that more epochs could potentially lead to further improvements, but care should be taken to monitor for overfitting, where training accuracy continues to improve while test accuracy starts to decline.



