Plant Disease Prediction Using Hybrid Deep CNN | ResNet50 | Kaggle Dataset
Developed a cotton plant disease prediction model using a Hybrid Deep CNN with ResNet50 architecture. Trained and tested the model on a Kaggle cotton leaf disease dataset for accurate disease classification and early detection.

Project Overview
Cotton Plant Disease Prediction using Hybrid Deep CNN and ResNet50 trained on Kaggle dataset.
• Developed a Cotton Disease Prediction System using Deep Learning (CNN with ResNet architecture) for accurate image
classification.
• Performed data preprocessing and augmentation (resizing, normalization, flipping, rotation) to improve model
generalization.
• Implemented the model using PyTorch (torch, tensor operations) and applied transfer learning for faster training and
better performance.
• Achieved reliable performance by using CrossEntropyLoss and Adam optimizer, along with validation accuracy
monitoring.
• Deployed the model using Streamlit to create an interactive UI for real-time disease prediction from uploaded images

Dataset
Cotton Plant Disease Prediction using Hybrid Deep CNN and ResNet50 trained on Kaggle dataset.

Steps Performed

a) Data Preprocesing b) Handled missing values c) Cleaned and structured raw data.

EDA

Improved model performance by using RasNet-50.

Train-Test-Split

a)Split data into training,testing and valid sets.
b) Test Accuracy: 90.56603773584905
c) Confusion Matrix :
                [[23  0  1  1]
                [ 1 27  0  0]
                [ 0  1 25  0]
                [ 0  6  0 21]]

d)Classification Report:-
                      precision    recall  f1-score   support

 diseased cotton leaf       0.96      0.92      0.94        25
diseased cotton plant       0.79      0.96      0.87        28
    fresh cotton leaf       0.96      0.96      0.96        26
   fresh cotton plant       0.95      0.78      0.86        27

             accuracy                           0.91       106
            macro avg       0.92      0.91      0.91       106
         weighted avg       0.91      0.91      0.91       106

After RasNet-50 Apply:-
a) ResNet Accuracy: 96.22641509433963
b)Report Classification:-
                        precision    recall  f1-score   support

 diseased cotton leaf       0.96      0.92      0.94        25
diseased cotton plant       0.79      0.96      0.87        28
    fresh cotton leaf       0.96      0.96      0.96        26
   fresh cotton plant       0.95      0.78      0.86        27

             accuracy                           0.91       106
            macro avg       0.92      0.91      0.91       106
         weighted avg       0.91      0.91      0.91       106
