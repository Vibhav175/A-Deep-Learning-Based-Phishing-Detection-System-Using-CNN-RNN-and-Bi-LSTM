# evaluations.py

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import os

def evaluate_model(model, history, X_test, y_test, save_results_dir='results'):
    os.makedirs(save_results_dir, exist_ok=True)

    # Plot the loss curve
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curve')
    plt.savefig(os.path.join(save_results_dir, 'loss_curve.png'))
    plt.show()

    # Plot the accuracy curve
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy Curve')
    plt.savefig(os.path.join(save_results_dir, 'accuracy_curve.png'))
    plt.show()

    # Predictions
    y_pred = model.predict(X_test)
    y_pred_classes = [round(pred[1]) for pred in y_pred]

    # Classification Report
    print("Classification Report:")
    classification_report_str = classification_report(y_test[:, 1], y_pred_classes)
    print(classification_report_str)
    with open(os.path.join(save_results_dir, 'classification_report.txt'), 'w') as report_file:
        report_file.write(classification_report_str)

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test[:, 1], y_pred_classes)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Benign', 'Phishing'], yticklabels=['Benign', 'Phishing'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(save_results_dir, 'confusion_matrix.png'))
    plt.show()

    # Save the model summary
    with open(os.path.join(save_results_dir, 'model_summary.txt'), 'w') as summary_file:
        model.summary(print_fn=lambda x: summary_file.write(x + '\n'))
