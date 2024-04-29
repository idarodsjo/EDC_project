import matplotlib.pyplot as plt
import seaborn as sns

def confusion_matrix(title, confusion_matrix, error_rate, visualize):
    if visualize:
        plt.figure(figsize=(10, 7))
        plt.title('Confusion matrix for ' + title + '\n' + 'Error rate: ' + str(error_rate * 100) + '%')
        sns.heatmap(confusion_matrix, annot=True, fmt='.0f')
        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        plt.show()