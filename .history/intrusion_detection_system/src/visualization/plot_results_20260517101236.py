import matplotlib.pyplot as plt
import seaborn as sns

def plot_label_distribution(y):
    plt.figure(figsize=(6, 4))
    sns.countplot(x=y)
    plt.title("Label Distribution (0: Normal, 1: Attack)")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.show()

def plot_feature_distribution(df, feature):
    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x=feature, hue='label', kde=True)
    plt.title(f"Distribution of {feature} by Label")
    plt.show()

def plot_confusion_matrix(cm):
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Attack'],
                yticklabels=['Normal', 'Attack'])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()