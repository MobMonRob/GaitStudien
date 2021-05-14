import matplotlib.pyplot as plt


def plot_angles(angles, title):
    plt.plot(angles, 'r')
    plt.grid(True)
    plt.title(title)
    plt.xlabel("Index")
    plt.ylabel("Angles")
    plt.show()

def plot_classification_results(feature_combinations, f1_scores, type):
    plt.title("F1-Werte für e")
    plt.xlabel("F1-Wert")
    plt.ylabel()