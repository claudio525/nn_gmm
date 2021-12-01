import numpy as np
import matplotlib.pyplot as plt

def vis_results(y: np.ndarray, est_y: np.ndarray, title="Title"):
    fig = plt.figure(figsize=(16, 10), dpi=200)

    plt.scatter(y, est_y, alpha=0.5, s=0.5)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], c="k")
    plt.xlabel("True")
    plt.ylabel("Estimated")

    plt.grid(linestyle="--", linewidth=0.5, alpha=0.5)
    plt.title(title)
    plt.tight_layout()

    plt.show()


def compare_results(y: np.ndarray, est_y_1: np.ndarray, est_y_2: np.ndarray):
    fig = plt.figure(figsize=(16, 10), dpi=200)

    plt.scatter(y, est_y_1, alpha=0.4, s=0.5, c="b")
    plt.scatter(y, est_y_2, alpha=0.4, s=0.5, c="r")
    plt.plot([y.min(), y.max()], [y.min(), y.max()], c="k")
    plt.xlabel("True")
    plt.ylabel("Estimated")

    plt.grid(linestyle="--", linewidth=0.5, alpha=0.5)
    plt.tight_layout()

    plt.show()