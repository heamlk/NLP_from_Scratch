import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import sys

import src.preprocess.dataset as dataset
import src.evaluate.evaluate as evaluate


def visualize(classes, pretrained=False, name="Preston"):

    # Set up plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    # NumPy uses CPU here so we need to use a CPU version
    cax = ax.matshow(evaluate.confusion_matrix(pretrained, name).cpu().numpy())
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticks(np.arange(len(classes)), labels=classes, rotation=90)
    ax.set_yticks(np.arange(len(classes)), labels=classes)

    # Force label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    # Set up title
    plt.title(f"Languages the RNN Guessed Correctly for the name {name}")

    plt.savefig("reports/figures/confusion.png")
    
    plt.show()

visualize(classes=dataset.alldata.labels_uniq) if "visualize" in sys.argv else None
