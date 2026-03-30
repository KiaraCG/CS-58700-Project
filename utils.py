import numpy as np
import matplotlib.pyplot as plt

def one_hot(y, n_classes):
    """
    Encode labels into ont-hot vectors
    """
    m = y.shape[0]
    y_1hot = np.zeros((m, n_classes), dtype=np.float32)
    y_1hot[np.arange(m), np.squeeze(y)] = 1
    return y_1hot

def save_plots(losses, train_accs, test_accs):
    """
    Plot two figures: loss vs. epoch and accuracy vs. epoch
    """
    n = len(losses)
    xs = np.arange(n)

    # plot losses
    fig, ax = plt.subplots()
    ax.plot(xs, losses, '--', linewidth=2, label='loss')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(loc='lower right')
    plt.savefig('loss.png')

    # plot train and test accuracies
    plt.clf()
    fig, ax = plt.subplots()
    ax.plot(xs, train_accs, '--', linewidth=2, label='train')
    ax.plot(xs, test_accs, '-', linewidth=2, label='test')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.legend(loc='lower right')
    plt.savefig('accuracy.png')