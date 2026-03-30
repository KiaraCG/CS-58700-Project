import torch
import sklearn.metrics as metrics
import logging
from G_equivariant_horizontal_reflections import Z2CNN
from load_datasets import mnist
from minibatcher import MiniBatcher
from utils import one_hot, save_plots

MAX_EPOCHS = 20
LEARNING_RATE = 1e-4

def main():
    X_train, y_train = mnist.load_train_data()
    X_test, y_test = mnist.load_test_data()

    # reshape the images into one dimension
    X_train = X_train.reshape((X_train.shape[0], -1))
    y_train_1hot = one_hot(y_train, mnist.N_CLASSES)
    X_test = X_test.reshape((X_test.shape[0], -1))
    y_test_1hot = one_hot(y_test, mnist.N_CLASSES)

    # to torch tensor
    X_train, y_train, y_train_1hot = torch.from_numpy(X_train), torch.from_numpy(y_train), torch.from_numpy(
        y_train_1hot)
    X_train = X_train.type(torch.FloatTensor)
    X_test, y_test, y_test_1hot = torch.from_numpy(X_test), torch.from_numpy(y_test), torch.from_numpy(y_test_1hot)
    X_test = X_test.type(torch.FloatTensor)

    # set up model
    n_examples = X_train.data.shape[0]
    shape = [X_train.shape[1], 300, 100, mnist.N_CLASSES]
    model = Z2CNN(reflection_type = 'horizontal')

    # start training
    losses = []
    train_accs = []
    test_accs = []

    # this only shuffles the indices
    batcher = MiniBatcher(n_examples, n_examples)
    for i_epoch in range(MAX_EPOCHS):
        logging.info("---------- EPOCH {} ----------".format(i_epoch))

        for train_idxs in batcher.get_one_batch():

            # fit to the training data
            loss = model.train_one_epoch(X_train[train_idxs], y_train[train_idxs], y_train_1hot[train_idxs],
                                         LEARNING_RATE)
            logging.info("loss = {}".format(loss))

            # monitor training and testing accuracy
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            train_acc = metrics.accuracy_score(y_train, y_train_pred)
            test_acc = metrics.accuracy_score(y_test, y_test_pred)
            logging.info("Accuracy(train) = {}".format(train_acc))
            logging.info("Accuracy(test) = {}".format(test_acc))

        # collect results for plotting for each epoch
        loss = model.loss(X_train, y_train, y_train_1hot)
        losses.append(loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)

    # plot
    save_plots(losses, train_accs, test_accs)

if __name__ == '__main__':
    main()
