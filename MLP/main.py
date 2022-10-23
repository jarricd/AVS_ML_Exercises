import utils
import arch
import torch
import pickle
from torch.utils.tensorboard import SummaryWriter
from sklearn.neural_network import MLPClassifier

def train_model(dataset, dset_class, model_class):
    writer = SummaryWriter()
    # Define the loss function and optimizer
    mlp = model_class()
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(mlp.parameters(), lr=1e-5) #SGD was mentioned in the slides
    epochs = 32
    i = 0
    # split into batches
    batch_size = 16
    splitted_batches = torch.split(dataset["train"], batch_size)
    splitted_classes = torch.split(dset_class["train"], batch_size)
    # train
    for epoch in range(0, epochs):
        current_loss = 0.0
        i = 0
        for sample, target_class in zip(splitted_batches, splitted_classes):
            optimizer.zero_grad()  # reset gradient
            outputs = mlp(sample)
            loss = loss_function(outputs, target_class)
            writer.add_scalar("Loss/train", loss, epoch)
            loss.backward()
            optimizer.step()
            current_loss += loss.item()
            if i % 256 == 255:
                print('Loss after mini-batch %5d: %.3f. Epoch: %d' %
                      (i + 1, current_loss / 256, epoch + 1))
                current_loss = 0.0
            i += 1

    with torch.no_grad():
        correct_guesses = 0

        # classify with what we have now
        for item, truth in zip(dataset["test"], dset_class["test"]):
            outputs = mlp(item)
            _, classifications = torch.max(outputs, 0)
            
            if truth == classifications:
                correct_guesses += 1

        accuracy = correct_guesses/len(dataset['test'])
        print(f"Accuracy_lda1: {accuracy*100}")
        writer.add_scalar("Accuracy", accuracy*100)
    writer.close()


def sklearn_classifier(dataset, dataset_classes):
    mlp_classifier = MLPClassifier(random_state=69, max_iter=200, solver="sgd", hidden_layer_sizes=(3, 500))
    mlp_classifier.fit(dataset["train"], dataset_classes["train"])
    correct_classifications = 0
    output = mlp_classifier.predict(dataset["test"])
    for item, truth in zip(output, dataset_classes["test"]):
        if item == truth:
            correct_classifications += 1

    accuracy = correct_classifications / len(dataset['test'])
    print(f"Accuracy: {accuracy * 100}")


if __name__ == "__main__":
    torch.manual_seed(0)
    lda_dataset, lda_class = utils.load_data(2, "LDA")
    with open('lda.dat', "rb") as f:
        lda_dataset_nn, lda_class_nn = pickle.load(f)
    with open('pca_10.dat', "rb") as f:
        pca_dataset_10_nn, pca_class_10_nn = pickle.load(f)
    with open('pca_20.dat', "rb") as f:
        pca_dataset_20_nn, pca_class_20_nn = pickle.load(f)
    with open('pca_30.dat', "rb") as f:
        pca_dataset_30_nn, pca_class_30_nn = pickle.load(f)

    train_model(lda_dataset_nn, lda_class_nn, arch.MLP)
    train_model(pca_dataset_10_nn, pca_class_10_nn, arch.MLP10)
    train_model(pca_dataset_20_nn, pca_class_20_nn, arch.MLP20)
    train_model(pca_dataset_30_nn, pca_class_30_nn, arch.MLP30)

    with open('lda_numpy.dat', "rb") as f:
        lda_dataset, lda_class = pickle.load(f)
    with open('pca_10_numpy.dat', "rb") as f:
        pca_dataset_10, pca_class_10 = pickle.load(f)
    with open('pca_20_numpy.dat', "rb") as f:
        pca_dataset_20, pca_class_20 = pickle.load(f)
    with open('pca_30_numpy.dat', "rb") as f:
        pca_dataset_30, pca_class_30 = pickle.load(f)
    print("LDA")
    sklearn_classifier(lda_dataset, lda_class)
    print("PCA10")
    sklearn_classifier(pca_dataset_10, pca_class_10)
    print("PCA20")
    sklearn_classifier(pca_dataset_20, pca_class_20)
    print("PCA30")
    sklearn_classifier(pca_dataset_30, pca_class_30)