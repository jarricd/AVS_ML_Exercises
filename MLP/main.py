import utils
import arch
import torch

if __name__ == "__main__":
    lda_dataset, lda_class = utils.load_data(2, "LDA")

    mlp = arch.MLP()

    # Define the loss function and optimizer
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(mlp.parameters(), lr=1e-4) #SGD was mentioned in the slides
    epochs = 20
    i = 0
    # split into batches
    batch_size = 16
    splitted_batches = torch.split(lda_dataset["train"], batch_size)
    splitted_classes = torch.split(lda_class["train"], batch_size)
    # train
    for epoch in range(0, epochs):
        current_loss = 0.0
        i = 0
        for sample, target_class in zip(splitted_batches, splitted_classes):
            optimizer.zero_grad()  # reset gradient
            outputs = mlp(sample)
            loss = loss_function(outputs, target_class)
            loss.backward()
            optimizer.step()
            current_loss += loss.item()
            if i % 500 == 0:
                print('Loss after mini-batch %5d: %.3f. Epoch: %d' %
                      (i + 1, current_loss / 500, epoch))
                current_loss = 0.0
            i += 1

    with torch.no_grad():
        correct_guesses = 0
        # classify with what we have now
        for item, truth in zip(lda_dataset["test"], lda_class["test"]):
            outputs = mlp(item)
            classification = int(torch.max(outputs)) # slash floating point to have "a closer guess"
            if truth == classification:
                correct_guesses += 1

        accuracy = correct_guesses/len(lda_dataset['test'])
        print(f"Accuracy: {accuracy}")
        pass
