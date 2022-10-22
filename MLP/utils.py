import numpy
import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import torch
import pickle

def load_data(n_dim, mode: str = "LDA"):
    test_filenames = [f"test{i}.txt" for i in range(0, 10)]
    train_filenames = [f"train{i}.txt" for i in range(0, 10)]
    dataset_dict = {"train": [], "test": []}
    class_dataset = {"train": [], "test": []}
    i = 0
    for test_file, train_file in zip(test_filenames, train_filenames):
        train = np.loadtxt(f"mnist_all/{train_file}")
        test = np.loadtxt(f"mnist_all/{test_file}")
        class_vector_train = np.ones(len(train)) * i
        class_vector_test = np.ones(len(test)) * i
        dataset_dict["train"].append(train)
        dataset_dict["test"].append(test)
        class_dataset["train"].append(class_vector_train)
        class_dataset["test"].append(class_vector_test)

        i += 1

    dataset_dict["train"] = np.concatenate(dataset_dict["train"])
    dataset_dict["test"] = np.concatenate(dataset_dict["test"])
    class_dataset["train"] = torch.from_numpy(np.concatenate(class_dataset["train"])).type(torch.long)
    class_dataset["test"] = torch.from_numpy(np.concatenate(class_dataset["test"])).type(torch.long)
    if mode == "LDA":
        lda_train = LinearDiscriminantAnalysis(n_components=n_dim)
        lda_train.fit(np.array(dataset_dict["train"]), class_dataset["train"])
        reduced_train = lda_train.transform(dataset_dict["train"])
        lda_test = LinearDiscriminantAnalysis(n_components=n_dim)
        lda_test.fit(np.array(dataset_dict["test"]), class_dataset["test"])
        reduced_test = lda_test.transform(dataset_dict["test"])

        dataset_dict["train"] = torch.from_numpy(reduced_train.astype(numpy.float32))
        dataset_dict["test"] = torch.from_numpy(reduced_test.astype(numpy.float32))
    elif mode == "PCA":
        lda_train = PCA(n_components=n_dim)
        lda_train.fit(np.array(dataset_dict["train"]), class_dataset["train"])
        reduced_train = lda_train.transform(dataset_dict["train"])
        lda_test = PCA(n_components=n_dim)
        lda_test.fit(np.array(dataset_dict["test"]), class_dataset["test"])
        reduced_test = lda_test.transform(dataset_dict["test"])

        dataset_dict["train"] = torch.from_numpy(reduced_train.astype(numpy.float32))
        dataset_dict["test"] = torch.from_numpy(reduced_test.astype(numpy.float32))
    else:
        return {}, {}  # invalid mode
    return dataset_dict, class_dataset


if __name__ == "__main__":
    lda_dataset, lda_class = load_data(2, "LDA")
    pca_dataset_10, pca_class_10 = load_data(10, "PCA")
    pca_dataset_20, pca_class_20 = load_data(20, "PCA")
    pca_dataset_30, pca_class_30 = load_data(30, "PCA")
    with open("lda.dat", 'wb') as pickle_file:
        pickle.dump((lda_dataset, lda_class), pickle_file)
    with open("pca_10.dat", 'wb') as pickle_file:
        pickle.dump((pca_dataset_10, pca_class_10), pickle_file)
    with open("pca_20.dat", 'wb') as pickle_file:
        pickle.dump((pca_dataset_20, pca_class_20), pickle_file)
    with open("pca_30.dat", 'wb') as pickle_file:
        pickle.dump((pca_dataset_30, pca_class_30), pickle_file)

