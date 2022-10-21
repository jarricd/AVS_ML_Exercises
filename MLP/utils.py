import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


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
    class_dataset["train"] = np.concatenate(class_dataset["train"])
    class_dataset["test"] = np.concatenate(class_dataset["test"])
    if mode == "LDA":
        lda_train = LinearDiscriminantAnalysis(n_components=n_dim)
        lda_train.fit(np.array(dataset_dict["train"]), class_dataset["train"])
        reduced_train = lda_train.transform(dataset_dict["train"])
        lda_test = LinearDiscriminantAnalysis(n_components=n_dim)
        lda_test.fit(np.array(dataset_dict["test"]), class_dataset["test"])
        reduced_test = lda_test.transform(dataset_dict["test"])

        dataset_dict["train"] = reduced_train
        dataset_dict["test"] = reduced_test
    elif mode == "PCA":
        lda_train = PCA(n_components=n_dim)
        lda_train.fit(np.array(dataset_dict["train"]), class_dataset["train"])
        reduced_train = lda_train.transform(dataset_dict["train"])
        lda_test = PCA(n_components=n_dim)
        lda_test.fit(np.array(dataset_dict["test"]), class_dataset["test"])
        reduced_test = lda_test.transform(dataset_dict["test"])

        dataset_dict["train"] = reduced_train
        dataset_dict["test"] = reduced_test
    else:
        return {}, {}  # invalid mode
    return dataset_dict, class_dataset
