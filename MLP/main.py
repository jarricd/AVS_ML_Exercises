import utils

if __name__ == "__main__":
    lda_dataset, lda_class = utils.load_data(2, "LDA")
    pca_dataset_10, pca_class_10 = utils.load_data(10, "PCA")
    pca_dataset_20, pca_class_20 = utils.load_data(20, "PCA")
    pca_dataset_30, pca_class_30 = utils.load_data(30, "PCA")




