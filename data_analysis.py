from utils import *
from preprocessing import cleanup_text
import seaborn as sns
from matplotlib import pyplot as plt

if __name__ == '__main__':
    sns.set_theme(style="whitegrid")

    dataset = retrieve_data_as_dataframe(
        "https://raw.githubusercontent.com/nicolasfara/biss-llm-final-exam/master/data/acti-a/subtaskA_train.csv"
    )
    test_dataset = retrieve_data_as_dataframe(
        "https://raw.githubusercontent.com/nicolasfara/biss-llm-final-exam/master/data/acti-a/subtaskA_test.csv"
    )
    # Compare side by side the distribution of the classes of the dataset and the test dataset
    _, axes = plt.subplots(1, 2, figsize=(20, 5), layout='tight')

    sns.countplot(x='conspiratorial', data=dataset, ax=axes[0], hue='conspiratorial', legend=False)
    axes[0].set_title("Train Dataset Class Distribution", fontsize=16)
    axes[0].set_xticklabels(['Not Conspiratorial', 'Conspiratorial'], fontsize=14)
    axes[0].set_xlabel("class", fontsize=14)

    sns.countplot(x='conspiratorial', data=test_dataset, ax=axes[1], hue='conspiratorial', legend=False)
    axes[1].set_title("Test Dataset Class Distribution", fontsize=16)
    axes[1].set_xticklabels(['Not Conspiratorial', 'Conspiratorial'], fontsize=14)
    axes[1].set_xlabel("class", fontsize=14)
    plt.savefig("charts/class_distribution.pdf")

    # plot the distribution length of the comments using the columns "comment_text" for dataset and test_dataset

    _, axes = plt.subplots(2, 2, figsize=(20, 7), layout='tight', sharex=True)
    dataset['comment_text_length'] = dataset['comment_text'].apply(lambda x: len(x))
    sns.histplot(dataset['comment_text_length'], kde=True, ax=axes[0, 0])
    axes[0, 0].set_title("Train Dataset Comment Length Distribution", fontsize=16)
    axes[0, 0].set_xlabel("Comment Length (characters)", fontsize=14)
    axes[0, 0].set_ylabel("Frequency", fontsize=14)

    test_dataset['comment_text_length'] = test_dataset['comment_text'].apply(lambda x: len(x))
    sns.histplot(test_dataset['comment_text_length'], kde=True, ax=axes[0, 1])
    axes[0, 1].set_title("Test Dataset Comment Length Distribution", fontsize=16)
    axes[0, 1].set_xlabel("Comment Length (characters)", fontsize=14)
    axes[0, 1].set_ylabel("Frequency", fontsize=14)

    dataset['comment_text_length_cleaned'] = dataset['comment_text'].apply(lambda x: len(cleanup_text(x)))
    sns.histplot(dataset['comment_text_length_cleaned'], kde=True, ax=axes[1, 0])
    axes[1, 0].set_title("Train Dataset Comment Length Distribution (Cleaned)", fontsize=16)
    axes[1, 0].set_xlabel("Comment Length (characters)", fontsize=14)
    axes[1, 0].set_ylabel("Frequency", fontsize=14)

    test_dataset['comment_text_length_cleaned'] = test_dataset['comment_text'].apply(lambda x: len(cleanup_text(x)))
    sns.histplot(test_dataset['comment_text_length_cleaned'], kde=True, ax=axes[1, 1])
    axes[1, 1].set_title("Test Dataset Comment Length Distribution (Cleaned)", fontsize=16)
    axes[1, 1].set_xlabel("Comment Length (characters)", fontsize=14)
    axes[1, 1].set_ylabel("Frequency", fontsize=14)
    plt.savefig("charts/comment_length_distribution.pdf")

    # provide statistics for the train and test dataset
    print("Train Dataset Statistics")
    print(dataset.describe())
    print("Test Dataset Statistics")
    print(test_dataset.describe())

    plt.show()
