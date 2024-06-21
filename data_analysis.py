from utils import *
import seaborn as sns
from matplotlib import pyplot as plt

if __name__ == '__main__':

    sns.set_theme(style="whitegrid")

    dataset = retrieve_data_as_dataframe(
        "https://raw.githubusercontent.com/nicolasfara/biss-llm-final-exam/master/data/acti-a/subtaskA_train.csv"
    )
    # Comparare anche il test set

    # plot the class distribution over the "conspiratorial" column
    # plot number on top of bar
    plt.figure(figsize=(10, 6), layout='tight')
    g = sns.countplot(x='conspiratorial', data=dataset, hue='conspiratorial', legend=False)
    g.set_title("Class Distribution", fontsize=18)
    g.set_xticklabels(['Not Conspiratorial', 'Conspiratorial'])
    g.set_xlabel("class")
    plt.savefig("charts/class_distribution.pdf")
    plt.show()

    # plot the distribution length of the comments using the columns "comment_text"
    dataset['comment_text_length'] = dataset['comment_text'].apply(lambda x: len(x))
    plt.figure(figsize=(10, 6), layout='tight')
    sns.histplot(dataset['comment_text_length'], kde=True)
    plt.title("Distribution of Comment Length", fontsize=18)
    plt.xlabel("Comment Length (characters)")
    plt.ylabel("Frequency")
    plt.savefig("charts/comment_length_distribution.pdf")
    plt.show()
