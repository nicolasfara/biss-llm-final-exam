from utils import *
import seaborn as sns
from matplotlib import pyplot as plt

if __name__ == '__main__':

    sns.set_theme(style="whitegrid")

    dataset = retrieve_data_as_dataframe(
        "https://raw.githubusercontent.com/nicolasfara/biss-llm-final-exam/master/data/acti-a/subtaskA_train.csv"
    )
    # plot the class distribution over the "conspiratorial" column
    # plot number on top of bar
    plt.figure(figsize=(10, 6), layout='tight')
    g = sns.countplot(x='conspiratorial', data=dataset, hue='conspiratorial', legend=False)
    g.set_title("Class Distribution", fontsize=20)
    g.set_xticklabels(['Not Conspiratorial', 'Conspiratorial'])
    g.set_xlabel("class")
    plt.savefig("charts/class_distribution.pdf")
    plt.show()
