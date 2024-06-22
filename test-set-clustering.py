import random
from datasets import Dataset
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_scheduler
from preprocessing import *
from torch.optim import AdamW
import time
from utils import *
import evaluate
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from tqdm.auto import tqdm
from sklearn.cluster import KMeans

model_name = "Musixmatch/umberto-commoncrawl-cased-v1"
# Maximum length to be considered in input
max_seq_length = 256


def tokenize_function(examples):
        return tokenizer(examples["comment_text"], padding="max_length", truncation=True, max_length=max_seq_length)

def get_embeddings(dataloader, device):
   
    # Define the model
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(labels), output_hidden_states=True)
    model.to(device)

    # Inference
    model.eval()
    all_labels = []
    all_preds = []
    for i, batch in enumerate(dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        hidden_states = outputs.hidden_states
        cls_embeddings = hidden_states[-1][:, 0, :].squeeze()
    
    return np.array(cls_embeddings)


if __name__ == "__main__":

    seed_val = 123
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.memory_summary(device=None, abbreviated=False)
        torch.cuda.manual_seed_all(seed_val)

    device = setup_torch()

    dataset = retrieve_data_as_dataframe(
        "https://raw.githubusercontent.com/nicolasfara/biss-llm-final-exam/master/data/acti-a/subtaskA_train.csv"
    )

    test_dataset = retrieve_data_as_dataframe(
        "https://raw.githubusercontent.com/nicolasfara/biss-llm-final-exam/master/data/acti-a/subtaskA_test_with_labels.csv"
    )   

    # test_dataset = test_dataset[:20]
    comments = test_dataset['comment_text'].tolist() # for output purposes

    # Clean the text
    cleaned_dataset = dataset.copy()
    cleaned_dataset["comment_text"] = cleaned_dataset["comment_text"].apply(cleanup_text)
    test_dataset["comment_text"] = test_dataset["comment_text"].apply(cleanup_text)
    cleaned_dataset.drop(columns=["Id"], inplace=True)
    test_dataset.drop(columns=["Id"], inplace=True)
    labels = cleaned_dataset["conspiratorial"].unique()

    # Define the tokenizer  
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    test_dataset = Dataset.from_pandas(test_dataset.reset_index(drop=True))
    test_tokenized_datasets = test_dataset.map(tokenize_function, batched=True)
    test_tokenized_datasets = test_tokenized_datasets.remove_columns(["comment_text"])
    test_tokenized_datasets = test_tokenized_datasets.rename_column("conspiratorial", "labels")
    test_tokenized_datasets.set_format("torch")

    test_dataloader = DataLoader(test_tokenized_datasets, batch_size=len(test_tokenized_datasets))

    embeddings = get_embeddings(test_dataloader, device)

    num_clusters = 2 
    kmeans = KMeans(n_clusters=num_clusters, random_state=seed_val).fit(embeddings)

    data = {'comment': comments,
        'conspiratorial': kmeans.labels_ }

    out_df = pd.DataFrame(data)
    out_df.to_csv('data/acti-a/subtaskA_test_clustered_labes.csv', index=False)