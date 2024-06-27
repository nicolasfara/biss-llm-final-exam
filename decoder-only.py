import random
import io
import os
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, confusion_matrix
from transformers import (set_seed,
                          TrainingArguments,
                          Trainer,
                          GPT2Config,
                          AutoTokenizer,
                          GPT2Tokenizer,
                          AutoModelForSequenceClassification, 
                          AutoTokenizer, 
                          get_scheduler, 
                          get_linear_schedule_with_warmup,
                          get_constant_schedule,
                          GPT2ForSequenceClassification)
from datasets import Dataset
from preprocessing import *
from torch.optim import AdamW, Adagrad
import time
from utils import *
import evaluate
import matplotlib.pyplot as plt
import seaborn as sns

# model_name = "openai-community/gpt2"
# model_name = "ruggsea/gpt-ita-fdi_lega"
model_name = "GroNLP/gpt2-small-italian"

# Maximum length to be considered in input
max_seq_length = 256

# Batch size
batch_size = 16

# Learning rate used during the training process
# If you use large models (such as Bert-large) it is a good idea to use
# smaller values, such as 5e-6
learning_rate = 5e-5

# Name of the fine_tuned_model
output_model_name = "decoder_best_model.pickle"

# Number of training epochs
num_train_epochs = 4

if __name__ == "__main__":
    torch.cuda.empty_cache()
    torch.cuda.memory_summary(device=None, abbreviated=False)

    seed_val = 123
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_val)

    device = setup_torch()
    dataset = retrieve_data_as_dataframe(
        "https://raw.githubusercontent.com/nicolasfara/biss-llm-final-exam/master/data/acti-a/subtaskA_train.csv"
    )
    test_dataset = retrieve_data_as_dataframe(
        "https://raw.githubusercontent.com/nicolasfara/biss-llm-final-exam/master/data/acti-a/subtaskA_test.csv"
    )
    # Clean the text
    cleaned_dataset = dataset.copy()
    cleaned_dataset["comment_text"] = cleaned_dataset["comment_text"].apply(cleanup_text)
    cleaned_dataset.drop(columns=["Id"], inplace=True)
    test_dataset.drop(columns=["Id"], inplace=True)
    labels = cleaned_dataset["conspiratorial"].unique()

    # Split the dataset into training, validation, and testing (using scikitlean)
    train_size = 0.8
    val_size = 0.1
    test_size = 0.1

    # train_dataset, val_dataset, test_dataset = np.split(
    #     cleaned_dataset.sample(frac=1, random_state=seed_val),
    #     [int(train_size * len(cleaned_dataset)), int((train_size + val_size) * len(cleaned_dataset))],
    # )
    
    train_dataset, val_dataset = np.split(
        cleaned_dataset.sample(frac=1, random_state=seed_val),
        [int(train_size * len(cleaned_dataset))],
    )
    
    print(f"Train size: {len(train_dataset)}")
    print(f"Validation size: {len(val_dataset)}")
    print(f"Test size: {len(test_dataset)}")
    
    # Model configuration
    model_config = GPT2Config.from_pretrained(pretrained_model_name_or_path=model_name, num_labels=len(labels))

    # Define a Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    

    def tokenize_function(examples):
        return tokenizer(text=examples["comment_text"], return_tensors="pt", padding=True, truncation=True,  max_length=max_seq_length)
        #return tokenizer(examples["comment_text"], padding="max_length", truncation=True, max_length=max_seq_length)

    train_dataset = Dataset.from_pandas(train_dataset.reset_index(drop=True))
    train_tokenized_datasets = train_dataset.map(tokenize_function, batched=True)
    print(train_tokenized_datasets)
    train_tokenized_datasets = train_tokenized_datasets.remove_columns(["comment_text"])
    train_tokenized_datasets = train_tokenized_datasets.rename_column("conspiratorial", "labels")
    train_tokenized_datasets.set_format("torch")
    val_dataset = Dataset.from_pandas(val_dataset.reset_index(drop=True))
    val_tokenized_datasets = val_dataset.map(tokenize_function, batched=True)
    val_tokenized_datasets = val_tokenized_datasets.remove_columns(["comment_text"])
    val_tokenized_datasets = val_tokenized_datasets.rename_column("conspiratorial", "labels")
    val_tokenized_datasets.set_format("torch")

    train_dataloader = DataLoader(train_tokenized_datasets, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_tokenized_datasets, batch_size=batch_size)

    model = GPT2ForSequenceClassification.from_pretrained(model_name, config=model_config)

    # resize model embedding to match new tokenizers
    model.resize_token_embeddings(len(tokenizer))
    
    # fix model padding token id
    model.config.pad_token_id = model.config.eos_token_id
    
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    num_training_steps = num_train_epochs * len(train_dataloader)
    
    # ['linear', 'cosine', 'cosine_with_restarts', 'polynomial', 'constant', 'constant_with_warmup', 'inverse_sqrt', 'reduce_lr_on_plateau', 'cosine_with_min_lr', 'warmup_stable_decay']
    lr_scheduler = get_scheduler(
        name="constant", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    model.to(device)

    progress_bar = tqdm(range(num_training_steps))
    train_loss = []
    validation_loss = []
    validation_f1_scores = []
    loss_fn = nn.CrossEntropyLoss()    

    # Training
    for epoch in range(num_train_epochs):
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            # The logits will be used for measuring the loss
            pred = outputs.logits
            loss = loss_fn(pred, batch['labels'])
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            # Calculating the running loss for logging purposes
            train_batch_loss = loss.item()
            train_last_loss = train_batch_loss / batch_size

            progress_bar.update(1)
        train_loss.append(train_last_loss)
        # Validation
        model.eval()
        all_labels = []
        all_preds = []
        for i, batch in enumerate(val_dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
                logits = outputs.logits
                loss = loss_fn(logits, batch['labels'])
                test_batch_loss = loss.item()

                test_last_loss = test_batch_loss / batch_size

                preds = logits.argmax(1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch['labels'].cpu().numpy())
    
        validation_loss.append(test_last_loss)
        validation_f1 = f1_score(all_labels, all_preds, average='weighted')
        validation_f1_scores.append(validation_f1)

          
        print("###############################")
        print("f1 score ", validation_f1_scores)
        print("Check ", len(validation_f1_scores) > 1 and validation_f1_scores[-1] > max(validation_f1_scores[:-1]))
        if len(validation_f1_scores) > 1:
            print(validation_f1_scores[-1])
            print(max(validation_f1_scores[:-1]))
        print("###############################")
        # Save the model if the performance on the development set increases
        if len(validation_f1_scores) > 1 and validation_f1_scores[-1] > max(validation_f1_scores[:-1]):
            print(f"Saving model with F1 score {validation_f1_scores[-1]}")
            torch.save(model.state_dict(), output_model_name)
        elif len(validation_f1_scores) <= 1: 
            torch.save(model.state_dict(), output_model_name)

    plt.plot(range(1, num_train_epochs + 1), train_loss, label="Training Loss")
    plt.plot(range(1, num_train_epochs + 1), validation_loss, label="Validation Loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(loc="upper left")
    plt.show()
    plt.savefig('decoder-only-loss.pdf')
    plt.close()
    

    plt.plot(range(1, num_train_epochs + 1), validation_f1_scores, label="Validation F1 Score")
    plt.title("F1 Score")
    plt.xlabel("Epochs")
    plt.ylabel("Validation F1 Score")
    plt.show()
    plt.savefig('decoder-only-F1Score.pdf')
    plt.close()

    # Evaluate the model
    best_model = torch.load(output_model_name)

    # Tokenize the test dataset
    test_dataset = Dataset.from_pandas(test_dataset.reset_index(drop=True))
    test_tokenized_datasets = test_dataset.map(tokenize_function, batched=True)
    test_tokenized_datasets = test_tokenized_datasets.remove_columns(["comment_text"])
    test_tokenized_datasets = test_tokenized_datasets.rename_column("conspiratorial", "labels")
    test_tokenized_datasets.set_format("torch")

    # Create a DataLoader for the test dataset
    test_dataloader = DataLoader(test_tokenized_datasets, batch_size=batch_size)

    # Load the best model
    best_model = GPT2ForSequenceClassification.from_pretrained(model_name, config=model_config)
    best_model.resize_token_embeddings(len(tokenizer))
    best_model.config.pad_token_id = model.config.eos_token_id
    # best_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(labels))
    best_model.load_state_dict(torch.load(output_model_name))
    best_model.to(device)
    best_model.eval()

    # Evaluate the model on the test set
    all_labels = []
    all_preds = []
    
    for i, batch in enumerate(test_dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        #batch = {k: v.type(torch.long).to(device) for k,v in batch.items()}
        with torch.no_grad():
            outputs = best_model(**batch)

        logits = outputs.logits
        preds = logits.argmax(1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch['labels'].cpu().numpy())

    # Calculate the F1 score
    test_f1 = f1_score(all_labels, all_preds, average='weighted')
    print(f"Test F1 Score: {test_f1}")

    # Print the classification report
    print(classification_report(all_labels, all_preds))

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds).tolist()
    sns.heatmap(cm,
                annot=True,
                fmt='g',
                xticklabels=['Non-Conspiratorial', 'Conspiratorial'],
                yticklabels=['Non-Conspiratorial', 'Conspiratorial'],
               )
    plt.xlabel('Predicted',fontsize=13)
    plt.ylabel('Actual',fontsize=13)
    plt.title('Confusion Matrix',fontsize=17)
    plt.savefig('decoder-only-confusion-matrix.pdf', dpi=500)

