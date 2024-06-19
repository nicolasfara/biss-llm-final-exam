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
from tqdm.auto import tqdm

model_name = "Musixmatch/umberto-commoncrawl-cased-v1"
# Maximum length to be considered in input
max_seq_length = 256
# Dropout applied to the embedding produced by BERT before the classifiation
out_dropout_rate = 0.1
# Dev percentage split, i.e., the percentage of training material to be use for
# evaluating the model during training
dev_perc = 0.1
# Batch size
batch_size = 32
# Learning rate used during the training process
# If you use large models (such as Bert-large) it is a good idea to use
# smaller values, such as 5e-6
learning_rate = 5e-6
# Name of the fine_tuned_model
output_model_name = "best_model.pickle"
# Number of training epochs
num_train_epochs = 6
# ADVANCED: Schedulers allow to define dynamic learning rates.
# You can find all available schedulers here
# https://huggingface.co/transformers/main_classes/optimizer_schedules.html
apply_scheduler = False
# Here a `Constant schedule with warmup`can be activated. More details here
# https://huggingface.co/transformers/main_classes/optimizer_schedules.html#transformers.get_constant_schedule_with_warmup
warmup_proportion = 0.1
# Print a log each n steps
print_each_n_step = 10


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
    # Clean the text
    cleaned_dataset = dataset.copy()
    cleaned_dataset["comment_text"] = cleaned_dataset["comment_text"].apply(cleanup_text)
    cleaned_dataset.drop(columns=["Id"], inplace=True)
    labels = cleaned_dataset["conspiratorial"].unique()

    # Split the dataset into training, validation, and testing (using scikitlean)
    train_size = 0.8
    val_size = 0.1
    test_size = 0.1

    train_dataset, val_dataset, test_dataset = np.split(
        cleaned_dataset.sample(frac=1, random_state=seed_val),
        [int(train_size * len(cleaned_dataset)), int((train_size + val_size) * len(cleaned_dataset))],
    )

    print(f"Train size: {len(train_dataset)}")
    print(f"Validation size: {len(val_dataset)}")
    print(f"Test size: {len(test_dataset)}")

    # Define a Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)


    def tokenize_function(examples):
        return tokenizer(examples["comment_text"], padding="max_length", truncation=True, max_length=max_seq_length)

    train_dataset = Dataset.from_pandas(train_dataset.reset_index(drop=True))
    train_tokenized_datasets = train_dataset.map(tokenize_function, batched=True)
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

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(labels))
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    num_training_steps = num_train_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    model.to(device)

    progress_bar = tqdm(range(num_training_steps))
    train_loss = []
    validation_loss = []
    validation_accuracy = []
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
        correct = 0
        model.eval()
        for i, batch in enumerate(val_dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)

            logits = outputs.logits
            loss = loss_fn(logits, batch['labels'])
            test_batch_loss = loss.item()

            test_last_loss = test_batch_loss / batch_size

            correct += (logits.argmax(1) == batch['labels']).sum().item()
            last_validation_accuracy = correct/((i + 1) * batch_size)

        validation_loss.append(test_last_loss)
        validation_accuracy.append(last_validation_accuracy)

    plt.plot(range(1, num_train_epochs + 1), train_loss, label="Training Loss")
    plt.plot(range(1, num_train_epochs + 1), validation_loss, label="Validation Loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(loc="upper left")
    plt.show()

    plt.plot(range(1, num_train_epochs + 1), validation_accuracy, label="Validation Accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Validation Accuracy")
    plt.show()



    # # Test the model
    # test_dataset = Dataset.from_pandas(test_dataset.reset_index(drop=True))
    # test_tokenized_datasets = test_dataset.map(tokenize_function, batched=True)
    # test_tokenized_datasets = test_tokenized_datasets.remove_columns(["comment_text"])
    # test_tokenized_datasets = test_tokenized_datasets.rename_column("conspiratorial", "labels")
    # test_tokenized_datasets.set_format("torch")
    # test_dataloader = DataLoader(test_tokenized_datasets, batch_size=batch_size)
    # model.eval()
    # for batch in test_dataloader:
    #     batch = {k: v.to(device) for k, v in batch.items()}
    #     with torch.no_grad():
    #         outputs = model(**batch)
    #
    #     logits = outputs.logits
    #     predictions = torch.argmax(logits, dim=-1)
    #     metric.add_batch(predictions=predictions, references=batch["labels"])
    #
    # print("Evaluation on test set")
    # print(metric.compute())

    # # Print the length distribution
    # plt.style.use("ggplot")
    # plt.hist(
    #     [len(tokenizer.encode_plus(text)["input_ids"]) for text, label in cleaned_dataset.itertuples(index=False)],
    #     bins=20
    # )
    # plt.title("Length Distribution of the Comments")
    # plt.xlabel("Number of Tokens")
    # plt.ylabel("Number of Comments")
    # plt.show()
    #
    # # Generate DataLoaders
    # label_map = {1: 1, 0: 0}
    # train_dataloader = generate_data_loader(train_dataset, label_map, tokenizer, max_seq_length, batch_size,
    #                                         do_shuffle=True)
    # val_dataloader = generate_data_loader(val_dataset, label_map, tokenizer, max_seq_length, batch_size)
    # test_dataloader = generate_data_loader(test_dataset, label_map, tokenizer, max_seq_length, batch_size)
    # print("DataLoaders generated successfully")
    #
    # # Define the model
    # classifier = Classifier(model_name, num_labels=len(labels), dropout_rate=out_dropout_rate)
    # # Put the model in the device (GPU or MPS or CPU)
    # classifier.to(device)
    # # Define the Optimizer. Here the ADAM optimizer (a sort of standard de-facto) is
    # # used. AdamW is a variant which also adopts Weigth Decay.
    # optimizer = torch.optim.AdamW(classifier.parameters(), lr=learning_rate)
    # # Define the loss function. Here the CrossEntropyLoss is used, which is the
    # # standard loss function for classification tasks.
    # nll_loss = nn.CrossEntropyLoss(ignore_index=-1)
    #
    # # Define the scheduler
    # num_training_steps = len(train_dataloader) * num_train_epochs
    # num_warmup_steps = int(warmup_proportion * num_training_steps)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=num_warmup_steps, T_mult=1)
    #
    # # Training
    # training_stats = []
    #
    # # Define the LOSS function. A CrossEntropyLoss is used for multi-class
    # # classification tasks.
    # nll_loss = torch.nn.CrossEntropyLoss(ignore_index=-1)
    # # All loss functions are available at https://pytorch.org/docs/stable/nn.html#loss-functions
    #
    # # Measure the total training time for the whole run
    # total_t0 = time.time()
    #
    # # NOTE: the measure to be maximized should depends on the task
    # # Here accuracy is used
    # best_dev_accuracy = -1
    #
    # for epoch_i in range(0, num_train_epochs):
    #     # --------
    #     # Training
    #     # --------
    #
    #     # Perform one full pass over the training set
    #     print("")
    #     print("======== Epoch {:} / {:} ========".format(epoch_i + 1, num_train_epochs))
    #     print("Training...")
    #
    #     # Measure how long the training epoch takes
    #     t0 = time.time()
    #     # Reset the total loss for this epoch
    #     train_loss = 0
    #     # Put the model into training mode
    #     classifier.train()
    #
    #     for step, batch in enumerate(train_dataloader):
    #         # Progress update every print_each_n_step batches
    #         if step % print_each_n_step == 0 and not step == 0:
    #             # Calculate elapsed time in minutes
    #             elapsed = format_time(time.time() - t0)
    #             # Report progress
    #             print("  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.".format(step, len(train_dataloader), elapsed))
    #
    #         # Unpack this training batch from our dataloader
    #         b_input_ids = batch[0].to(device)
    #         b_input_mask = batch[1].to(device)
    #         b_labels = batch[2].to(device)
    #
    #         # clear the gradients of all optimized variables
    #         optimizer.zero_grad()
    #         # forward pass: compute predicted outputs by passing inputs to the model
    #         train_logits, _ = classifier(b_input_ids, b_input_mask)
    #         # calculate the loss
    #         loss = nll_loss(train_logits, b_labels)
    #         # backward pass: compute gradient of the loss with respect to model parameters
    #         loss.backward()
    #         # perform a single optimization step (parameter update)
    #         optimizer.step()
    #         # update running training loss
    #         train_loss += loss.item()
    #
    #         # Update the learning rate with the scheduler, if specified
    #         if apply_scheduler:
    #             scheduler.step()
    #
    #     # Calculate the average loss over all of the batches
    #     avg_train_loss = train_loss / len(train_dataloader)
    #
    #     # Measure how long this epoch took
    #     training_time = format_time(time.time() - t0)
    #
    #     print("")
    #     print("  Average training loss: {0:.3f}".format(avg_train_loss))
    #     print("  Training epcoh took: {:}".format(training_time))
    #
    #     # ---------------------------------
    #     # Evaluation on the Development set
    #     # ---------------------------------
    #
    #     print("")
    #     print("Running Test...")
    #
    #     t0 = time.time()
    #
    #     # Put the model in evaluation mode (dropout layers behave differently during evaluation)
    #     classifier.eval()
    #
    #     # Apply the evaluate_method defined above to estimate
    #     avg_dev_loss, dev_accuracy = evaluate(
    #         val_dataloader,
    #         classifier,
    #         tokenizer,
    #         labels,
    #         label_map,
    #         device,
    #         nll_loss=nll_loss
    #     )
    #
    #     # Measure how long the validation run took
    #     test_time = format_time(time.time() - t0)
    #
    #     print("  Accuracy: {0:.3f}".format(dev_accuracy))
    #     print("  Test Loss: {0:.3f}".format(avg_dev_loss))
    #     print("  Test took: {:}".format(test_time))
    #
    #     # Record all statistics from this epoch
    #     training_stats.append(
    #         {
    #             "epoch": epoch_i + 1,
    #             "Training Loss": avg_train_loss,
    #             "Valid. Loss": avg_dev_loss,
    #             "Valid. Accur.": dev_accuracy,
    #             "Training Time": training_time,
    #             "Test Time": test_time
    #         }
    #     )
    #
    # train_losses = []
    # val_losses = []
    # train_acc = []
    # val_acc = []
    #
    # for stat in training_stats:
    #     train_losses.append(stat["Training Loss"])
    #     val_losses.append(stat["Valid. Loss"])
    #     val_acc.append(stat["Valid. Accur."])
    #     print(stat)
    #
    # plt.plot(range(1, num_train_epochs + 1), train_losses, label="Training Loss")
    # plt.plot(range(1, num_train_epochs + 1), val_losses, label="Validation Loss")
    # plt.title("Loss")
    # plt.xlabel("Epochs")
    # plt.ylabel("Loss")
    # plt.legend(loc="upper left")
    # plt.show()
    #
    # plt.plot(range(1, num_train_epochs + 1), val_acc, label="Val Accuracy")
    # plt.title("Accuracy")
    # plt.xlabel("Epochs")
    # plt.ylabel("Val. Accuracy")
    # plt.show()
    #
    # print("\nTraining complete!")
    # print("Total training took {:} (h:mm:ss)".format(format_time(time.time() - total_t0)))
