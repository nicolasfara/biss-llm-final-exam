import datetime
import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset, RandomSampler, SequentialSampler
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def format_time(elapsed):
    """
    Takes a time in seconds and returns a string hh:mm:ss
    """
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def setup_torch():
    # Tell PyTorch to use the GPU, if available
    _device = None
    if torch.cuda.is_available():
        _device = torch.device("cuda")
        print("There are %d GPU(s) available." % torch.cuda.device_count())
        print("We will use the GPU:", torch.cuda.get_device_name(0))
    # Use the Mac's MPS, if available
    elif torch.backends.mps.is_available():
        _device = torch.device("mps")
        print("Using the MPS device for Mac")
    # Use the GPU otherwise
    else:
        print("No GPU available, using the CPU instead.")
        _device = torch.device("cpu")

    return _device


def retrieve_data_as_dataframe(url):
    return pd.read_csv(url)


def generate_dataloader(dataset, tokenizer, max_seq_length, batch_size, max_sequence_len=None):
    max_sequence_len = tokenizer.model_max_length if max_sequence_len is None else max_sequence_len

    def collator(sequences):
        texts = [sequence['comment_text'] for sequence in sequences]
        # Get all labels from sequences list.
        labels = [int(sequence['conspiratorial']) for sequence in sequences]
        # Call tokenizer on all texts to convert into tensors of numbers with
        # appropriate padding.
        inputs = tokenizer(text=texts, return_tensors="pt", padding=True, truncation=True, max_length=max_sequence_len)
        # Update the inputs with the associated encoded labels as tensor.
        inputs.update({'labels': torch.tensor(labels)})

        return inputs

    return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collator)


def generate_data_loader(examples, label_map, tokenizer, batch_size, max_seq_length, do_shuffle=False):
    """
    Generate a Dataloader given the input examples.

    `examples`: a list of pairs (input_text, label)
    `label_mal`: a dictionary used to assign an ID to each label
    `tokenize`: the tokenizer used to convert input sentences into word pieces
    `do_shuffle`: a boolean parameter to shuffle input examples (usefull in training)
    """

    # Generate input examples to the Transformer
    input_ids = []
    input_mask_array = []
    label_id_array = []

    # Tokenization
    for (_, row) in examples.iterrows():
        (text, label) = (row["comment_text"], row["conspiratorial"])
        # tokenizer.encode_plus is a crucial method which:
        # 1. Tokenizes examples
        # 2. Trims sequences to a max_seq_length
        # 3. Applies a pad to shorter sequences
        # 4. Assigns the [CLS] special word-piece such as the other ones (e.g., [SEP])
        encoded_sent = tokenizer.encode_plus(
            text, add_special_tokens=True, max_length=max_seq_length, padding="max_length", truncation=True
        )
        # Convert input word pieces to IDs of the corresponding input embeddings
        input_ids.append(encoded_sent["input_ids"])
        # Store the attention mask to avoid computations over "padded" elements
        input_mask_array.append(encoded_sent["attention_mask"])

        # Converts labels to IDs
        _id = -1
        if label in label_map:
            _id = label_map[label]
        label_id_array.append(_id)

    # Convert to Tensor which are used in PyTorch
    input_ids = torch.tensor(input_ids)
    input_mask_array = torch.tensor(input_mask_array)
    label_id_array = torch.tensor(label_id_array, dtype=torch.long)

    # Building the TensorDataset
    dataset = TensorDataset(input_ids, input_mask_array, label_id_array)

    if do_shuffle:
        # This will shuffle examples each time a new batch is required
        sampler = RandomSampler
    else:
        sampler = SequentialSampler

    # Building the DataLoader
    return DataLoader(
        dataset,  # The training samples
        sampler=sampler(dataset),  # the adopted sampler
        batch_size=batch_size  # Train with this batch size
    )

def evaluate(
        dataloader,
        classifier,
        tokenizer,
        labels,
        id_to_label_map,
        device,
        nll_loss,
        print_classification_output=False,
        print_result_summary=False
):
    """
    Evaluation method which will be applied to development and test datasets.
    It returns the pair (average loss, accuracy)

    dataloader: a dataloader containing examples to be classified
    classifier: the BERT-based classifier
    print_classification_output: to log the classification outcomes
    """
    total_loss = 0
    gold_classes = []
    system_classes = []

    if print_classification_output:
        print("\n------------------------")
        print("  Classification outcomes")
        print("is_correct\tgold_label\tsystem_label\ttext")
        print("------------------------")

    # For each batch of examples from the input dataloader
    for batch in dataloader:
        # Unpack this training batch from our dataloader. Notice this is populated
        # in the method `generate_data_loader`nll_loss
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        # Tell pytorch not to bother with constructing the compute graph during
        # the forward pass, since this is only needed for backprop (training).
        with torch.no_grad():
            # Each batch is classifed
            logits, _ = classifier(b_input_ids, b_input_mask)
            # Evaluate the loss.
            total_loss += nll_loss(logits, b_labels)

        # Accumulate the predictions and the input labels
        _, preds = torch.max(logits, 1)
        system_classes += preds.detach().cpu()
        gold_classes += b_labels.detach().cpu()

        # Print the output of the classification for each input element
        if print_classification_output:
            for ex_id in range(len(b_input_mask)):
                input_strings = tokenizer.decode(b_input_ids[ex_id], skip_special_tokens=True)
                # convert class id to the real label
                predicted_label = id_to_label_map[preds[ex_id].item()]
                gold_standard_label = "UNKNOWN"
                # convert the gold standard class ID into a real label
                if b_labels[ex_id].item() in id_to_label_map:
                    gold_standard_label = id_to_label_map[b_labels[ex_id].item()]
                # put the prefix "[OK]" if the classification is correct
                output = "[OK]" if predicted_label == gold_standard_label else "[NO]"
                # print the output
                print(f"{output}\t{gold_standard_label}\t{predicted_label}\t{input_strings}")

    # Calculate the average loss over all of the batches.
    avg_loss = total_loss / len(dataloader)
    avg_loss = avg_loss.item()

    # Report the final accuracy for this validation run.
    system_classes = torch.stack(system_classes).numpy()
    gold_classes = torch.stack(gold_classes).numpy()
    accuracy = np.sum(system_classes == gold_classes) / len(system_classes)

    if print_result_summary:
        print("\n------------------------")
        print("  Summary")
        print("------------------------")
        #remove unused classes in the test material
        filtered_label_list = []
        for i in range(len(labels)):
            if i in gold_classes:
                filtered_label_list.append(id_to_label_map[i])
        print(classification_report(gold_classes, system_classes)) #target_names=filtered_label_list))

        print("\n------------------------")
        print("  Confusion Matrix")
        print("------------------------")
        conf_mat = confusion_matrix(gold_classes, system_classes)
        for row_id in range(len(conf_mat)):
            print(f"{filtered_label_list[row_id]}\t{conf_mat[row_id]}")

    return avg_loss, accuracy
