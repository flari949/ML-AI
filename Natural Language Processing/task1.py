import os

import nltk
import numpy as np
from datasets import load_dataset, Dataset, DatasetDict, load_metric
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, TFAutoModelForSeq2SeqLM, AdamWeightDecay, \
    create_optimizer
from transformers.keras_callbacks import KerasMetricCallback

# Create 'saved' folder if it doesn't exist
if not os.path.isdir("saved"):
    os.mkdir('saved')

# Specify the names of the save files
save_name = os.path.join('saved', 'task1_')
net_save_name = save_name + 'summarizer_weights.h5'

# Import WikiLingua dataset
ds = load_dataset("wiki_lingua", "english", split="train")

# Load the ROUGE metric for model evaluate
metric = load_metric("rouge")

print(ds.features)

# Filter out empty, non-valid data
ds = ds.filter(lambda doc: 'article' in doc
                           and doc['article']['document']
                           and doc['article']['summary'])

# Format data for dataset
train_documents = []
train_summaries = []
for item in ds["article"]:
    train_documents.append(item["document"][0])
    train_summaries.append(item["summary"][0])

# Create new dataset with reformatted data
ds = Dataset.from_dict({"text": train_documents, "summary": train_summaries})

# Split dataset into training (0.9), testing (0.05), and validation (0.05) sets
train_test_data = ds.train_test_split(test_size=0.1)
test_valid_data = train_test_data['test'].train_test_split(test_size=0.5)
ds = DatasetDict({
    'train': train_test_data['train'],
    'test': test_valid_data['test'],
    'valid': test_valid_data['train']
})

print(ds['test'].features)

# Define base model
checkpoint = "t5-small"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

prefix = "summarize: "


''' Function to preprocess data:
        - Prefix input with prompt
        - Tokenize text and summary into simpler grammatical units
        - Truncate sequences to have a maximum length
'''
def preprocess_function(examples):
    inputs = [prefix + doc for doc in examples["text"]]
    model_inputs = tokenizer(inputs, max_length=1024, truncation=True)

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            text_target=examples["summary"], max_length=128, truncation=True
        )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


# Apply preprocessing to entire dataset
tokenized_ds = ds.map(preprocess_function, batched=True)

# Define model
model = TFAutoModelForSeq2SeqLM.from_pretrained(checkpoint)

# Data collator objects to dynamically batch data
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, return_tensors="np")
generation_data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, return_tensors="np", pad_to_multiple_of=128)

# Format datasets to TensorFlow format
train_set = model.prepare_tf_dataset(
    tokenized_ds["train"],
    batch_size=8,
    shuffle=True,
    collate_fn=data_collator
)
validation_set = model.prepare_tf_dataset(
    tokenized_ds["valid"],
    batch_size=8,
    shuffle=False,
    collate_fn=data_collator
)
generation_set = model.prepare_tf_dataset(
    tokenized_ds["test"],
    batch_size=8,
    shuffle=False,
    collate_fn=generation_data_collator
)


# Function to compute rouge metrics
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    for label in labels:
        label[label < 0] = tokenizer.pad_token_id
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_predictions = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_predictions]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

    result = metric.compute(predictions=decoded_predictions, references=decoded_labels, use_stemmer=True)
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)
    return result


# Define optimizer
optimizer = AdamWeightDecay(learning_rate=2e-5, weight_decay_rate=0.01)

# Compile model
model.compile(optimizer=optimizer)

# Compute model metrics
metric_callback = KerasMetricCallback(
    compute_metrics, eval_dataset=generation_set, predict_with_generate=True, use_xla_generation=True
)

train = False
if train:
    # Train model & save model weights
    model.fit(train_set, validation_data=validation_set, epochs=3, callbacks=[metric_callback])
    model.save_weights(net_save_name)

else:
    model.load_weights(net_save_name)
    outputs = []

    # Testing - Values returned /100
    for doc in ds['test']['text'][:100]:
        doc = "summarize: " + doc
        tokenized = tokenizer([doc], return_tensors='np')
        out = model.generate(**tokenized, max_length=128)
        with tokenizer.as_target_tokenizer():
            value = tokenizer.decode(out[0])
        outputs.append(value)

    result = metric.compute(predictions=outputs, references=ds['test']['summary'][:100],
                            rouge_types=["rouge1", "rouge2", "rougeL"])
    print("Rouge-1 ", result["rouge1"].mid)
    print("Rouge-2 ", result["rouge2"].mid)
    print("Rouge-L ", result["rougeL"].mid)
