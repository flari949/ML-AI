import os

import nltk
import numpy as np
import tensorflow as tf
from datasets import load_dataset, DatasetDict, load_metric
from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, AdamWeightDecay, pipeline, \
    create_optimizer
from transformers.keras_callbacks import KerasMetricCallback

# Create 'saved' folder if it doesn't exist
if not os.path.isdir("saved"):
    os.mkdir('saved')

# Specify the names of the save files
save_name = os.path.join('saved', 'task2_')
net_save_name = save_name + 'summarizer_weights.h5'

# Import dataset splits with English and German parallel key value pairs
# Only train split used as contains significant amount of data
data = load_dataset("GEM/wiki_lingua", "en_de", split="train")

# Filter out alternate text translations
data = data.filter(lambda doc: doc['source_language'] == 'en' and doc['target_language'] == 'de')

# Remove irrelevant dataset columns, formatting mono-directional, cross-lingual dataset with refined sub dictionaries
dataset = data.remove_columns(["gem_id", "gem_parent_id", "source_language", "target_language", "references"])

# Split dataset into training, testing, and validation sets
train_test_data = dataset.train_test_split(test_size=0.1)
test_valid_data = train_test_data['test'].train_test_split(test_size=0.5)
dataset = DatasetDict({
    'train': train_test_data['train'],
    'test': test_valid_data['test'],
    'valid': test_valid_data['train']
})

print(dataset)
# Import rouge metric
metric = load_metric("rouge")
# Define base model
model_checkpoint = "Helsinki-NLP/opus-mt-en-de"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

prefix = ""


# Preprocess dataset value key pairs
def preprocess_function(examples):
    inputs = [prefix + doc for doc in examples["source"]]
    model_inputs = tokenizer(inputs, max_length=1024, truncation=True)

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            text_target=examples["target"], max_length=128, truncation=True
        )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


# Apply preprocessing to entire dataset
tokenized_ds = dataset.map(preprocess_function, batched=True)


# Function to compute rouge metrics
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    for label in labels:
        label[label < 0] = tokenizer.pad_token_id
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_preds = [
        "\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds
    ]
    decoded_labels = [
        "\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels
    ]

    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

    prediction_lens = [
        np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions
    ]
    result["gen_len"] = np.mean(prediction_lens)
    return result


# Import model weights
model = TFAutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
# Set data formatters
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, return_tensors="np")
generation_data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, return_tensors="np", pad_to_multiple_of=128)

# Format dataset splits to TensorFlow format
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

# Define optimization function
optimizer = AdamWeightDecay(learning_rate=2e-5, weight_decay_rate=0.01)
# Compile model
model.compile(optimizer=optimizer)
# Define a model callback function
metric_callback = KerasMetricCallback(
    compute_metrics, eval_dataset=generation_set, predict_with_generate=True, use_xla_generation=True
)

train = False
if train:
    # Train model & save model weights
    model.fit(train_set, validation_data=validation_set, epochs=5, callbacks=[metric_callback])
    model.save_weights(net_save_name)

else:
    # Load saved model weights
    model.load_weights(net_save_name)
    outputs = []
    # Test model on first 100 testing dataset data pairs, values returned /100
    for doc in dataset['test']['source'][:100]:
        tokenized = tokenizer([doc], return_tensors='np', max_length=512)
        out = model.generate(**tokenized, max_length=128)
        with tokenizer.as_target_tokenizer():
            value = tokenizer.decode(out[0])
        outputs.append(value)

    result = metric.compute(predictions=outputs, references=dataset['test']['target'][:100],
                            rouge_types=["rouge1", "rouge2", "rougeL"])
    print("Rouge1 ", result["rouge1"].mid)
    print("Rouge2 ", result["rouge2"].mid)
    print("RougeL ", result["rougeL"].mid)

# Testing existing models, values returned /100
test_existing_model = False
if test_existing_model:
    trans_model = "Helsinki-NLP/opus-mt-en-de"
    sum_model = "google/mt5-base"
    translator = pipeline("translation", trans_model, framework="tf")
    summarizer = pipeline("summarization", sum_model, framework="tf")

    outputs = []

    for doc in dataset['test']['source'][:100]:
        summary = summarizer(doc, max_length=128)
        summary = summary[0]['summary_text']
        outputs.append(translator(summary, max_length=128))

    print("Analysing results :")
    result = metric.compute(predictions=outputs, references=dataset['test']['target'][:100],
                            rouge_types=["rouge1", "rouge2", "rougeL"])
    print("Rouge1 ", result["rouge1"].mid)
    print("Rouge2 ", result["rouge2"].mid)
    print("RougeL ", result["rougeL"].mid)
