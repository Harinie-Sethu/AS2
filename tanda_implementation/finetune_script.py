import torch
import os
from huggingface_hub import login
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import OneHotEncoder
from datasets import Dataset, load_dataset
import argparse
import warnings
import wandb

warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "albert-base-v2"

login(token="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")

def split_array_by_number(arr, number):
    result = []
    current_split = []
    
    for item in arr:
        if item == number:
            if current_split:
                result.append(current_split)
                return current_split
        else:
            current_split.append(item)
    if current_split:
        result.append(current_split)
    
    return result

def compute_metrics(eval_pred):
    predictions, labels, inputs = eval_pred
    
    splitnum = 0
    if model_name == "roberta-base":
        splitnum = 2
    elif model_name == "bert-base-uncased":
        splitnum = 102
    elif model_name == "albert-base-v2":
        splitnum = 3
    elif model_name == "distilbert-base-uncased":
        splitnum = 102

    per_qn_inputs = {}

    for i in range(len(inputs)):
        split_inputs = split_array_by_number(inputs[i], splitnum)
        qn = tuple(split_inputs)
        if qn not in per_qn_inputs:
            per_qn_inputs[qn] = {}
            per_qn_inputs[qn]["predictions"] = []
            per_qn_inputs[qn]["labels"] = []
        per_qn_inputs[qn]["predictions"].append(predictions[i])
        per_qn_inputs[qn]["labels"].append(labels[i])

    avg_prec_scores = []
    enc = OneHotEncoder(sparse=False)
    labels = enc.fit_transform(np.array(labels).reshape(-1,1))

    reciprocal_ranks = []

    for qn in per_qn_inputs:
        if per_qn_inputs[qn]["labels"].count(1) == 0 or per_qn_inputs[qn]["labels"].count(0) == 0:
            continue
        per_qn_inputs[qn]['predictions'] = np.array(per_qn_inputs[qn]['predictions'])
        per_qn_inputs[qn]['labels'] = enc.fit_transform(np.array(per_qn_inputs[qn]['labels']).reshape(-1,1))
        
        #print(per_qn_inputs[qn]['predictions'], per_qn_inputs[qn]['labels'])
        avg_prec_scores.append(average_precision_score(per_qn_inputs[qn]["labels"], per_qn_inputs[qn]["predictions"]))

        true_label = per_qn_inputs[qn]["labels"]
        pred_label = per_qn_inputs[qn]["predictions"]

        sorted_pred_label = np.argsort(pred_label)[::-1]

        for j in range(len(sorted_pred_label)):
            row = sorted_pred_label[j]
            rank = np.where(row == 1)[0]
            if rank.size > 0:
                reciprocal_ranks.append(1/(rank[0]+1))
                break
    
    
    map_score = np.mean(avg_prec_scores)
    mrr_score = np.mean(reciprocal_ranks)

    return {
        "mAP" : map_score,
        "mRR" : mrr_score
    }

def preprocess_function(examples, tokenizer):
    return tokenizer(examples["sentence1"], examples["sentence2"], truncation=True)

def main(model_name, epochs, transfer_learning_rate=2e-5, adapt_learning_rate=1e-6):
    asnq_dev = pd.read_csv("/home2/kapilrk04/anlp_proj/data_sets/asnq/dev.tsv", sep="\t", names=["sentence1", "sentence2", "label"])
    print("Loaded dev data")
    asnq_train = pd.read_csv("/home2/kapilrk04/anlp_proj/data_sets/asnq/train.tsv", sep="\t", names=["sentence1", "sentence2", "label"])
    print("Loaded train data")

    trainNeg = asnq_train[asnq_train['label']==3].sample(frac=0.025)
    trainNeg.loc[:,'label'] = 0
    trainPos = asnq_train[asnq_train['label']==4].sample(frac=0.1)
    trainPos.loc[:,'label'] = 1

    train_set = pd.concat([trainNeg, trainPos])
    train_set['idx'] = range(1, len(train_set) + 1)
    print("Took subsets of train data")

    devNeg = asnq_dev[asnq_dev['label']==3].sample(frac=0.025)
    devNeg.loc[:,'label'] = 0
    devPos = asnq_dev[asnq_dev['label']==4].sample(frac=0.1)
    devPos.loc[:,'label'] = 1

    dev_set = pd.concat([devNeg, devPos])
    dev_set['idx'] = range(1, len(dev_set) + 1)
    print("took subset of dev data")

    train_dataset = Dataset.from_pandas(train_set)
    dev_dataset = Dataset.from_pandas(dev_set)

    train_dataset = train_dataset.remove_columns('__index_level_0__')
    dev_dataset = dev_dataset.remove_columns('__index_level_0__')

    wiki_qa_dataset = load_dataset("wiki_qa")

    wiki_qa_set = {
        "train" : {},
        "validation" : {},
        "test" : {}
    }

    for split in ["train", "validation", "test"]:
        for example in wiki_qa_dataset[split]:
            if example["question_id"] not in wiki_qa_set[split]:
                wiki_qa_set[split][example["question_id"]] = {
                    "question" : example["question"],
                    "answers" : [],
                    "labels" : [],
                    "sum_labels" : 0
                }
            wiki_qa_set[split][example["question_id"]]["answers"].append(example["answer"])
            wiki_qa_set[split][example["question_id"]]["labels"].append(example["label"])
            wiki_qa_set[split][example["question_id"]]["sum_labels"] += example["label"]

    wiki_qa_trainp = [{"sentence1" : wiki_qa_set["train"][qn]["question"], "sentence2" : wiki_qa_set["train"][qn]["answers"][i], "label" : wiki_qa_set["train"][qn]["labels"][i]} for qn in wiki_qa_set["train"] for i in range(len(wiki_qa_set["train"][qn]["answers"])) if wiki_qa_set["train"][qn]["sum_labels"] > 0 and wiki_qa_set["train"][qn]["sum_labels"] < len(wiki_qa_set["train"][qn]["labels"])]
    wiki_qa_validationp = [{"sentence1" : wiki_qa_set["validation"][qn]["question"], "sentence2" : wiki_qa_set["validation"][qn]["answers"][i], "label" : wiki_qa_set["validation"][qn]["labels"][i]} for qn in wiki_qa_set["validation"] for i in range(len(wiki_qa_set["validation"][qn]["answers"])) if wiki_qa_set["validation"][qn]["sum_labels"] > 0 and wiki_qa_set["validation"][qn]["sum_labels"] < len(wiki_qa_set["validation"][qn]["labels"])]
    wiki_qa_testp = [{"sentence1" : wiki_qa_set["test"][qn]["question"], "sentence2" : wiki_qa_set["test"][qn]["answers"][i], "label" : wiki_qa_set["test"][qn]["labels"][i]} for qn in wiki_qa_set["test"] for i in range(len(wiki_qa_set["test"][qn]["answers"])) if wiki_qa_set["test"][qn]["sum_labels"] > 0 and wiki_qa_set["test"][qn]["sum_labels"] < len(wiki_qa_set["test"][qn]["labels"])]

    wiki_qa_trainp = pd.DataFrame(wiki_qa_trainp)
    wiki_qa_validationp = pd.DataFrame(wiki_qa_validationp)
    wiki_qa_testp = pd.DataFrame(wiki_qa_testp)

    wiki_qa_trainp['idx'] = range(1, len(wiki_qa_trainp) + 1)
    wiki_qa_validationp['idx'] = range(1, len(wiki_qa_validationp) + 1)
    wiki_qa_testp['idx'] = range(1, len(wiki_qa_testp) + 1)

    wiki_train_ds = Dataset.from_pandas(wiki_qa_trainp)
    wiki_test_ds = Dataset.from_pandas(wiki_qa_testp)
    wiki_valid_ds = Dataset.from_pandas(wiki_qa_validationp)

    run_name = f"{model_name}_transfer_(learning_rate={transfer_learning_rate})_(adapt_learning_rate={adapt_learning_rate})"
    with wandb.init(project="anlp_proj_paramtuning_outputs", entity="kapilrk-04", name=run_name):
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

        encoded_train_dataset = train_dataset.map(lambda x: preprocess_function(x, tokenizer), batched=True)
        encoded_dev_dataset = dev_dataset.map(lambda x: preprocess_function(x, tokenizer), batched=True)

        encoded_wikiqa_train_dataset = wiki_train_ds.map(lambda x: preprocess_function(x, tokenizer), batched=True)
        encoded_wikiqa_test_dataset = wiki_test_ds.map(lambda x: preprocess_function(x, tokenizer), batched=True)
        encoded_wikiqa_dev_dataset = wiki_valid_ds.map(lambda x: preprocess_function(x, tokenizer), batched=True)

        training_args = TrainingArguments(
            output_dir=f"/scratch/kapilrk04/{model_name}_transfer_(epochs={epochs})",
            evaluation_strategy = "epoch",
            save_strategy="epoch",
            learning_rate=transfer_learning_rate,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            include_inputs_for_metrics = True,
            num_train_epochs=epochs,
            weight_decay=0.01,
            fp16=False,
            report_to="wandb",
            run_name=run_name
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=encoded_train_dataset,
            eval_dataset=encoded_dev_dataset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics
        )

        trainer.train()

        trainer.save_model(f"/scratch/kapilrk04/{model_name}_transfer_(epochs={epochs})")

        model = AutoModelForSequenceClassification.from_pretrained(f"/scratch/kapilrk04/{model_name}_transfer_(epochs={epochs})")

        training_args = TrainingArguments(
            output_dir=f"/scratch/kapilrk04/{model_name}_adapt_(epochs={epochs})_wiki",
            evaluation_strategy = "epoch",
            save_strategy="epoch",
            learning_rate=adapt_learning_rate,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            include_inputs_for_metrics = True,
            num_train_epochs=3,
            weight_decay=0.01,
            fp16=False,
            report_to="wandb",
            run_name=run_name
        )

        trainer1 = Trainer(
            model=model,
            args=training_args,
            train_dataset=encoded_wikiqa_train_dataset,
            eval_dataset=encoded_wikiqa_dev_dataset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics
        )

        trainer2 = Trainer(
            model=model,
            args=training_args,
            train_dataset=encoded_wikiqa_train_dataset,
            eval_dataset=encoded_wikiqa_test_dataset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics
        )

        print("Scores without Adapt")
        trainer2.evaluate()

        print("Scores with Adapt")
        trainer1.train()
        
        print("Output after Adapt")
        trainer2.evaluate()


if __name__ == "__main__":
    for tlr in [2e-5, 5e-5, 1e-6, 2e-6, 5e-6]:
        for alr in [1e-6, 2e-6, 5e-6]:
            print("Transfer Learning Rate:", tlr, "\tAdapt Learning Rate:", alr)
            main(model_name, 3, tlr, alr)








