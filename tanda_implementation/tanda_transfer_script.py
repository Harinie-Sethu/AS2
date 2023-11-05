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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def parse_args(): 
    parser = argparse.ArgumentParser(description='Tanda Transfer Script')
    parser.add_argument('--model_name', type=str, default='roberta-base', help='model name')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--epochs', type=int, default=9, help='number of epochs')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='learning rate')
    parser.add_argument('--output_dir', type=str, default=f"/scratch/kapilrk04/roberta-base-transfer_model", help='output directory')
    args = parser.parse_args()
    return args

def main(args = None):
    asnq_dev = pd.read_csv("/home2/kapilrk04/anlp_proj/data_sets/asnq/dev.tsv", sep="\t", names=["sentence1", "sentence2", "label"])
    print("Loaded dev data")
    asnq_train = pd.read_csv("/home2/kapilrk04/anlp_proj/data_sets/asnq/train.tsv", sep="\t", names=["sentence1", "sentence2", "label"])
    print("Loaded train data")

    trainNeg = asnq_train[asnq_train['label']==3].sample(frac=0.25)
    trainNeg.loc[:,'label'] = 0
    trainPos = asnq_train[asnq_train['label']==4]
    trainPos.loc[:,'label'] = 1

    train_set = pd.concat([trainNeg, trainPos])
    train_set['idx'] = range(1, len(train_set) + 1)
    print("Took subsets of train data")

    devNeg = asnq_dev[asnq_dev['label']==3].sample(frac=0.25)
    devNeg.loc[:,'label'] = 0
    devPos = asnq_dev[asnq_dev['label']==4]
    devPos.loc[:,'label'] = 1

    dev_set = pd.concat([devNeg, devPos])
    dev_set['idx'] = range(1, len(dev_set) + 1)
    print("took subset of dev data")

    train_dataset = Dataset.from_pandas(train_set)
    dev_dataset = Dataset.from_pandas(dev_set)

    train_dataset = train_dataset.remove_columns('__index_level_0__')
    dev_dataset = dev_dataset.remove_columns('__index_level_0__')

    modelname = args.model_name
    tokenizer = AutoTokenizer.from_pretrained(modelname, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(modelname, num_labels=2)
    model.config.pad_token_id = model.config.eos_token_id

    encoded_train_dataset = train_dataset.map(lambda examples: preprocess_function(examples, tokenizer), batched=True)
    encoded_dev_dataset = dev_dataset.map(lambda examples: preprocess_function(examples, tokenizer), batched=True)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy = "epoch",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        include_inputs_for_metrics = True,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        fp16=False,
        report_to="wandb",
        run_name="gpt2_tanda"
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

    trainer.save_model(args.output_dir)

if __name__ == "__main__":
    args = parse_args()
    main(args)







