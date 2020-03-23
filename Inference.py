import torch
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from utils_ner import read_examples_from_file, convert_examples_to_features
import numpy as np
from transformers import (
    BertConfig,
    BertForTokenClassification,
    BertTokenizer,
)


def load_and_cache_examples(tokenizer, labels):

    examples = read_examples_from_file('./', mode= 'test')

    features = convert_examples_to_features(examples,
        labels,
        128,
        tokenizer,
        cls_token=tokenizer.cls_token,
        cls_token_segment_id=0,
        sep_token=tokenizer.sep_token,
        pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0]
    )

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    return dataset

def get_labels(path):
    if path:
        with open(path, "r") as f:
            labels = f.read().splitlines()
        if "O" not in labels:
            labels = ["O"] + labels
        return labels
    else:
        return ["O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]


def evaluate(model, tokenizer, labels=get_labels('./labels_.txt')):
    eval_dataset = load_and_cache_examples(tokenizer, labels)

    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=64)
    # Eval!
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    model.eval()
    for batch in eval_dataloader:
        batch = tuple(t.cpu() for t in batch)

        with torch.no_grad():
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]

            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs["labels"].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

    preds = np.argmax(preds, axis=2)

    label_map = {i: label for i, label in enumerate(labels)}

    out_label_list = [[] for _ in range(out_label_ids.shape[0])]
    preds_list = [[] for _ in range(out_label_ids.shape[0])]

    for i in range(out_label_ids.shape[0]):
        for j in range(out_label_ids.shape[1]):
            if out_label_ids[i, j] != -100:
                out_label_list[i].append(label_map[out_label_ids[i][j]])
                preds_list[i].append(label_map[preds[i][j]])

    return preds_list


def inference():
    tokenizer = BertTokenizer.from_pretrained('./Model/vocab.txt', do_lower_case=False, strip_accents=True, keep_accents=True, use_fast=True)
    config = BertConfig.from_pretrained('./Model',
                                          num_labels=4,
                                          id2label={str(i): label for i, label in enumerate(get_labels('./labels_.txt'))},
                                          label2id={label: i for i, label in enumerate(get_labels('./labels_.txt'))},
                                          cache_dir=None)
    model = BertForTokenClassification.from_pretrained('./Model', config = config)
    model.cpu()
    predictions = evaluate(model, tokenizer)

    return predictions
