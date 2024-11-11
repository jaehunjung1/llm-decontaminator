import argparse
import json
from pathlib import Path

import ipdb
import jsonlines
import torch
import os
import random

import openai
import pandas as pd
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util


def read_dataset(r_path, data_type):
    print(f"Reading {r_path}...")

    if data_type == "nli":
        if r_path in ["Jaehun/data-diversity-nli-ood-test", "Jaehun/data-diversity-nli-ood-test-v2"]:
            dataset = list(load_dataset(r_path)["test"])
        else:
            with jsonlines.open(r_path) as f:
                dataset = list(f)

        for sample in dataset:
            sample["text"] = f"""Premise: {sample["premise"]}\nHypothesis: {sample["hypothesis"]}\nLabel: {sample["label"]}"""

    elif data_type == "math":
        with jsonlines.open(r_path) as f:
            dataset = list(f)

        for sample in dataset:
            sample["text"] = sample["problem"]

            if "id" not in sample and "problem_id" in sample:
                sample["id"] = sample["problem_id"]

    else:
        raise NotImplementedError

    print("Done.")
    return dataset


def bert_encode(model, data, batch_size=32, device=None):
    return model.encode(data, batch_size=batch_size, show_progress_bar=True, device=device)


def top_k_similarity(train_embs, test_embs, top_k):
    # # Compute cosine-similarities
    # cosine_scores = util.cos_sim(test_embs, train_embs)
    # # Find the top-k most similar train_embs for each test_emb
    # top_k_indices = torch.topk(cosine_scores, k=top_k, dim=1).indices

    # Compute cosine-sim
    cosine_scores = util.cos_sim(train_embs, test_embs)

    # find the top-k most similar test_embs for each train_emb
    top_k_indices = torch.topk(cosine_scores, k=top_k, dim=1).indices

    return top_k_indices


def build_database(model, train_path, test_path, output_path, data_type, top_k=1, batch_size=32, device=None):
    if Path(output_path).exists():
        print(f"{output_path} already exists, loading database...")
        with jsonlines.open(output_path) as f:
            db = list(f)

    else:
        print(f"Creating new database in {output_path}...")

        train_cases = read_dataset(train_path, data_type)
        test_cases = read_dataset(test_path, data_type)
        train_embs = bert_encode(model, [sample["text"] for sample in train_cases], batch_size=batch_size, device=device)
        test_embs = bert_encode(model, [sample["text"] for sample in test_cases], batch_size=batch_size, device=device)
        top_k_indices = top_k_similarity(train_embs, test_embs, top_k)

        db = []

        for i, train_case in enumerate(train_cases):
            top_k_cases = [test_cases[index] for index in top_k_indices[i]]
            db.append({
                "train": train_case['text'],
                "test": [sample['text'] for sample in top_k_cases],
                "train_id": train_case['id'],
                "test_ids": [sample['id'] for sample in top_k_cases],
            })

        with open(output_path, "w") as f:
            for each in db:
                f.write(json.dumps(each) + "\n")

    return db


def parse_args():
    args = argparse.Namespace()

    args.train_path = "../targeted-diversification-internal-internal/math-reasoning/data/filtered/aqua_rat.test.1.problems.ngram-filtered.jsonl"
    args.test_path = "../targeted-diversification-internal/math-reasoning/data/preprocessed/aqua_rat.test.jsonl"
    args.output_path = Path("./data/database/math-reasoning/db-aqua_rat.test.1.problems.ngram-filtered.jsonl")

    # args.train_path = "../targeted-diversification-internal/sentence-nli/data/generated/seed.wanli-ood/20240928-merged.filtered.jsonl"
    # args.test_path = "Jaehun/data-diversity-nli-ood-test-v2"
    # args.output_path = Path("./data/database/db-20240928-merged.filtered.jsonl")

    assert not args.output_path.exists(), f"{args.output_path} already exists."

    args.bert_model = "multi-qa-MiniLM-L6-cos-v1"
    args.top_k = 1
    args.batch_size = 32
    args.device = "cuda:0"

    args.data_type = 'math'

    return args


if __name__ == "__main__":
    args = parse_args()

    model = SentenceTransformer(args.bert_model)
    build_database(model, args.train_path, args.test_path, args.output_path, args.data_type, args.top_k, args.batch_size, args.device)
