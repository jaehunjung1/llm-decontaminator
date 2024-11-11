from argparse import Namespace
from pathlib import Path

from sentence_transformers import SentenceTransformer

from detect_instruct import datatype_to_instruct
from llm_detect import llm_detect, llm_detect_faster, set_openai_api_key
from vector_db import build_database
from show_samples import show


def parse_args():
    args = Namespace()

    args.train_path = "../targeted-diversification-internal-internal/math-reasoning/data/filtered/aqua_rat.test.1.problems.ngram-filtered.jsonl"
    args.test_path = "../targeted-diversification-internal/math-reasoning/data/preprocessed/aqua_rat.test.jsonl"
    args.output_path = Path("./data/database/math-reasoning/db-aqua_rat.test.1.problems.ngram-filtered.jsonl")

    # args.train_path = "../targeted-diversification-internal/sentence-nli/data/generated/seed.wanli-ood/20240928-merged.filtered.jsonl"
    # args.test_path = "Jaehun/data-diversity-nli-ood-test-v2"
    # args.output_path = Path("./data/database/db-20240928-merged.filtered.jsonl")

    args.bert_model = "multi-qa-MiniLM-L6-cos-v1"
    args.top_k = 1
    args.batch_size = 32
    args.device = "cuda:0"

    # args.model = "gpt-4o-mini-2024-07-18"
    args.model = "meta-llama/Meta-Llama-3.1-70B-Instruct"
    args.data_type = 'math'
    args.max_workers = 5

    return args


if __name__ == "__main__":
    args = parse_args()

    bert_model = SentenceTransformer(args.bert_model)
    database = build_database(bert_model, args.train_path, args.test_path, args.output_path, args.data_type, args.top_k, args.batch_size, args.device)

    instruct = datatype_to_instruct(args.data_type)
    print("Starting LLM detection...")
    database = llm_detect_faster(args.model, database, args.output_path, instruct, max_workers=args.max_workers)

    show(database)

    