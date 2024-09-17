from argparse import Namespace
from pathlib import Path

from sentence_transformers import SentenceTransformer

from detect_instruct import datatype_to_instruct
from llm_detect import llm_detect, llm_detect_faster, set_openai_api_key
from vector_db import build_database
from show_samples import show


def parse_args():
    args = Namespace()

    args.train_path = "../targeted-diversification-internal-internal/sentence-nli/data/filtered/bigbench.test.Llama-3.1-70B-generated.1.filtered.jsonl"
    args.test_path = "../targeted-diversification-internal-internal/sentence-nli/data/preprocessed/bigbench.test.preprocessed.jsonl"
    args.output_path = Path("./data/database/db-bigbench.test.Llama-3.1-70B-generated.1.filtered.jsonl")

    assert not args.output_path.exists(), f"{args.output_path} already exists."

    args.bert_model = "multi-qa-MiniLM-L6-cos-v1"
    args.top_k = 1
    args.batch_size = 32
    args.device = "cuda:0"

    args.model = "gpt-4o-mini-2024-07-18"
    args.data_type = 'nli'
    args.max_workers = 3

    return args


if __name__ == "__main__":
    args = parse_args()

    set_openai_api_key()

    bert_model = SentenceTransformer(args.bert_model)
    database = build_database(bert_model, args.train_path, args.test_path, args.output_path, args.data_type, args.top_k, args.batch_size, args.device)

    instruct = datatype_to_instruct(args.data_type)
    print("Starting LLM detection...")
    database = llm_detect_faster(args.model, database, args.output_path, instruct, max_workers=args.max_workers)

    show(database)

    