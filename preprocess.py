# from argparse import Namespace
# from pathlib import Path
# from typing import Dict
#
# import ipdb
# import jsonlines
# from tqdm import tqdm
#
#
# def parse_args():
#     args = Namespace()
#
#     args.input_filename = Path(
#         "../targeted-diversification-internal-internal/sentence-nli/data/filtered/bigbench.test.Llama-3.1-70B-generated.1.filtered.jsonl"
#     )
#     args.out_filename = Path("./data/train") / args.input_filename.name
#     ipdb.set_trace()  # todo check args.out_filename
#
#     # args.input_filename = Path(
#     #     "../targeted-diversification-internal-internal/sentence-nli/data/preprocessed/bigbench.test.preprocessed.jsonl"
#     # )
#     # args.out_filename = Path("./data/test") / args.input_filename.name
#
#     args.task_type = "nli"
#
#     return args
#
#
# def format_nli_sample_to_text(sample: Dict) -> str:
#
#
#
# if __name__ == "__main__":
#     args = parse_args()
#
#     with jsonlines.open(args.input_filename) as f:
#         samples = list(f)
#
#     out_samples = []
#     for sample in tqdm(out_samples):
#
