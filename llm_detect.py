import argparse
import concurrent.futures
import json
import os
import time
from pathlib import Path

import ipdb
import openai
from openai import OpenAI
from tqdm import tqdm

from detect_instruct import datatype_to_instruct


def set_openai_api_key():
    if not (api_key := os.getenv("OPENAI_API_KEY")):
        api_key = open(f"/gscratch/xlab/jaehunj/OPENAI_API_KEYS/xlab.distillation", "r").read().strip()
        os.environ["OPENAI_API_KEY"] = api_key

    openai.api_key = api_key
    return api_key


def set_avior_api_key():
    if not (api_key := os.getenv("AVIOR_API_KEY")):
        if Path(f"/home/lustre/api_keys/AVIOR_API_KEY/test").exists():
            directory = Path(f"/home/lustre/api_keys/AVIOR_API_KEY/test")

        elif Path(f"/lustre/fsw/portfolios/llmservice/users/jaehunj/api_keys/AVIOR_API_KEY/test").exists():
            directory = Path("/lustre/fs1/portfolios/llmservice/users/jaehunj/api_keys/AVIOR_API_KEY/test")

        elif Path(f"/gscratch/xlab/jaehunj/AVIOR_API_KEY/test").exists():
            directory = Path(f"/gscratch/xlab/jaehunj/AVIOR_API_KEY/test")

        else:
            raise FileNotFoundError

        api_key = open(directory, "r").read().strip()
        os.environ["AVIOR_API_KEY"] = api_key

    return api_key


def detect_contamination(client, model, question1, question2, instruct):
    retries = 0
    while retries < 30:
        try:
            prompt = "part1: \{\n" + question1 + "\n\}\npart2: \{\n" + question2 + "\n\}"

            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": instruct},
                    {"role": "user", "content": prompt}
                ],
                timeout=3,
                temperature=0.3,
            )

            pred = completion.choices[0].message.content

            if pred == "True":
                return True
            elif pred == "False":
                return False

            raise Exception("Invalid prediction: {}".format(pred))
        except Exception as e:
            print(f"Retrying...{e}")
            
        time.sleep(1)
        retries += 1

    print(f"Failed to get prediction after {retries} retries.")
    return False


def llm_detect(model, database, output_path, instruct, max_workers=32):
    results = []
    futures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for i, pairs in enumerate(database):
            test_case = pairs["train"]
            case_results = []
            for train_case in pairs["test"]:
                future = executor.submit(detect_contamination, model, test_case, train_case, instruct)
                case_results.append(future)
            futures.append(case_results)

        for case_results in futures:
            results.append([future.result() for future in case_results])

    for i in range(len(database)):
        database[i]["results"] = results[i]

    with open(output_path, "w") as fout:
        for each in database:
            fout.write(json.dumps(each) + "\n")

    return database


def llm_detect_faster(model, database, output_path, instruct, max_workers=32):
    if "Llama" in model:
        client = openai.Client(
            base_url="http://avior.mlfoundry.com/live-inference/v1", api_key=set_avior_api_key()
        )
    else:
        client = openai.Client(api_key=set_openai_api_key())

    with tqdm(total=len(database)) as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(
                detect_contamination, client, model, pair["test"][0], pair["train"], instruct
            ): pair for pair in database}

            results = []
            for future in concurrent.futures.as_completed(futures):
                pair = futures[future]
                pair["results"] = [future.result()]
                results.append(pair)
                pbar.update(1)

    with open(output_path, "w") as f:
        for each in results:
            f.write(json.dumps(each) + "\n")

    return database


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="LLM Decontaminator")
    parser.add_argument("--model", type=str, default="gpt-4", help="The name of the OpenAI model to use")
    parser.add_argument("--database_path", type=str, required=True, help="The path to the JSONL database file")
    parser.add_argument("--output_path", type=str, required=True, help="The path to the output JSONL file")
    parser.add_argument("--data-type", type=str, default="code", help="The name of the instruction function to use")
    parser.add_argument("--max-workers", type=int, default=4, help="The maximum number of worker threads to use")

    args = parser.parse_args()

    set_openai_api_key()

    model = args.model
    database_path = args.database_path
    output_path = args.output_path
    data_type = args.data_type
    max_workers = args.max_workers

    instruct = datatype_to_instruct(data_type)

    with open(database_path, "r") as fin:
        database = [json.loads(l) for l in fin]

    # call the llm_detect function with the parsed arguments
    database = llm_detect(model, database, output_path, instruct, max_workers)
    rephrase_num = sum([1 if True in each["results"] else 0 for each in database])

    print("Rephrased {} test cases.".format(rephrase_num))

