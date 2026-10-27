import argparse
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download
from tokenizers import Tokenizer

DATASET_ID = "lmsys/lmsys-chat-1m"
TOKENIZER_ID = "hf-internal-testing/llama-tokenizer"
AUTHORS_FILE_NAME = "input_output_word_counts10000+(1).csv"
LENGTH_COLUMNS = ["input", "output", "input_tokens", "output_tokens"]


def iter_conversations(num_conversations):
    files = sorted(
        f
        for f in HfApi().list_repo_files(DATASET_ID, repo_type="dataset")
        if f.endswith(".parquet")
    )
    remaining = num_conversations
    for filename in files:
        path = hf_hub_download(DATASET_ID, filename, repo_type="dataset")
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(
            batch_size=10000,
            columns=["conversation_id", "model", "language", "turn", "conversation"],
        ):
            for row in batch.to_pylist():
                yield row
                remaining -= 1
                if remaining == 0:
                    return


def split_turns(conversation, turns):
    user = [m.get("content") or "" for m in conversation if m.get("role") == "user"]
    assistant = [
        m.get("content") or "" for m in conversation if m.get("role") == "assistant"
    ]
    if turns == "first":
        return (user[0] if user else ""), (assistant[0] if assistant else "")
    return "\n".join(user), "\n".join(assistant)


def build_lengths(num_conversations, turns):
    tokenizer = Tokenizer.from_pretrained(TOKENIZER_ID)
    records, inputs, outputs = [], [], []
    for row in iter_conversations(num_conversations):
        prompt, response = split_turns(row["conversation"], turns)
        records.append(
            {
                "conversation_id": row["conversation_id"],
                "model": row["model"],
                "language": row["language"],
                "turn": row["turn"],
                "input": len(prompt.split()),
                "output": len(response.split()),
            }
        )
        inputs.append(prompt)
        outputs.append(response)

    df = pd.DataFrame(records)
    df["input_tokens"] = [
        len(e.ids) for e in tokenizer.encode_batch(inputs, add_special_tokens=False)
    ]
    df["output_tokens"] = [
        len(e.ids) for e in tokenizer.encode_batch(outputs, add_special_tokens=False)
    ]
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Build per-request input/output lengths (words and Llama-2 tokens) from LMSYS-Chat-1M."
    )
    parser.add_argument("--num-conversations", type=int, default=10000)
    parser.add_argument(
        "--turns",
        choices=["first", "all"],
        default="first",
        help="first: first user/assistant turn; all: concatenation of all user/assistant turns.",
    )
    parser.add_argument("--output", type=str, default="")
    parser.add_argument(
        "--no-authors-copy",
        action="store_true",
        help=f"Do not write the '{AUTHORS_FILE_NAME}' copy expected by run_online_lp_policies.py --repro-input-output-csv.",
    )
    parser.add_argument(
        "--repo-root", type=str, default=str(Path(__file__).resolve().parents[1])
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root)
    output = (
        Path(args.output)
        if args.output
        else repo_root
        / "data"
        / "processed_traces"
        / f"lmsys_chat_1m_{args.turns}{args.num_conversations}.csv"
    )

    df = build_lengths(args.num_conversations, args.turns)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    print(f"Wrote {len(df)} rows to {output}")

    if not args.no_authors_copy:
        authors_copy = repo_root / AUTHORS_FILE_NAME
        df[["input", "output"]].to_csv(authors_copy, index=False)
        print(f"Wrote authors-format copy (words) to {authors_copy}")

    stats = df[LENGTH_COLUMNS].describe(percentiles=[0.5, 0.9, 0.99]).T
    print(stats[["mean", "50%", "90%", "99%", "max"]].round(1).to_string())
    for prefix, (prefill, decode) in {
        "words": ("input", "output"),
        "tokens": ("input_tokens", "output_tokens"),
    }.items():
        over = ((df[prefill] + df[decode]) > 4096).mean() * 100
        print(f"{prefix}: {over:.2f}% of requests exceed 4096 total and get clipped by Vidur")


if __name__ == "__main__":
    main()
