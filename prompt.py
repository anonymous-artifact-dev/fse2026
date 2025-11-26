import os
import time
import argparse
from typing import List, Tuple, Set
import random
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from subprocess import run
from openai import OpenAI
from openai.types.chat import (
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)


DATASET_DIR = "dataset"
RESULTS_DIR = "results"
REPORTS_DIR = os.path.join(RESULTS_DIR, "reports")
TEMP_DIR = "temp"

TARGET_FILES = [
    "app.csv",
    "so_test.csv",
    "github.csv",
    "jira_test.csv",
    "gerrit.csv",
    "tweets.csv",
    "tweets_p.csv",
    "tweets_n.csv",
]

VALID_LABELS = {
    "github.csv": ["positive", "neutral", "negative"]
}


# ===== Utilities =====
def is_multilabel_file(filename: str) -> bool:
    return filename.lower().startswith("tweets")


def ensure_dirs():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    os.makedirs(TEMP_DIR, exist_ok=True)


def read_csv_any(path: str) -> pd.DataFrame:
    tried = ["utf-8", "utf-8-sig", "cp1252", "latin-1"]
    last_err = None
    for enc in tried:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Failed to read csv file: {path} , {last_err}")


def read_dataset(path: str, fname: str) -> pd.DataFrame:
    df = read_csv_any(path)
    if "text" not in df.columns or "label" not in df.columns:
        raise RuntimeError(f"No 'text' or 'label': {path}")
    df = df.dropna(subset=["text", "label"]).reset_index(drop=True)
    if fname == "github.csv":
        valid = [l.lower() for l in VALID_LABELS["github.csv"]]
        df = df[df["label"].astype(str).str.lower().isin(valid)].reset_index(drop=True)
    return df


def extract_labels_single(df: pd.DataFrame, fname: str) -> List[str]:
    labels = sorted(set(map(lambda x: str(x).strip().lower(), df["label"].tolist())))
    labels = [l for l in labels if l not in ("none", "unknown")]
    if fname in VALID_LABELS:
        valid = [l.lower() for l in VALID_LABELS[fname]]
        labels = [l for l in labels if l in valid]
    return labels


def extract_labels_multi(df: pd.DataFrame) -> List[str]:
    uniq: Set[str] = set()
    for raw in df["label"].astype(str).tolist():
        parts = [p.strip().lower() for p in raw.split(",") if p.strip()]
        uniq.update(parts)
    return sorted(uniq)


def pair_or_split(fname: str, dataset_dir: str, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    stem = os.path.splitext(fname)[0]
    base = stem.replace("_test", "")
    if fname in ("so_test.csv", "jira_test.csv"):
        train_path = os.path.join(dataset_dir, f"{base}_train.csv")
        test_path  = os.path.join(dataset_dir, f"{base}_test.csv")
        if not os.path.isfile(train_path) or not os.path.isfile(test_path):
            raise RuntimeError(f"No train/test set: {train_path}, {test_path}")
        train_df = read_dataset(train_path, f"{base}_train.csv")
        test_df  = read_dataset(test_path, f"{base}_test.csv")
        return train_df, test_df
    path = os.path.join(dataset_dir, fname)
    df = read_dataset(path, fname)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=seed, shuffle=True)
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def balanced_sample_single(train_df: pd.DataFrame, k: int, allowed_labels: List[str], seed: int = 42) -> pd.DataFrame:
    if k <= 0:
        return pd.DataFrame(columns=train_df.columns)

    df = train_df.copy()
    df["__lbl__"] = df["label"].astype(str).str.lower()
    deny = {"none", "unknown"}
    allow = set(l.lower() for l in allowed_labels)
    df = df[~df["__lbl__"].isin(deny)]
    df = df[df["__lbl__"].isin(allow)]
    if df.empty:
        return pd.DataFrame(columns=train_df.columns)

    random.seed(seed)
    labels = sorted(df["__lbl__"].unique())
    buckets = {lb: df[df["__lbl__"] == lb] for lb in labels}
    per = max(1, k // max(1, len(labels)))

    picked = []
    for lb in labels:
        sub = buckets[lb]
        if len(sub) <= per:
            picked.append(sub)
        else:
            picked.append(sub.sample(per, random_state=seed))

    result = pd.concat(picked) if picked else pd.DataFrame(columns=df.columns)
    if not result.empty:
        result = result.sample(frac=1.0, random_state=seed)

    if len(result) < k:
        remain = df.drop(result.index, errors="ignore")
        if len(remain) > 0:
            need = k - len(result)
            add = remain.sample(min(need, len(remain)), random_state=seed)
            result = pd.concat([result, add])

    result = result.head(k).drop(columns=["__lbl__"], errors="ignore").reset_index(drop=True)
    return result


def random_sample_multi(train_df: pd.DataFrame, k: int, seed: int = 42) -> pd.DataFrame:
    if k <= 0:
        return pd.DataFrame(columns=train_df.columns)
    return train_df.sample(min(k, len(train_df)), random_state=seed).reset_index(drop=True)


def apply_da_to_fewshot(dataset_tag: str, da_method: str, few_df: pd.DataFrame) -> Tuple[pd.DataFrame, float]:
    if da_method == "none":
        return few_df.copy(), 0.0
    input_path  = os.path.join(TEMP_DIR, f"{dataset_tag}_few_in.csv")
    output_path = os.path.join(TEMP_DIR, f"{dataset_tag}_few_{da_method}.csv")
    few_df.to_csv(input_path, index=False)
    start = time.time()
    run(["python", f"augment/{da_method}.py", "--input", input_path, "--output", output_path], check=True)
    elapsed = time.time() - start
    aug_df = read_csv_any(output_path)
    if "text" not in aug_df.columns or "label" not in aug_df.columns:
        raise RuntimeError(f"No 'text'/'label': {output_path}")
    merged = pd.concat([few_df, aug_df], ignore_index=True)
    merged = merged.dropna(subset=["text", "label"]).reset_index(drop=True)
    return merged, elapsed


def make_fewshot_block_single(few_df: pd.DataFrame) -> str:
    lines = []
    for _, r in few_df.iterrows():
        text = str(r["text"]).strip().replace('\n',' ')
        label = str(r["label"]).strip().lower()
        if label in ("none", "unknown"):
            continue
        lines.append(f'Text: "{text}"\nAnswer: {label}')
    return "\n\n".join(lines)


def make_fewshot_block_multi(few_df: pd.DataFrame) -> str:
    lines = []
    for _, r in few_df.iterrows():
        text = str(r["text"]).strip().replace('\n',' ')
        raw  = str(r["label"])
        labels = [p.strip().lower() for p in raw.split(",") if p.strip()]
        ans = ",".join(labels) if labels else "none"
        lines.append(f'Text: "{text}"\nAnswer: {ans}')
    return "\n\n".join(lines)


def build_messages_single_with_fewshot(label_list: List[str], few_block: str, text: str):
    sys_prompt = "You are an expert text classifier. Follow the format exactly."
    guide = f"Choose exactly one label from: ({', '.join(label_list)}). Respond ONLY with the label."
    user_prompt = f"{guide}\n\n### Examples\n{few_block}\n\n### Query\nText: \"{text}\"\nAnswer:"
    return [
        ChatCompletionSystemMessageParam(role="system", content=sys_prompt),
        ChatCompletionUserMessageParam(role="user", content=user_prompt),
    ]


def build_messages_multi_with_fewshot(label_list: List[str], few_block: str, text: str):
    sys_prompt = "You are an expert multilabel text classifier. Follow the format exactly."
    none_rule = ""
    if "none" in [l.lower() for l in label_list]:
        none_rule = " If none apply, reply exactly 'none' (not combined)."
    guide = (
        f"Choose zero or more labels from: ({', '.join(label_list)}). "
        f"Return a comma-separated list with NO spaces (e.g., a,b).{none_rule}"
    )
    user_prompt = f"{guide}\n\n### Examples\n{few_block}\n\n### Query\nText: \"{text}\"\nAnswer:"
    return [
        ChatCompletionSystemMessageParam(role="system", content=sys_prompt),
        ChatCompletionUserMessageParam(role="user", content=user_prompt),
    ]


def build_messages_single_zero(label_list: List[str], text: str):
    sys_prompt = "You are an expert text classifier. Follow the format exactly."
    guide = f"Choose exactly one label from: ({', '.join(label_list)}). Respond ONLY with the label."
    user_prompt = f"{guide}\n\n### Query\nText: \"{text}\"\nAnswer:"
    return [
        ChatCompletionSystemMessageParam(role="system", content=sys_prompt),
        ChatCompletionUserMessageParam(role="user", content=user_prompt),
    ]


def build_messages_multi_zero(label_list: List[str], text: str):
    sys_prompt = "You are an expert multilabel text classifier. Follow the format exactly."
    none_rule = ""
    if "none" in [l.lower() for l in label_list]:
        none_rule = " If none apply, reply exactly 'none' (not combined)."
    guide = (
        f"Choose zero or more labels from: ({', '.join(label_list)}). "
        f"Return a comma-separated list with NO spaces (e.g., a,b).{none_rule}"
    )
    user_prompt = f"{guide}\n\n### Query\nText: \"{text}\"\nAnswer:"
    return [
        ChatCompletionSystemMessageParam(role="system", content=sys_prompt),
        ChatCompletionUserMessageParam(role="user", content=user_prompt),
    ]


def parse_prediction_single(reply: str, label_list: List[str]) -> str:
    pred = reply.strip().lower().split()[0] if reply.strip() else "none"
    if pred not in label_list:
        pred = "none"
    return pred


def parse_prediction_multi(reply: str, label_list: List[str]) -> List[str]:
    reply = reply.strip().lower()
    if reply == "":
        return ["none"] if "none" in label_list else []
    parts = [p.strip() for p in reply.split(",") if p.strip()]
    if "none" in parts and len(parts) > 1:
        parts = [p for p in parts if p != "none"]
    parts = [p for p in parts if p in label_list]
    if not parts and "none" in label_list:
        return ["none"]
    return parts


def evaluate_single(y_true: List[str], y_pred: List[str], label_list: List[str]) -> Tuple[float, str]:
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, labels=label_list, zero_division=0, digits=4)
    return acc, report


def evaluate_multi(y_true_multi: List[List[str]], y_pred_multi: List[List[str]], label_list: List[str]) -> str:
    mlb = MultiLabelBinarizer(classes=label_list)
    Y_true = mlb.fit_transform(y_true_multi)
    Y_pred = mlb.transform(y_pred_multi)
    report = classification_report(Y_true, Y_pred, target_names=label_list, zero_division=0, digits=4)
    return report


def safe_openai_chat(client: OpenAI, messages, model="gpt-4.1", max_retries=2) -> str:
    last_err = None
    for attempt in range(max_retries + 1):
        try:
            resp = client.chat.completions.create(model=model, messages=messages)
            return resp.choices[0].message.content or ""
        except Exception as e:
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Failed to call OpenAI API: {last_err}")


def main():
    parser = argparse.ArgumentParser(description="Few-shot + DA on few-shot, then test classification (k=0 => zero-shot).")
    parser.add_argument("--dataset_dir", type=str, default=DATASET_DIR)
    parser.add_argument("--results_dir", type=str, default=RESULTS_DIR)
    parser.add_argument("--model", type=str, default="gpt-4.1", help="e.g., gpt-4.1 or gpt-4.1-mini")
    parser.add_argument("--shots", type=str, default="1,2,4,8", help="few-shot counts, comma-separated (ignored if --fewshot_k is set)")
    parser.add_argument("--fewshot_k", type=int, default=None, help="single K to run (overrides --shots if set)")
    parser.add_argument("--da", type=str, default="none", help="DA method applied to few-shot only (none,aeda,ssmba,...)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None, help="limit test rows per dataset")
    args = parser.parse_args()

    random.seed(args.seed)
    ensure_dirs()
    client = OpenAI()

    if args.fewshot_k is not None:
        shots_list = [int(args.fewshot_k)]
    else:
        shots_list = [int(s.strip()) for s in args.shots.split(",") if s.strip().isdigit()]

    for fname in TARGET_FILES:
        print(f"\n=== Dataset: {fname} ===")
        try:
            train_df, test_df = pair_or_split(fname, args.dataset_dir, seed=args.seed)
        except Exception as e:
            print(f"Skip {fname}: {e}")
            continue

        multilabel = is_multilabel_file(fname)

        if multilabel:
            labels = extract_labels_multi(train_df)
        else:
            labels = extract_labels_single(train_df, fname)
        if "none" not in labels:
            labels.append("none")

        if args.limit:
            test_df = test_df.head(args.limit).reset_index(drop=True)

        for K in shots_list:
            dataset_tag = f"{os.path.splitext(fname)[0]}_k{K}"
            print(f"- Shots={K}, DA={args.da}")

            if K == 0:
                zero_shot = True
                few_orig = pd.DataFrame(columns=["text", "label"])
                few_used = few_orig.copy()
                da_elapsed = 0.0
                if multilabel:
                    build_msg_fn = lambda label_list, few_block, text: build_messages_multi_zero(label_list, text)
                    parse_fn = parse_prediction_multi
                else:
                    build_msg_fn = lambda label_list, few_block, text: build_messages_single_zero(label_list, text)
                    parse_fn = parse_prediction_single
                few_block = "(zero-shot: no examples provided)"
            else:
                zero_shot = False
                if multilabel:
                    few_orig = random_sample_multi(train_df, K, seed=args.seed)
                else:
                    few_orig = balanced_sample_single(train_df, K, allowed_labels=[l for l in labels if l != "none"], seed=args.seed)

                if few_orig.empty:
                    continue

                few_used, da_elapsed = apply_da_to_fewshot(dataset_tag, args.da, few_orig)

                if multilabel:
                    few_block = make_fewshot_block_multi(few_used)
                    build_msg_fn = build_messages_multi_with_fewshot
                    parse_fn = parse_prediction_multi
                else:
                    few_block = make_fewshot_block_single(few_used)
                    if not few_block.strip():
                        continue
                    build_msg_fn = build_messages_single_with_fewshot
                    parse_fn = parse_prediction_single

            overall_start = time.time()
            y_true_single, y_pred_single = [], []
            y_true_multi,  y_pred_multi  = [], []
            results = []
            sum_infer_time = 0.0

            for i, row in test_df.iterrows():
                text = str(row["text"])
                raw_label = str(row["label"]).strip().lower()

                msg = build_msg_fn([lb for lb in labels], few_block, text)

                t0 = time.time()
                try:
                    reply = safe_openai_chat(client, msg, model=args.model)
                except Exception as e:
                    reply = ""
                    print(f"[{i}] API Failed: {e}")
                infer_t = time.time() - t0
                sum_infer_time += infer_t

                if multilabel:
                    true_labels = [p.strip() for p in raw_label.split(",") if p.strip()] or ["none"]
                    pred_labels = parse_fn(reply, labels)
                    y_true_multi.append(true_labels)
                    y_pred_multi.append(pred_labels)
                    results.append((text, ",".join(true_labels), ",".join(pred_labels), infer_t))
                else:
                    true_label = raw_label if raw_label else "none"
                    pred = parse_fn(reply, labels)
                    y_true_single.append(true_label)
                    y_pred_single.append(pred)
                    results.append((text, true_label, pred, infer_t))

                if (i+1) % 20 == 0:
                    print(f"  ... {i+1} rows processed")

            overall_elapsed = time.time() - overall_start


            if multilabel:
                report = evaluate_multi(y_true_multi, y_pred_multi, labels)
                acc_str = "Accuracy: - (use F1 for multilabel)"
            else:
                acc, report = evaluate_single(y_true_single, y_pred_single, labels)
                acc_str = f"Accuracy: {acc:.4f}"


            model_tag = args.model.replace("/", "-")
            base = os.path.splitext(fname)[0]
            out_name = f"chatgpt_{base}_{args.da}_k{K}_{model_tag}.txt"
            out_path = os.path.join(RESULTS_DIR, out_name)
            rep_name = f"{base}_{args.da}_k{K}_{model_tag}_report.txt"
            rep_path = os.path.join(REPORTS_DIR, rep_name)

            with open(out_path, "w", encoding="utf-8") as f:
                f.write(f"=== ChatGPT Classification Results ===\n")
                f.write(f"dataset: {fname}\n")
                f.write(f"model: {args.model}\n")
                f.write(f"mode: {'multi-label' if multilabel else 'single-label'}\n")
                f.write(f"labels: {', '.join(labels)}\n")
                f.write(f"few-shot K: {K} ({'zero-shot' if zero_shot else 'few-shot'})\n")
                f.write(f"DA method (few-shot only): {args.da if not zero_shot else 'skipped in zero-shot'}\n\n")

                f.write("=== Few-shot Examples (post-DA) ===\n")
                f.write(few_block + "\n\n")

                f.write("=== Predictions ===\n")
                for idx, (tx, yt, yp, tsec) in enumerate(results):
                    f.write(f"[{idx}] Text: {tx}\n")
                    f.write(f"     True: {yt} | Pred: {yp} | Time: {tsec:.2f} sec\n")
                f.write("\n=== Evaluation Metrics ===\n")
                f.write(acc_str + "\n\n")
                f.write("Classification Report:\n")
                f.write(report + "\n")

                f.write("\n=== Time Summary ===\n")
                f.write(f"DA on few-shot: {da_elapsed:.2f} sec\n")
                f.write(f"Total inference (sum over test): {sum_infer_time:.2f} sec\n")
                f.write(f"Overall elapsed (DA + inference + overhead): {overall_elapsed:.2f} sec\n")

            with open(rep_path, "w", encoding="utf-8") as f:
                f.write(report)

            print(f"Output Saved: {out_path}")
            print(f"Report saved: {rep_path}")

    print("\n Completed.")

if __name__ == "__main__":
    main()
