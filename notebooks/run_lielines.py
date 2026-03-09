"""
run_lielines.py

Runs the mmochtak/lieline BERT model over the sentence corpus and writes
a new CSV with two extra columns:
    lie_label  - model's predicted label
    lie_score  - confidence score for that label

Parallelisation strategy:
  - One reader process streams the input CSV in chunks via a shared queue
  - N GPU worker processes each load the model independently and pull chunks
  - Results are buffered and written in original row order
  - A checkpoint file records the last fully written chunk so runs can resume

Usage:
    python run_lielines.py
"""

import os
import csv
import json
import time
import torch
import pandas as pd
import torch.multiprocessing as mp
from datetime import datetime
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
INPUT_FILE       = "/home/tom/data/sentence_corpus.csv"
OUTPUT_FILE      = "/home/tom/data/sentence_corpus_predicted.csv"
CHECKPOINT_FILE  = "/home/tom/data/sentence_corpus_predicted.checkpoint.json"
MODEL            = "mmochtak/lieline"
BATCH_SIZE       = 1024   # sentences per GPU forward pass — tune to fill VRAM
CHUNK_SIZE       = 50_000 # rows per work unit dispatched to a GPU worker
MAX_LENGTH       = 512    # BERT token limit
QUEUE_DEPTH      = 4      # max chunks in flight per GPU (backpressure)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# GPU worker
# ---------------------------------------------------------------------------

def gpu_worker(rank: int, num_gpus: int, input_q: mp.Queue, output_q: mp.Queue, model_name: str):
    """
    Runs in a separate process. Pulls (chunk_id, sentences) from input_q,
    runs batched inference directly on the model (no pipeline overhead),
    pushes (chunk_id, labels, scores) to output_q.
    Exits when it receives None (sentinel).
    """
    device = f"cuda:{rank}"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, torch_dtype=torch.float16
    ).to(device)
    model.eval()

    id2label = model.config.id2label
    print(f"  [GPU {rank}] Ready — {torch.cuda.get_device_name(rank)}", flush=True)

    while True:
        item = input_q.get()
        if item is None:
            output_q.put(None)
            break

        chunk_id, sentences = item
        all_labels = []
        all_scores = []

        for i in range(0, len(sentences), BATCH_SIZE):
            batch = sentences[i : i + BATCH_SIZE]

            # Batch-tokenise in one call — far faster than the pipeline's per-sentence loop
            inputs = tokenizer(
                batch,
                max_length=MAX_LENGTH,
                truncation=True,
                padding=True,        # pad to longest in this batch, not MAX_LENGTH
                return_tensors="pt",
            ).to(device)

            with torch.no_grad():
                logits = model(**inputs).logits

            probs     = torch.softmax(logits, dim=-1)
            predicted = probs.argmax(dim=-1)

            for j in range(len(batch)):
                idx = predicted[j].item()
                all_labels.append(id2label[idx])
                all_scores.append(round(probs[j, idx].item(), 6))

        output_q.put((chunk_id, all_labels, all_scores))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_checkpoint() -> int:
    """Return the next chunk_id to process (0 if no checkpoint)."""
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE) as f:
            data = json.load(f)
        print(f"  Resuming from checkpoint: next chunk = {data['next_chunk_id']}")
        return data["next_chunk_id"]
    return 0


def save_checkpoint(next_chunk_id: int):
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump({"next_chunk_id": next_chunk_id}, f)


def count_rows(path: str) -> int:
    """Fast line count (excludes header)."""
    with open(path, "rb") as f:
        return sum(1 for _ in f) - 1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No CUDA GPUs detected.")

    print(f"\n{'='*70}")
    print(f"LieLines inference")
    print(f"Model:   {MODEL}")
    print(f"Input:   {INPUT_FILE}")
    print(f"Output:  {OUTPUT_FILE}")
    print(f"GPUs:    {num_gpus}  (batch_size={BATCH_SIZE} each)")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")

    resume_from = load_checkpoint()
    total_rows  = count_rows(INPUT_FILE)
    total_chunks = (total_rows + CHUNK_SIZE - 1) // CHUNK_SIZE
    print(f"  {total_rows:,} sentences → {total_chunks:,} chunks of {CHUNK_SIZE:,}\n")

    # -----------------------------------------------------------------------
    # Spawn GPU workers
    # -----------------------------------------------------------------------
    mp.set_start_method("spawn", force=True)

    input_q  = mp.Queue(maxsize=num_gpus * QUEUE_DEPTH)
    output_q = mp.Queue()

    workers = [
        mp.Process(
            target=gpu_worker,
            args=(rank, num_gpus, input_q, output_q, MODEL),
            daemon=True,
        )
        for rank in range(num_gpus)
    ]
    for w in workers:
        w.start()

    # -----------------------------------------------------------------------
    # Open output file
    # -----------------------------------------------------------------------
    write_mode = "a" if resume_from > 0 else "w"
    out_f = open(OUTPUT_FILE, write_mode, newline="", encoding="utf-8")

    # Detect original columns from input header
    with open(INPUT_FILE, encoding="utf-8") as tmp:
        original_fieldnames = next(csv.reader(tmp))
    out_fieldnames = original_fieldnames + ["lie_label", "lie_score"]

    writer = csv.DictWriter(out_f, fieldnames=out_fieldnames)
    if write_mode == "w":
        writer.writeheader()

    # -----------------------------------------------------------------------
    # Stream input → queue → collect output → write in order
    # -----------------------------------------------------------------------
    # pending maps chunk_id → [df_chunk, labels, scores]
    # labels/scores start as None and are filled when the worker result arrives.
    pending       = {}
    next_to_write = resume_from
    chunks_dispatched = 0
    chunks_done       = 0
    total_written     = resume_from * CHUNK_SIZE
    start = time.time()

    pbar = tqdm(total=total_chunks, initial=resume_from,
                desc="Chunks", unit="chunk")

    def collect_from_queue():
        """Drain output_q without blocking; update pending with results."""
        nonlocal chunks_done
        while not output_q.empty():
            item = output_q.get_nowait()
            if item is None:
                continue   # stray sentinel — ignore
            cid, labels, scores = item
            pending[cid][1] = labels
            pending[cid][2] = scores
            chunks_done += 1

    def try_write_pending():
        """Flush consecutive completed chunks to disk in order."""
        nonlocal next_to_write, total_written
        while next_to_write in pending and pending[next_to_write][1] is not None:
            df_chunk, labels, scores = pending.pop(next_to_write)
            rows = df_chunk.to_dict("records")
            for row, label, score in zip(rows, labels, scores):
                row["lie_label"] = label
                row["lie_score"] = score
                writer.writerow(row)
            out_f.flush()
            save_checkpoint(next_to_write + 1)
            total_written += len(rows)
            elapsed = time.time() - start
            rate = total_written / elapsed if elapsed > 0 else 0
            pbar.set_postfix(written=f"{total_written:,}", sent_s=f"{rate:,.0f}")
            pbar.update(1)
            next_to_write += 1

    # Read CSV in chunks, skip already-processed chunks
    reader = pd.read_csv(INPUT_FILE, chunksize=CHUNK_SIZE, low_memory=False)

    for chunk_id, df_chunk in enumerate(reader):
        if chunk_id < resume_from:
            continue

        sentences = df_chunk["sentence"].fillna("").astype(str).tolist()

        # Store df before dispatching so we can pair it with results later
        pending[chunk_id] = [df_chunk, None, None]
        input_q.put((chunk_id, sentences))   # blocks on backpressure — intentional
        chunks_dispatched += 1

        # Collect any results that are already done
        collect_from_queue()
        try_write_pending()

    # Signal each worker to stop
    for _ in workers:
        input_q.put(None)

    # Drain all remaining results
    while chunks_done < chunks_dispatched:
        item = output_q.get()   # blocking wait
        if item is None:
            continue
        cid, labels, scores = item
        pending[cid][1] = labels
        pending[cid][2] = scores
        chunks_done += 1
        try_write_pending()

    pbar.close()
    out_f.close()

    for w in workers:
        w.join()

    # Clean up checkpoint now that we're done
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)

    elapsed = time.time() - start
    print(f"\n{'='*70}")
    print(f"Done. {total_written:,} sentences labelled.")
    print(f"Output: {OUTPUT_FILE}")
    print(f"Elapsed: {elapsed/60:.1f} min  ({total_written/elapsed:,.0f} sent/s avg)")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
