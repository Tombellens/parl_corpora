from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import os
import time
from datetime import datetime
from worker import process_single_file

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)

    data_dir = "/home/tom/data/italy"
    csv_files = ["italy.csv"]
    source_lang = "it"
    target_lang = "en"

    print(f"\n{'='*80}")
    print(f"🚀 Starting Italian parliament translation")
    print(f"📚 Method: OPUS-MT via EasyNMT (ParlaMint methodology)")
    print(f"📅 Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📁 Files to process: {csv_files}")
    print(f"🇮🇹 Source language: Italian (it) → English (en)")
    print(f"{'='*80}\n")

    overall_start = time.time()
    completed = 0
    failed = 0

    tasks = [(csv_file, data_dir, source_lang, target_lang, None)
             for csv_file in csv_files]

    with ProcessPoolExecutor(max_workers=1) as executor:
        futures = {
            executor.submit(process_single_file, csv_file, data_dir, source_lang,
                          target_lang, year_cutoff=None, batch_size=32): csv_file
            for csv_file, data_dir, source_lang, target_lang, _ in tasks
        }

        for future in as_completed(futures):
            csv_file = futures[future]
            try:
                result = future.result(timeout=36000)
                print(f"\n{result}\n", flush=True)
                completed += 1
            except Exception as e:
                print(f"\n❌ {csv_file} failed: {type(e).__name__}: {e}\n", flush=True)
                failed += 1

            print(f"📊 Progress: {completed + failed}/{len(tasks)} files processed ({completed} ✅, {failed} ❌)", flush=True)

    overall_duration = time.time() - overall_start

    print(f"\n{'='*80}")
    print(f"✅ ALL PROCESSING COMPLETE")
    print(f"📅 End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"⏱️  Total time: {overall_duration/3600:.2f} hours")
    print(f"📊 Completed: {completed}/{len(tasks)}")
    print(f"❌ Failed: {failed}/{len(tasks)}")
    print(f"{'='*80}\n")
