from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import os
import time
from datetime import datetime
from worker import process_single_file

# Language mapping - OPUS-MT language codes
# These match the ParlaMint methodology
lang_map = {
    "Bundestag": "de",       # German      # English (UK) - will skip translation
    "Congreso": "es",        # Spanish
    "Folketing": "da",       # Danish      # English (New Zealand) - will skip translation
    "PSP": "cs",             # Czech
    "TweedeKamer": "nl",     # Dutch
    "Corp_Riksdagen_V2": "sv"
}

# Year cutoff per country (None = process all, otherwise only < year)
year_cutoffs = {
    "Bundestag": None,       # Translate all
    "Congreso": None,        # Translate all
    "Folketing": None,       # Translate all
    "PSP": None,             # Translate all
    "TweedeKamer": None,     # Translate all
    "Corp_Riksdagen_V2": None
}

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)

    data_dir = "/home/tom/data/parlspeech"
    csv_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]

    tasks = []
    for csv_file in csv_files:
        name = os.path.splitext(os.path.basename(csv_file))[0]
        source_lang = lang_map.get(name)

        if source_lang:
            # Skip files that are already in English
            if source_lang == 'en':
                print(f"⏭️  Skipping {csv_file} (already in English)")
                continue

            year_cutoff = year_cutoffs.get(name)
            target_lang = "en"  # All translate to English
            tasks.append((csv_file, data_dir, source_lang, target_lang, year_cutoff))

    print(f"\n{'='*80}")
    print(f"🚀 Starting translation using ParlaMint methodology")
    print(f"📚 Method: OPUS-MT via EasyNMT")
    print(f"📅 Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📁 Files to process: {len(tasks)}")
    print(f"⚙️  Workers: 1")
    print(f"💾 Batch size: 32")
    print(f"\n📝 Reference: ParlaMint corpus translation methodology")
    print(f"   https://github.com/TajaKuzman/Parlamint-translation")
    print(f"\n📅 Year cutoffs:")
    for name, cutoff in year_cutoffs.items():
        if name not in ["Commons", "NZHoR"]:  # Skip English corpora
            if cutoff:
                print(f"   {name}: < {cutoff}")
            else:
                print(f"   {name}: All years")
    print(f"{'='*80}\n")

    overall_start = time.time()
    completed = 0
    failed = 0

    # Use 1 worker - OPUS-MT models are relatively lightweight
    # but EasyNMT downloads models on-demand which can cause issues with parallel processing
    with ProcessPoolExecutor(max_workers=1) as executor:
        futures = {
            executor.submit(process_single_file, csv_file, data_dir, source_lang,
                          target_lang, year_cutoff=year_cutoff, batch_size=32): csv_file
            for csv_file, data_dir, source_lang, target_lang, year_cutoff in tasks
        }

        for future in as_completed(futures):
            csv_file = futures[future]
            try:
                result = future.result(timeout=36000)  # 10 hour timeout per file
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
    print(f"\n📝 Methodology: Replicated ParlaMint corpus translation approach")
    print(f"   Model: OPUS-MT (Helsinki-NLP)")
    print(f"   Framework: EasyNMT")
    print(f"   Translation level: Sentence-by-sentence with automatic chunking")
    print(f"{'='*80}\n")
