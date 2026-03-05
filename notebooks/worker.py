import time
import pandas as pd
import traceback
import os
import json
from datetime import datetime
from easynmt import EasyNMT
import nltk
import re

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')

def split_into_sentences(text, source_lang='en', max_sentence_length=500):
    """
    Split text into sentences with language-specific handling

    Args:
        text: Input text
        source_lang: Source language code
        max_sentence_length: Maximum characters per sentence (for fallback splitting)
    """

    # SPECIAL CASE: Czech uses double spaces as sentence delimiters
    if source_lang == 'cs':
        # Split on double (or more) spaces
        sentences = re.split(r'\s{2,}', text)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 5]

        if sentences:
            return sentences

    # For other languages, try NLTK first
    try:
        sentences = nltk.sent_tokenize(text)

        # Check if NLTK failed (returns only 1 sentence for long text)
        if len(sentences) == 1 and len(text) > 1000:
            # Fallback: split on common sentence endings + newlines
            sentences = re.split(r'(?<=[.!?])\s+|\n+', text)
            sentences = [s.strip() for s in sentences if s.strip()]

        # If still just one long sentence, force split by length
        if len(sentences) == 1 and len(text) > max_sentence_length:
            # Split on any reasonable boundary (punctuation or newline)
            sentences = re.split(r'[.!?;:\n]+', text)
            sentences = [s.strip() for s in sentences if s.strip() and len(s) > 10]

        # If STILL one sentence, split by character limit as last resort
        if len(sentences) == 1 and len(text) > max_sentence_length:
            # Split into chunks of max_sentence_length at word boundaries
            words = text.split()
            sentences = []
            current = []
            current_len = 0

            for word in words:
                if current_len + len(word) + 1 > max_sentence_length and current:
                    sentences.append(' '.join(current))
                    current = [word]
                    current_len = len(word)
                else:
                    current.append(word)
                    current_len += len(word) + 1

            if current:
                sentences.append(' '.join(current))

        return sentences

    except Exception as e:
        # Ultimate fallback: return as-is
        print(f"Warning: sentence splitting failed: {e}", flush=True)
        return [text]

def process_single_file(csv_file, data_dir, source_lang, target_lang='en', year_cutoff=None, batch_size=32):
    """
    Process using EXACT ParlaMint methodology:
    - OPUS-MT models via EasyNMT
    - Sentence-by-sentence translation
    - Language-specific sentence splitting
    """

    try:
        name = os.path.splitext(os.path.basename(csv_file))[0]
        checkpoint_file = f"checkpoint_{name}.json"
        output_file = f"translated_{name}.csv"
        stats_file = f"translation_stats_{name}.json"

        if os.path.exists(output_file):
            print(f"[{csv_file}] ✅ Already completed, skipping.", flush=True)
            return f"✅ {csv_file}: Already completed (found {output_file})"

        start_idx = 0
        sentence_stats = {'total_texts': 0, 'total_sentences': 0, 'avg_sentences_per_text': 0, 'single_sentence_texts': 0}

        if os.path.exists(checkpoint_file):
            with open(checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
                start_idx = checkpoint['last_completed_idx']
                if 'sentence_stats' in checkpoint:
                    sentence_stats = checkpoint['sentence_stats']
                print(f"[{csv_file}] 📂 Resuming from checkpoint at index {start_idx}", flush=True)

        print(f"[{csv_file}] 🔧 Loading OPUS-MT model via EasyNMT (lang: {source_lang})...", flush=True)

        # This is EXACTLY what ParlaMint used
        model = EasyNMT('opus-mt')

        print(f"[{csv_file}] 📖 Reading CSV...", flush=True)
        path = os.path.join(data_dir, csv_file)
        df = pd.read_csv(path)

        if "text" not in df.columns:
            return f"⚠️ {csv_file}: No 'text' column"

        original_count = len(df)
        if year_cutoff is not None and "date" in df.columns:
            df['year'] = pd.to_datetime(df['date'], errors='coerce').dt.year
            df = df[df['year'] < year_cutoff].copy()
            df = df.drop('year', axis=1)
            print(f"[{csv_file}] 📅 Filtered: {len(df)}/{original_count} rows with year < {year_cutoff}", flush=True)

        if len(df) == 0:
            return f"⚠️ {csv_file}: No rows to translate after filtering"

        texts_to_translate = df["text"].dropna().astype(str).tolist()
        valid_indices = df["text"].dropna().index.tolist()

        if not texts_to_translate:
            return f"⚠️ {csv_file}: No valid text"

        total_texts = len(texts_to_translate)

        # Special message for Czech
        if source_lang == 'cs':
            print(f"[{csv_file}] 🇨🇿 Czech detected: using double-space sentence splitting", flush=True)

        print(f"[{csv_file}] 🚀 Processing {total_texts} texts (sentence-by-sentence like ParlaMint)...", flush=True)

        if start_idx == 0:
            df['en_translation'] = None
        else:
            if os.path.exists(f"partial_{output_file}"):
                partial_df = pd.read_csv(f"partial_{output_file}")
                df = partial_df

        file_start_time = time.time()
        last_checkpoint_time = time.time()

        for i in range(start_idx, total_texts, batch_size):
            batch_start = time.time()

            batch_texts = texts_to_translate[i:i+batch_size]
            batch_indices = valid_indices[i:i+batch_size]

            # EXPLICIT sentence-level translation (ParlaMint approach)
            batch_translated = []

            for text in batch_texts:
                # Split into sentences with language-specific handling
                sentences = split_into_sentences(text, source_lang=source_lang, max_sentence_length=500)
                sentence_stats['total_sentences'] += len(sentences)

                # Track texts that only have 1 sentence (potential issue)
                if len(sentences) == 1 and len(text) > 500:
                    sentence_stats['single_sentence_texts'] += 1

                # Translate each sentence individually
                if len(sentences) > 0:
                    translated_sentences = model.translate(
                        sentences,
                        source_lang=source_lang,
                        target_lang=target_lang
                    )
                    # Recombine into full text
                    full_translation = ' '.join(translated_sentences)
                    batch_translated.append(full_translation)
                else:
                    batch_translated.append('')

                sentence_stats['total_texts'] += 1

            # Update statistics
            if sentence_stats['total_texts'] > 0:
                sentence_stats['avg_sentences_per_text'] = sentence_stats['total_sentences'] / sentence_stats['total_texts']

            for j, translation in enumerate(batch_translated):
                df_idx = batch_indices[j]
                df.at[df_idx, 'en_translation'] = translation

            batch_duration = time.time() - batch_start
            batch_num = (i - start_idx) // batch_size + 1

            if batch_num % 10 == 0 or (time.time() - last_checkpoint_time) > 300:
                elapsed = time.time() - file_start_time
                progress = (i + len(batch_texts)) / total_texts * 100
                speed = (i + len(batch_texts) - start_idx) / elapsed
                remaining = (total_texts - i - len(batch_texts)) / speed if speed > 0 else 0

                avg_sent = sentence_stats['avg_sentences_per_text']
                single_sent_pct = (sentence_stats['single_sentence_texts'] / sentence_stats['total_texts'] * 100) if sentence_stats['total_texts'] > 0 else 0

                print(f"[{csv_file}] {progress:.1f}% | {i+len(batch_texts)}/{total_texts} | "
                      f"{speed:.1f} texts/sec | Avg {avg_sent:.1f} sent/text | "
                      f"Single-sent: {single_sent_pct:.1f}% | ETA: {remaining/60:.1f} min", flush=True)

                checkpoint = {
                    'last_completed_idx': i + len(batch_texts),
                    'timestamp': datetime.now().isoformat(),
                    'progress_pct': progress,
                    'speed': speed,
                    'sentence_stats': sentence_stats
                }
                with open(checkpoint_file, 'w') as f:
                    json.dump(checkpoint, f)

                df.to_csv(f"partial_{output_file}", index=False)
                last_checkpoint_time = time.time()

        total_duration = time.time() - file_start_time
        speed = total_texts / total_duration

        df.to_csv(output_file, index=False)

        single_sent_pct = (sentence_stats['single_sentence_texts'] / sentence_stats['total_texts'] * 100) if sentence_stats['total_texts'] > 0 else 0

        final_stats = {
            'model': 'OPUS-MT via EasyNMT (ParlaMint sentence-by-sentence methodology)',
            'source_language': source_lang,
            'total_texts': sentence_stats['total_texts'],
            'total_sentences_translated': sentence_stats['total_sentences'],
            'avg_sentences_per_text': sentence_stats['avg_sentences_per_text'],
            'single_sentence_texts_pct': single_sent_pct,
            'translation_speed_texts_per_sec': speed,
            'total_duration_minutes': total_duration / 60
        }

        with open(stats_file, 'w') as f:
            json.dump(final_stats, f, indent=2)

        if os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)
        if os.path.exists(f"partial_{output_file}"):
            os.remove(f"partial_{output_file}")

        result = (f"✅ {csv_file}: {total_texts} texts ({sentence_stats['total_sentences']} sentences, "
                 f"{single_sent_pct:.1f}% single-sent) in {total_duration/60:.1f} min ({speed:.1f} texts/sec)")
        print(result, flush=True)
        return result

    except Exception as e:
        error_details = f"❌ {csv_file} CRASHED: {type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        print(error_details, flush=True)

        if 'i' in locals():
            checkpoint = {
                'last_completed_idx': i,
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'sentence_stats': sentence_stats if 'sentence_stats' in locals() else {}
            }
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint, f)
            print(f"[{csv_file}] 💾 Checkpoint saved at index {i}", flush=True)

        return error_details
