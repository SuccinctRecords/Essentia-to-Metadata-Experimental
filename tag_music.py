#!/usr/bin/env python3
"""
Analyze music library with Essentia and write genre/mood/instrument/context tags.
Supports both interactive mode and CLI arguments for automation.
All classifier models use the discogs-effnet embedding backbone.
"""
import os
import json
import sys
import argparse
from pathlib import Path
from datetime import datetime
import platform
import urllib.request
import urllib.error
import numpy as np
from essentia.standard import MonoLoader, TensorflowPredictEffnetDiscogs, TensorflowPredict2D
import essentia
essentia.log.warningActive = False
import mutagen
from mutagen.flac import FLAC
from mutagen.id3 import ID3, TCON, COMM, TXXX
from mutagen.oggvorbis import OggVorbis
from mutagen.oggopus import OggOpus
from mutagen.mp4 import MP4
from mutagen.aiff import AIFF
from mutagen.wavpack import WavPack
from mutagen.musepack import Musepack
from mutagen.apev2 import APEv2
from mutagen.asf import ASF

# Supported audio file extensions
AUDIO_EXTENSIONS = {
    '.flac',                    # FLAC - Vorbis comments
    '.mp3',                     # MP3 - ID3v2
    '.ogg', '.oga',             # Ogg Vorbis - Vorbis comments
    '.opus',                    # Opus - Vorbis comments
    '.m4a', '.m4b', '.mp4', '.aac',  # AAC/ALAC - MP4 atoms
    '.wma',                     # WMA - ASF attributes
    '.aiff', '.aif',            # AIFF - ID3v2
    '.wav',                     # WAV - ID3v2 (via mutagen)
    '.wv',                      # WavPack - APEv2
    '.ape',                     # Monkey's Audio - APEv2
    '.mpc', '.mp+',             # Musepack - APEv2
    '.dsf',                     # DSD Stream File - ID3v2
}

# Model directory (fixed)
MODEL_DIR = os.path.expanduser('~/essentia_models')

# Embedding model (shared by all classifiers)
EMBEDDING_MODEL_FILE = 'discogs-effnet-bs64-1.pb'
EMBEDDING_MODEL_URL = 'https://essentia.upf.edu/models/music-style-classification/discogs-effnet/discogs-effnet-bs64-1.pb'

# ─────────────────────────────────────────────────────────────────────────────
# MODEL REGISTRY
# Each entry defines a classifier head that runs on top of the shared
# discogs-effnet embedding.  Fields:
#   display_name  – human-readable label (shown in TUI)
#   category      – grouping for TUI display
#   tag_field     – Vorbis comment / tag key to write results to
#   model_file    – .pb filename (stored in MODEL_DIR)
#   metadata_file – .json filename (class labels)
#   pb_url        – download URL for the .pb
#   json_url      – download URL for the .json
#   input_layer   – TensorflowPredict2D input (None = default)
#   output_layer  – TensorflowPredict2D output (None = default)
#   activation    – "softmax" (binary 2-class) or "sigmoid" (multi-label)
#   multi_label   – True for sigmoid models that return multiple results
# ─────────────────────────────────────────────────────────────────────────────
_BASE = 'https://essentia.upf.edu/models/classification-heads'

MODEL_REGISTRY = {
    # ── Genre ────────────────────────────────────────────────────────────
    'genre_discogs400': {
        'display_name': 'Genre (Discogs 400)',
        'category': 'Genre',
        'tag_field': 'GENRE',
        'model_file': 'genre_discogs400-discogs-effnet-1.pb',
        'metadata_file': 'genre_discogs400-discogs-effnet-1.json',
        'pb_url': f'{_BASE}/genre_discogs400/genre_discogs400-discogs-effnet-1.pb',
        'json_url': f'{_BASE}/genre_discogs400/genre_discogs400-discogs-effnet-1.json',
        'input_layer': 'serving_default_model_Placeholder',
        'output_layer': 'PartitionedCall:0',
        'activation': 'sigmoid',
        'multi_label': True,
    },
    'mtg_jamendo_genre': {
        'display_name': 'Genre (MTG-Jamendo)',
        'category': 'Genre',
        'tag_field': 'GENRE_JAMENDO',
        'model_file': 'mtg_jamendo_genre-discogs-effnet-1.pb',
        'metadata_file': 'mtg_jamendo_genre-discogs-effnet-1.json',
        'pb_url': f'{_BASE}/mtg_jamendo_genre/mtg_jamendo_genre-discogs-effnet-1.pb',
        'json_url': f'{_BASE}/mtg_jamendo_genre/mtg_jamendo_genre-discogs-effnet-1.json',
        'input_layer': None,
        'output_layer': None,
        'activation': 'sigmoid',
        'multi_label': True,
    },

    # ── Mood / Theme ─────────────────────────────────────────────────────
    'mtg_jamendo_moodtheme': {
        'display_name': 'Mood/Theme (MTG-Jamendo)',
        'category': 'Mood',
        'tag_field': 'MOOD',
        'model_file': 'mtg_jamendo_moodtheme-discogs-effnet-1.pb',
        'metadata_file': 'mtg_jamendo_moodtheme-discogs-effnet-1.json',
        'pb_url': f'{_BASE}/mtg_jamendo_moodtheme/mtg_jamendo_moodtheme-discogs-effnet-1.pb',
        'json_url': f'{_BASE}/mtg_jamendo_moodtheme/mtg_jamendo_moodtheme-discogs-effnet-1.json',
        'input_layer': None,
        'output_layer': None,
        'activation': 'sigmoid',
        'multi_label': True,
    },
    'mood_aggressive': {
        'display_name': 'Mood: Aggressive',
        'category': 'Mood',
        'tag_field': 'MOOD_AGGRESSIVE',
        'model_file': 'mood_aggressive-discogs-effnet-1.pb',
        'metadata_file': 'mood_aggressive-discogs-effnet-1.json',
        'pb_url': f'{_BASE}/mood_aggressive/mood_aggressive-discogs-effnet-1.pb',
        'json_url': f'{_BASE}/mood_aggressive/mood_aggressive-discogs-effnet-1.json',
        'input_layer': None,
        'output_layer': 'model/Softmax',
        'activation': 'softmax',
        'multi_label': False,
    },
    'mood_happy': {
        'display_name': 'Mood: Happy',
        'category': 'Mood',
        'tag_field': 'MOOD_HAPPY',
        'model_file': 'mood_happy-discogs-effnet-1.pb',
        'metadata_file': 'mood_happy-discogs-effnet-1.json',
        'pb_url': f'{_BASE}/mood_happy/mood_happy-discogs-effnet-1.pb',
        'json_url': f'{_BASE}/mood_happy/mood_happy-discogs-effnet-1.json',
        'input_layer': None,
        'output_layer': 'model/Softmax',
        'activation': 'softmax',
        'multi_label': False,
    },
    'mood_party': {
        'display_name': 'Mood: Party',
        'category': 'Mood',
        'tag_field': 'MOOD_PARTY',
        'model_file': 'mood_party-discogs-effnet-1.pb',
        'metadata_file': 'mood_party-discogs-effnet-1.json',
        'pb_url': f'{_BASE}/mood_party/mood_party-discogs-effnet-1.pb',
        'json_url': f'{_BASE}/mood_party/mood_party-discogs-effnet-1.json',
        'input_layer': None,
        'output_layer': 'model/Softmax',
        'activation': 'softmax',
        'multi_label': False,
    },
    'mood_relaxed': {
        'display_name': 'Mood: Relaxed',
        'category': 'Mood',
        'tag_field': 'MOOD_RELAXED',
        'model_file': 'mood_relaxed-discogs-effnet-1.pb',
        'metadata_file': 'mood_relaxed-discogs-effnet-1.json',
        'pb_url': f'{_BASE}/mood_relaxed/mood_relaxed-discogs-effnet-1.pb',
        'json_url': f'{_BASE}/mood_relaxed/mood_relaxed-discogs-effnet-1.json',
        'input_layer': None,
        'output_layer': 'model/Softmax',
        'activation': 'softmax',
        'multi_label': False,
    },
    'mood_sad': {
        'display_name': 'Mood: Sad',
        'category': 'Mood',
        'tag_field': 'MOOD_SAD',
        'model_file': 'mood_sad-discogs-effnet-1.pb',
        'metadata_file': 'mood_sad-discogs-effnet-1.json',
        'pb_url': f'{_BASE}/mood_sad/mood_sad-discogs-effnet-1.pb',
        'json_url': f'{_BASE}/mood_sad/mood_sad-discogs-effnet-1.json',
        'input_layer': None,
        'output_layer': 'model/Softmax',
        'activation': 'softmax',
        'multi_label': False,
    },
    'mood_acoustic': {
        'display_name': 'Mood: Acoustic',
        'category': 'Mood',
        'tag_field': 'MOOD_ACOUSTIC',
        'model_file': 'mood_acoustic-discogs-effnet-1.pb',
        'metadata_file': 'mood_acoustic-discogs-effnet-1.json',
        'pb_url': f'{_BASE}/mood_acoustic/mood_acoustic-discogs-effnet-1.pb',
        'json_url': f'{_BASE}/mood_acoustic/mood_acoustic-discogs-effnet-1.json',
        'input_layer': None,
        'output_layer': 'model/Softmax',
        'activation': 'softmax',
        'multi_label': False,
    },
    'mood_electronic': {
        'display_name': 'Mood: Electronic',
        'category': 'Mood',
        'tag_field': 'MOOD_ELECTRONIC',
        'model_file': 'mood_electronic-discogs-effnet-1.pb',
        'metadata_file': 'mood_electronic-discogs-effnet-1.json',
        'pb_url': f'{_BASE}/mood_electronic/mood_electronic-discogs-effnet-1.pb',
        'json_url': f'{_BASE}/mood_electronic/mood_electronic-discogs-effnet-1.json',
        'input_layer': None,
        'output_layer': 'model/Softmax',
        'activation': 'softmax',
        'multi_label': False,
    },

    # ── Context ──────────────────────────────────────────────────────────
    'danceability': {
        'display_name': 'Danceability',
        'category': 'Context',
        'tag_field': 'DANCEABILITY',
        'model_file': 'danceability-discogs-effnet-1.pb',
        'metadata_file': 'danceability-discogs-effnet-1.json',
        'pb_url': f'{_BASE}/danceability/danceability-discogs-effnet-1.pb',
        'json_url': f'{_BASE}/danceability/danceability-discogs-effnet-1.json',
        'input_layer': None,
        'output_layer': 'model/Softmax',
        'activation': 'softmax',
        'multi_label': False,
    },
    'voice_instrumental': {
        'display_name': 'Voice / Instrumental',
        'category': 'Context',
        'tag_field': 'VOICE_INSTRUMENTAL',
        'model_file': 'voice_instrumental-discogs-effnet-1.pb',
        'metadata_file': 'voice_instrumental-discogs-effnet-1.json',
        'pb_url': f'{_BASE}/voice_instrumental/voice_instrumental-discogs-effnet-1.pb',
        'json_url': f'{_BASE}/voice_instrumental/voice_instrumental-discogs-effnet-1.json',
        'input_layer': None,
        'output_layer': 'model/Softmax',
        'activation': 'softmax',
        'multi_label': False,
    },
    'gender': {
        'display_name': 'Voice Gender',
        'category': 'Context',
        'tag_field': 'VOICE_GENDER',
        'model_file': 'gender-discogs-effnet-1.pb',
        'metadata_file': 'gender-discogs-effnet-1.json',
        'pb_url': f'{_BASE}/gender/gender-discogs-effnet-1.pb',
        'json_url': f'{_BASE}/gender/gender-discogs-effnet-1.json',
        'input_layer': None,
        'output_layer': 'model/Softmax',
        'activation': 'softmax',
        'multi_label': False,
    },
    'tonal_atonal': {
        'display_name': 'Tonal / Atonal',
        'category': 'Context',
        'tag_field': 'TONAL_ATONAL',
        'model_file': 'tonal_atonal-discogs-effnet-1.pb',
        'metadata_file': 'tonal_atonal-discogs-effnet-1.json',
        'pb_url': f'{_BASE}/tonal_atonal/tonal_atonal-discogs-effnet-1.pb',
        'json_url': f'{_BASE}/tonal_atonal/tonal_atonal-discogs-effnet-1.json',
        'input_layer': None,
        'output_layer': 'model/Softmax',
        'activation': 'softmax',
        'multi_label': False,
    },
    'timbre': {
        'display_name': 'Timbre (Bright / Dark)',
        'category': 'Context',
        'tag_field': 'TIMBRE',
        'model_file': 'timbre-discogs-effnet-1.pb',
        'metadata_file': 'timbre-discogs-effnet-1.json',
        'pb_url': f'{_BASE}/timbre/timbre-discogs-effnet-1.pb',
        'json_url': f'{_BASE}/timbre/timbre-discogs-effnet-1.json',
        'input_layer': None,
        'output_layer': 'model/Softmax',
        'activation': 'softmax',
        'multi_label': False,
    },
    'approachability': {
        'display_name': 'Approachability',
        'category': 'Context',
        'tag_field': 'APPROACHABILITY',
        'model_file': 'approachability_2c-discogs-effnet-1.pb',
        'metadata_file': 'approachability_2c-discogs-effnet-1.json',
        'pb_url': f'{_BASE}/approachability/approachability_2c-discogs-effnet-1.pb',
        'json_url': f'{_BASE}/approachability/approachability_2c-discogs-effnet-1.json',
        'input_layer': None,
        'output_layer': 'model/Softmax',
        'activation': 'softmax',
        'multi_label': False,
    },
    'engagement': {
        'display_name': 'Engagement',
        'category': 'Context',
        'tag_field': 'ENGAGEMENT',
        'model_file': 'engagement_2c-discogs-effnet-1.pb',
        'metadata_file': 'engagement_2c-discogs-effnet-1.json',
        'pb_url': f'{_BASE}/engagement/engagement_2c-discogs-effnet-1.pb',
        'json_url': f'{_BASE}/engagement/engagement_2c-discogs-effnet-1.json',
        'input_layer': None,
        'output_layer': 'model/Softmax',
        'activation': 'softmax',
        'multi_label': False,
    },

    # ── Instrument ───────────────────────────────────────────────────────
    'mtg_jamendo_instrument': {
        'display_name': 'Instruments (MTG-Jamendo)',
        'category': 'Instrument',
        'tag_field': 'INSTRUMENT',
        'model_file': 'mtg_jamendo_instrument-discogs-effnet-1.pb',
        'metadata_file': 'mtg_jamendo_instrument-discogs-effnet-1.json',
        'pb_url': f'{_BASE}/mtg_jamendo_instrument/mtg_jamendo_instrument-discogs-effnet-1.pb',
        'json_url': f'{_BASE}/mtg_jamendo_instrument/mtg_jamendo_instrument-discogs-effnet-1.json',
        'input_layer': None,
        'output_layer': None,
        'activation': 'sigmoid',
        'multi_label': True,
    },

    # ── Auto-tagging ─────────────────────────────────────────────────────
    'mtg_jamendo_top50tags': {
        'display_name': 'Auto-tags (Jamendo Top50)',
        'category': 'Auto-tag',
        'tag_field': 'TAGS_JAMENDO',
        'model_file': 'mtg_jamendo_top50tags-discogs-effnet-1.pb',
        'metadata_file': 'mtg_jamendo_top50tags-discogs-effnet-1.json',
        'pb_url': f'{_BASE}/mtg_jamendo_top50tags/mtg_jamendo_top50tags-discogs-effnet-1.pb',
        'json_url': f'{_BASE}/mtg_jamendo_top50tags/mtg_jamendo_top50tags-discogs-effnet-1.json',
        'input_layer': None,
        'output_layer': None,
        'activation': 'sigmoid',
        'multi_label': True,
    },
    'mtt': {
        'display_name': 'Auto-tags (MagnaTagATune)',
        'category': 'Auto-tag',
        'tag_field': 'TAGS_MTT',
        'model_file': 'mtt-discogs-effnet-1.pb',
        'metadata_file': 'mtt-discogs-effnet-1.json',
        'pb_url': f'{_BASE}/mtt/mtt-discogs-effnet-1.pb',
        'json_url': f'{_BASE}/mtt/mtt-discogs-effnet-1.json',
        'input_layer': None,
        'output_layer': None,
        'activation': 'sigmoid',
        'multi_label': True,
    },
}

# Ordered list of categories for display
MODEL_CATEGORIES = ['Genre', 'Mood', 'Context', 'Instrument', 'Auto-tag']


def format_genre_tag(raw_genre, style='parent_child'):
    """Format genre tags for readability (handles '---' separator)."""
    if style == 'raw':
        return raw_genre
    
    if '---' in raw_genre:
        parts = raw_genre.split('---')
        parent = parts[0].strip()
        child = parts[1].strip() if len(parts) > 1 else ''
        
        if style == 'parent_child':
            return f"{parent} - {child}" if child else parent
        elif style == 'child_parent':
            return f"{child} - {parent}" if child else parent
        elif style == 'child_only':
            return child if child else parent
    
    return raw_genre


def format_label(raw_label):
    """Format any label for readability (capitalize words)."""
    return raw_label.replace('_', ' ').title()


# ─────────────────────────────────────────────────────────────────────────────
# DOWNLOAD MANAGER
# ─────────────────────────────────────────────────────────────────────────────

def get_downloaded_models():
    """Return set of model IDs that have both .pb and .json in MODEL_DIR."""
    downloaded = set()
    if not os.path.isdir(MODEL_DIR):
        return downloaded
    files_on_disk = set(os.listdir(MODEL_DIR))
    for model_id, info in MODEL_REGISTRY.items():
        if info['model_file'] in files_on_disk and info['metadata_file'] in files_on_disk:
            downloaded.add(model_id)
    return downloaded


def is_embedding_downloaded():
    """Check if the shared embedding model is present."""
    return os.path.isfile(os.path.join(MODEL_DIR, EMBEDDING_MODEL_FILE))


def _download_file(url, dest_path):
    """Download a file with progress indication."""
    filename = os.path.basename(dest_path)
    sys.stdout.write(f"     Downloading {filename}...")
    sys.stdout.flush()
    try:
        urllib.request.urlretrieve(url, dest_path)
        sys.stdout.write(" done\n")
    except urllib.error.URLError as e:
        sys.stdout.write(f" FAILED: {e}\n")
        raise


def download_embedding():
    """Download the embedding model if not already present."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    dest = os.path.join(MODEL_DIR, EMBEDDING_MODEL_FILE)
    if os.path.isfile(dest):
        return
    print("\n📦 Downloading embedding model (required, ~30MB)...")
    _download_file(EMBEDDING_MODEL_URL, dest)
    print("   ✅ Embedding model ready")


def download_models(model_ids):
    """Download specified models. Returns list of successfully downloaded IDs."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    # Always ensure embedding is present
    download_embedding()
    
    downloaded = []
    for model_id in model_ids:
        info = MODEL_REGISTRY[model_id]
        pb_path = os.path.join(MODEL_DIR, info['model_file'])
        json_path = os.path.join(MODEL_DIR, info['metadata_file'])
        
        try:
            if not os.path.isfile(pb_path):
                _download_file(info['pb_url'], pb_path)
            if not os.path.isfile(json_path):
                _download_file(info['json_url'], json_path)
            downloaded.append(model_id)
        except Exception as e:
            print(f"   ⚠️  Failed to download {info['display_name']}: {e}")
    
    return downloaded


def show_model_status():
    """Display which models are downloaded and which are not."""
    downloaded = get_downloaded_models()
    has_embedding = is_embedding_downloaded()
    
    print("\n" + "=" * 70)
    print("📦 MODEL STATUS")
    print("=" * 70)
    
    print(f"\n   Embedding model: {'✅ Downloaded' if has_embedding else '❌ Not downloaded'}")
    print(f"   Model directory: {MODEL_DIR}")
    print()
    
    for cat in MODEL_CATEGORIES:
        models_in_cat = [(mid, m) for mid, m in MODEL_REGISTRY.items() if m['category'] == cat]
        print(f"   {cat}:")
        for model_id, info in models_in_cat:
            status = '✅' if model_id in downloaded else '❌'
            print(f"     {status} {info['display_name']}")
        print()
    
    print(f"   Total: {len(downloaded)}/{len(MODEL_REGISTRY)} models downloaded")
    return downloaded


class Logger:
    """Dual output to console and log file"""
    def __init__(self, log_file):
        self.log_file = log_file
        self.file_handle = open(log_file, 'w', encoding='utf-8')
        self.write_header()
    
    def write_header(self):
        header = f"""
{'=' * 80}
ESSENTIA MUSIC TAGGER - LOG FILE
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'=' * 80}

"""
        self.file_handle.write(header)
        self.file_handle.flush()
    
    def log(self, message, console=True, file=True):
        if console:
            print(message)
        if file:
            self.file_handle.write(message + '\n')
            self.file_handle.flush()
    
    def log_config(self, config, music_path, selected_models):
        """Log configuration including selected models."""
        if isinstance(music_path, list):
            path_str = '\n                 '.join(music_path)
        else:
            path_str = music_path

        config_text = f"""
CONFIGURATION:
{'-' * 80}
Target Directory: {path_str}
Model Directory: {MODEL_DIR}
Selected Models: {', '.join(selected_models)}
"""
        if 'genre_discogs400' in selected_models:
            config_text += f"""Genre Settings:
  - Number of genres: {config.top_n_genres}
  - Confidence threshold: {config.genre_threshold:.1%}
  - Genre format: {config.genre_format}
"""
        config_text += f"""Multi-label threshold: {config.multi_label_threshold:.2%}
Other Settings:
  - Dry run mode: {config.dry_run}
  - Write confidence tags: {config.write_confidence_tags}
  - Overwrite existing: {config.overwrite_existing}
  - Verbose output: {config.verbose}
{'=' * 80}

"""
        self.file_handle.write(config_text)
        self.file_handle.flush()
    
    def log_analysis(self, filepath, results, relative_path):
        """Log detailed analysis results for one file."""
        log_entry = f"\nFILE: {relative_path}\n{'-' * 80}\n"
        
        for model_id, model_results in results.items():
            info = MODEL_REGISTRY[model_id]
            log_entry += f"\n  [{info['display_name']}] → {info['tag_field']}\n"
            if info['multi_label']:
                if model_results.get('tags'):
                    for t in model_results['tags']:
                        log_entry += f"    • {t['label']}: {t['confidence']:.2%}\n"
                else:
                    log_entry += "    (none above threshold)\n"
            else:
                if model_results.get('winner'):
                    w = model_results['winner']
                    log_entry += f"    Winner: {w['label']} ({w['confidence']:.2%})\n"
                    if model_results.get('all'):
                        for a in model_results['all']:
                            log_entry += f"    • {a['label']}: {a['confidence']:.2%}\n"
        
        log_entry += f"\n{'=' * 80}\n"
        self.file_handle.write(log_entry)
        self.file_handle.flush()
    
    def log_summary(self, processed, errors, skipped):
        summary = f"""
{'=' * 80}
PROCESSING SUMMARY
{'=' * 80}
Total Processed: {processed}
Errors: {errors}
Skipped: {skipped}
Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'=' * 80}
"""
        self.file_handle.write(summary)
        self.file_handle.flush()
    
    def close(self):
        self.file_handle.close()


SETTINGS_FILE = os.path.expanduser('~/.essentia_tagger.json')


def load_settings():
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {}


def save_settings(settings):
    try:
        with open(SETTINGS_FILE, 'w', encoding='utf-8') as f:
            json.dump(settings, f, indent=2)
    except IOError as e:
        print(f"   ⚠️  Could not save settings: {e}")


class Config:
    """Runtime configuration"""
    def __init__(self):
        self.dry_run = True
        self.top_n_genres = 3
        self.genre_threshold = 0.15
        self.multi_label_threshold = 0.005
        self.write_confidence_tags = True
        self.overwrite_existing = False
        self.verbose = True
        self.log_file = None
        self.genre_format = 'parent_child'
        self.default_library_path = None
        self.selected_models = []  # list of model IDs to run


class EssentiaAnalyzer:
    """Analyze audio files with selected Essentia models.
    
    All classifiers share a single embedding computed once per file.
    """
    
    def __init__(self, config, logger, selected_models):
        self.config = config
        self.logger = logger
        self.selected_models = selected_models  # list of model IDs
        
        logger.log("\n🔄 Loading models...")
        
        embedding_path = os.path.join(MODEL_DIR, EMBEDDING_MODEL_FILE)
        self.embedding_model = TensorflowPredictEffnetDiscogs(
            graphFilename=embedding_path,
            output="PartitionedCall:1"
        )
        logger.log("   ✅ Loaded embedding model")
        
        # Load each selected classifier
        self.classifiers = {}  # model_id -> (predict2d, labels)
        for model_id in selected_models:
            info = MODEL_REGISTRY[model_id]
            model_path = os.path.join(MODEL_DIR, info['model_file'])
            meta_path = os.path.join(MODEL_DIR, info['metadata_file'])
            
            try:
                kwargs = {'graphFilename': model_path}
                if info['input_layer']:
                    kwargs['input'] = info['input_layer']
                if info['output_layer']:
                    kwargs['output'] = info['output_layer']
                
                predict = TensorflowPredict2D(**kwargs)
                
                with open(meta_path, 'r') as f:
                    metadata = json.load(f)
                labels = metadata['classes']
                
                self.classifiers[model_id] = (predict, labels)
                logger.log(f"   ✅ Loaded {info['display_name']} ({len(labels)} classes)")
            except Exception as e:
                logger.log(f"   ⚠️  Could not load {info['display_name']}: {e}")
        
        logger.log(f"   ✅ {len(self.classifiers)} model(s) loaded successfully!\n")
    
    def analyze_file(self, filepath):
        """Analyze a single audio file with all selected models.
        
        Returns dict keyed by model_id, each containing:
          For multi_label models:
            'tags' – list of {'label', 'confidence'} above threshold
            'formatted_tags' – list of formatted label strings
          For softmax (binary) models:
            'winner' – {'label', 'confidence'} for winning class
            'all' – list of {'label', 'confidence'} for all classes
            'formatted_winner' – formatted winner label string
        """
        try:
            audio = MonoLoader(
                filename=str(filepath),
                sampleRate=16000,
                resampleQuality=4
            )()
            
            # Compute embeddings once
            embeddings = self.embedding_model(audio)
            
            results = {}
            
            for model_id, (predict, labels) in self.classifiers.items():
                info = MODEL_REGISTRY[model_id]
                
                predictions = predict(embeddings)
                activations = np.mean(predictions, axis=0)
                
                if info['multi_label']:
                    results[model_id] = self._process_multi_label(
                        model_id, activations, labels, info
                    )
                else:
                    results[model_id] = self._process_softmax(
                        model_id, activations, labels, info
                    )
            
            return results
            
        except Exception as e:
            self.logger.log(f"     ⚠️  Error analyzing: {e}")
            return None
    
    def _process_multi_label(self, model_id, activations, labels, info):
        """Process a multi-label (sigmoid) model's output."""
        result = {}
        
        if model_id == 'genre_discogs400':
            # Special handling: top-N with genre threshold + formatting
            top_indices = np.argsort(activations)[::-1][:self.config.top_n_genres * 2]
            tags = []
            for idx in top_indices:
                if len(tags) >= self.config.top_n_genres:
                    break
                if activations[idx] >= self.config.genre_threshold:
                    tags.append({
                        'label': labels[idx],
                        'confidence': float(activations[idx])
                    })
            
            # If no genres pass threshold, take top 1
            if not tags:
                top_idx = int(np.argmax(activations))
                tags.append({
                    'label': labels[top_idx],
                    'confidence': float(activations[top_idx])
                })
            
            result['tags'] = tags
            result['formatted_tags'] = [
                format_genre_tag(t['label'], style=self.config.genre_format) 
                for t in tags
            ]
            # Debug: store top 10
            all_top = np.argsort(activations)[::-1][:10]
            result['debug_top'] = [
                (labels[idx], float(activations[idx])) for idx in all_top
            ]
        else:
            # Generic multi-label: collect all above threshold
            tags = []
            for idx, act in enumerate(activations):
                if act >= self.config.multi_label_threshold:
                    tags.append({
                        'label': labels[idx],
                        'confidence': float(act)
                    })
            tags = sorted(tags, key=lambda x: x['confidence'], reverse=True)
            result['tags'] = tags[:10]  # limit to top 10
            result['formatted_tags'] = [format_label(t['label']) for t in result['tags']]
        
        return result
    
    def _process_softmax(self, model_id, activations, labels, info):
        """Process a 2-class softmax model's output."""
        all_classes = []
        for idx, act in enumerate(activations):
            all_classes.append({
                'label': labels[idx],
                'confidence': float(act)
            })
        all_classes.sort(key=lambda x: x['confidence'], reverse=True)
        
        winner = all_classes[0]
        return {
            'winner': winner,
            'all': all_classes,
            'formatted_winner': format_label(winner['label']),
        }


class TagWriter:
    """Write analysis results to audio file tags.
    
    Results dict is keyed by model_id.  Each model writes to its
    configured tag_field.  The GENRE tag_field gets special treatment
    (written to the native genre field); everything else is written as
    a custom tag / comment.
    """
    
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
    
    def write_tags(self, filepath, results):
        """Write all model results to file tags."""
        if self.config.dry_run:
            self._log_dry_run(results)
            return
        
        try:
            file_ext = filepath.suffix.lower()
            
            if file_ext == '.flac':
                self._write_flac(filepath, results)
            elif file_ext == '.mp3':
                self._write_mp3(filepath, results)
            elif file_ext in ('.ogg', '.oga'):
                self._write_ogg(filepath, results)
            elif file_ext == '.opus':
                self._write_opus(filepath, results)
            elif file_ext in ('.m4a', '.m4b', '.mp4', '.aac'):
                self._write_mp4(filepath, results)
            elif file_ext == '.wma':
                self._write_wma(filepath, results)
            elif file_ext in ('.aiff', '.aif'):
                self._write_aiff(filepath, results)
            elif file_ext in ('.wav', '.dsf'):
                self._write_id3_generic(filepath, results)
            elif file_ext in ('.wv', '.ape', '.mpc', '.mp+'):
                self._write_apev2(filepath, results)
            else:
                self.logger.log(f"     ⚠️  Unsupported format: {file_ext}")
                
        except Exception as e:
            self.logger.log(f"     ⚠️  Error writing tags: {e}")
    
    def _log_dry_run(self, results):
        """Show what tags would be written."""
        parts = []
        for model_id, model_results in results.items():
            info = MODEL_REGISTRY[model_id]
            if info['multi_label']:
                if model_results.get('formatted_tags'):
                    parts.append(f"{info['tag_field']}: {', '.join(model_results['formatted_tags'][:3])}")
            else:
                if model_results.get('formatted_winner'):
                    conf = model_results['winner']['confidence']
                    parts.append(f"{info['tag_field']}: {model_results['formatted_winner']} ({conf:.0%})")
        if parts:
            self.logger.log(f"     [DRY RUN] Would write: {' | '.join(parts)}")
    
    # ── Tag value helpers ────────────────────────────────────────────────
    
    def _build_tag_values(self, results):
        """Build a dict of {tag_field: value_string} from all model results."""
        tags = {}
        confidence_tags = {}
        
        for model_id, model_results in results.items():
            info = MODEL_REGISTRY[model_id]
            tag_field = info['tag_field']
            
            if info['multi_label']:
                if model_results.get('formatted_tags'):
                    tags[tag_field] = '; '.join(model_results['formatted_tags'])
                    if self.config.write_confidence_tags and model_results.get('tags'):
                        details = [f"{t['label']}: {t['confidence']:.2%}" for t in model_results['tags'][:5]]
                        confidence_tags[f"ESSENTIA_{tag_field}"] = f"Essentia: {', '.join(details)}"
            else:
                if model_results.get('formatted_winner'):
                    w = model_results['winner']
                    tags[tag_field] = model_results['formatted_winner']
                    if self.config.write_confidence_tags:
                        details = [f"{a['label']}: {a['confidence']:.2%}" for a in model_results['all']]
                        confidence_tags[f"ESSENTIA_{tag_field}"] = f"Essentia: {', '.join(details)}"
        
        return tags, confidence_tags
    
    # ── Format-specific writers ──────────────────────────────────────────
    
    def _write_flac(self, filepath, results):
        audio = FLAC(filepath)
        tags_written = self._write_vorbis_comments(audio, results)
        if tags_written:
            audio.save()
            self.logger.log(f"     ✅ Written tags: {', '.join(tags_written)}", console=False)
    
    def _write_mp3(self, filepath, results):
        try:
            audio = ID3(filepath)
        except Exception:
            audio = ID3()
        self._write_id3_tags(audio, results)
        audio.save(filepath)
    
    def _write_ogg(self, filepath, results):
        audio = OggVorbis(filepath)
        tags_written = self._write_vorbis_comments(audio, results)
        if tags_written:
            audio.save()
            self.logger.log(f"     ✅ Written tags: {', '.join(tags_written)}", console=False)
    
    def _write_opus(self, filepath, results):
        audio = OggOpus(filepath)
        tags_written = self._write_vorbis_comments(audio, results)
        if tags_written:
            audio.save()
            self.logger.log(f"     ✅ Written tags: {', '.join(tags_written)}", console=False)
    
    def _write_aiff(self, filepath, results):
        audio = AIFF(filepath)
        if audio.tags is None:
            audio.add_tags()
        self._write_id3_tags(audio.tags, results)
        audio.save()
    
    def _write_id3_generic(self, filepath, results):
        audio = mutagen.File(filepath)
        if audio is None:
            self.logger.log("     ⚠️  Could not open file for tagging")
            return
        if audio.tags is None:
            audio.add_tags()
        self._write_id3_tags(audio.tags, results)
        audio.save()
    
    def _write_vorbis_comments(self, audio, results):
        """Shared writer for Vorbis-comment formats (FLAC, OGG, Opus).
        
        Vorbis comments support arbitrary key=value, so we write each
        tag_field as its own key.
        """
        tags, confidence_tags = self._build_tag_values(results)
        tags_written = []
        
        for key, value in tags.items():
            if self.config.overwrite_existing or key not in audio:
                audio[key] = value
                tags_written.append(f"{key}={value}")
            else:
                self.logger.log(f"     ⏭️  Skipping {key} (already exists)")
        
        for key, value in confidence_tags.items():
            audio[key] = value
            tags_written.append(key)
        
        return tags_written
    
    def _write_id3_tags(self, tags, results):
        """Shared ID3v2 writer for MP3, AIFF, WAV, DSF.
        
        GENRE goes to TCON frame.  Everything else goes to TXXX frames.
        Confidence info goes to COMM frames.
        """
        tag_values, confidence_tags = self._build_tag_values(results)
        tags_written = []
        
        for key, value in tag_values.items():
            if key == 'GENRE':
                has_existing = bool(tags.getall('TCON'))
                if self.config.overwrite_existing or not has_existing:
                    tags.delall('TCON')
                    tags.add(TCON(encoding=3, text=value))
                    tags_written.append(f"TCON={value}")
                else:
                    self.logger.log("     ⏭️  Skipping GENRE (already has TCON)")
            else:
                # Write as TXXX frame with description = tag_field
                frame_key = f'TXXX:{key}'
                has_existing = bool(tags.getall(frame_key))
                if self.config.overwrite_existing or not has_existing:
                    tags.delall(frame_key)
                    tags.add(TXXX(encoding=3, desc=key, text=[value]))
                    tags_written.append(f"TXXX:{key}={value}")
                else:
                    self.logger.log(f"     ⏭️  Skipping {key} (already exists)")
        
        for key, value in confidence_tags.items():
            desc = key
            tags.add(COMM(encoding=3, lang='eng', desc=desc, text=value))
            tags_written.append(desc)
        
        if tags_written:
            self.logger.log(f"     ✅ Written tags: {', '.join(tags_written)}", console=False)
    
    def _write_mp4(self, filepath, results):
        """Write to MP4/M4A/AAC (iTunes-style atoms)."""
        audio = MP4(filepath)
        tag_values, confidence_tags = self._build_tag_values(results)
        tags_written = []
        
        for key, value in tag_values.items():
            if key == 'GENRE':
                has_existing = '\xa9gen' in audio.tags if audio.tags else False
                if self.config.overwrite_existing or not has_existing:
                    audio['\xa9gen'] = [value]
                    tags_written.append(f"genre={value}")
                else:
                    self.logger.log("     ⏭️  Skipping GENRE (already exists)")
            else:
                # Custom freeform atom
                atom_key = f'----:com.apple.iTunes:{key}'
                audio[atom_key] = [
                    mutagen.mp4.MP4FreeForm(
                        value.encode('utf-8'),
                        dataformat=mutagen.mp4.AtomDataType.UTF8
                    )
                ]
                tags_written.append(f"{key}={value}")
        
        for key, value in confidence_tags.items():
            atom_key = f'----:com.apple.iTunes:{key}'
            audio[atom_key] = [
                mutagen.mp4.MP4FreeForm(
                    value.encode('utf-8'),
                    dataformat=mutagen.mp4.AtomDataType.UTF8
                )
            ]
            tags_written.append(key)
        
        if tags_written:
            audio.save()
            self.logger.log(f"     ✅ Written tags: {', '.join(tags_written)}", console=False)
    
    def _write_wma(self, filepath, results):
        """Write to WMA/ASF tags."""
        audio = ASF(filepath)
        tag_values, confidence_tags = self._build_tag_values(results)
        tags_written = []
        
        for key, value in tag_values.items():
            if key == 'GENRE':
                has_existing = 'WM/Genre' in audio if audio.tags else False
                if self.config.overwrite_existing or not has_existing:
                    audio['WM/Genre'] = value
                    tags_written.append(f"WM/Genre={value}")
                else:
                    self.logger.log("     ⏭️  Skipping GENRE (already exists)")
            elif key == 'MOOD':
                audio['WM/Mood'] = value
                tags_written.append(f"WM/Mood={value}")
            else:
                audio[key] = value
                tags_written.append(f"{key}={value}")
        
        for key, value in confidence_tags.items():
            audio[key] = value
            tags_written.append(key)
        
        if tags_written:
            audio.save()
            self.logger.log(f"     ✅ Written tags: {', '.join(tags_written)}", console=False)
    
    def _write_apev2(self, filepath, results):
        """Write APEv2 tags (WavPack, Monkey's Audio, Musepack)."""
        try:
            audio = mutagen.File(filepath)
            if audio is None:
                self.logger.log("     ⚠️  Could not open file for tagging")
                return
            if audio.tags is None:
                audio.add_tags()
        except Exception:
            self.logger.log("     ⚠️  Could not read/create APEv2 tags")
            return
        
        tag_values, confidence_tags = self._build_tag_values(results)
        tags_written = []
        
        for key, value in tag_values.items():
            tag_name = 'Genre' if key == 'GENRE' else key
            has_existing = tag_name in audio.tags
            if self.config.overwrite_existing or not has_existing:
                audio.tags[tag_name] = value
                tags_written.append(f"{tag_name}={value}")
            else:
                self.logger.log(f"     ⏭️  Skipping {tag_name} (already exists)")
        
        for key, value in confidence_tags.items():
            audio.tags[key] = value
            tags_written.append(key)
        
        if tags_written:
            audio.save()
            self.logger.log(f"     ✅ Written tags: {', '.join(tags_written)}", console=False)


def scan_library(root_path, analyzer, tag_writer, config, logger):
    """Recursively scan and process music library."""
    root = Path(root_path)
    
    logger.log("\n🔍 Scanning for audio files...")
    files = list(root.rglob('*'))
    audio_files = [f for f in files if f.suffix.lower() in AUDIO_EXTENSIONS]
    
    if not audio_files:
        logger.log("❌ No audio files found in this directory!")
        return
    
    logger.log(f"🎵 Found {len(audio_files)} audio files")
    logger.log(f"{'=' * 70}\n")
    
    processed = 0
    skipped = 0
    errors = 0
    
    for i, filepath in enumerate(audio_files, 1):
        try:
            relative_path = filepath.relative_to(root)
        except ValueError:
            relative_path = filepath.name
        
        logger.log(f"[{i}/{len(audio_files)}] {relative_path}")
        
        results = analyzer.analyze_file(filepath)
        
        if results:
            # Print results to console
            for model_id, model_results in results.items():
                info = MODEL_REGISTRY[model_id]
                if info['multi_label']:
                    if model_results.get('formatted_tags'):
                        tags_str = ', '.join(model_results['formatted_tags'][:5])
                        logger.log(f"     🏷️  {info['display_name']}: {tags_str}")
                else:
                    if model_results.get('formatted_winner'):
                        w = model_results['winner']
                        logger.log(f"     🏷️  {info['display_name']}: {model_results['formatted_winner']} ({w['confidence']:.0%})")
            
            # Log to file
            logger.log_analysis(filepath, results, relative_path)
            
            # Write tags
            tag_writer.write_tags(filepath, results)
            
            if not config.dry_run:
                logger.log("     ✅ Tags written")
            
            processed += 1
        else:
            errors += 1
        
        logger.log("")
    
    logger.log(f"\n{'=' * 70}")
    logger.log(f"📊 SUMMARY")
    logger.log(f"{'=' * 70}")
    logger.log(f"✅ Processed: {processed}")
    logger.log(f"❌ Errors: {errors}")
    logger.log(f"⏭️  Skipped: {skipped}")
    
    logger.log_summary(processed, errors, skipped)


def _read_key():
    """Read a single keypress, returning special keys as names.
    Returns: str - single char or 'up', 'down', 'enter', 'backspace', 'q'
    """
    if platform.system() == 'Windows':
        import msvcrt
        ch = msvcrt.getwch()
        if ch in ('\r', '\n'):
            return 'enter'
        if ch == '\x08' or ch == '\x7f':
            return 'backspace'
        if ch in ('\x00', '\xe0'):  # special key prefix on Windows
            ch2 = msvcrt.getwch()
            if ch2 == 'H':
                return 'up'
            if ch2 == 'P':
                return 'down'
            return None
        return ch
    else:
        import tty
        import termios
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
            if ch == '\r' or ch == '\n':
                return 'enter'
            if ch == '\x7f' or ch == '\x08':
                return 'backspace'
            if ch == '\x1b':  # escape sequence
                ch2 = sys.stdin.read(1)
                if ch2 == '[':
                    ch3 = sys.stdin.read(1)
                    if ch3 == 'A':
                        return 'up'
                    if ch3 == 'B':
                        return 'down'
                return None
            return ch
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)


def _clear_lines(n):
    """Move cursor up n lines and clear them."""
    for _ in range(n):
        sys.stdout.write('\x1b[A')   # move up
        sys.stdout.write('\x1b[2K')  # clear line
    sys.stdout.flush()


def browse_directory(start_path):
    """Interactive directory browser with arrow-key navigation and multi-select.
    
    Args:
        start_path: Root directory to start browsing from
    
    Returns:
        list[str] - one or more selected directory paths, or None if cancelled
    """
    current_path = Path(start_path)
    selected_idx = 0
    page_size = 15
    scroll_offset = 0
    selected_set = []  # ordered list of selected folder paths (str)
    
    while True:
        # Get subdirectories of current path
        try:
            subdirs = sorted(
                [d for d in current_path.iterdir() if d.is_dir()],
                key=lambda d: d.name.lower()
            )
        except PermissionError:
            print("\n   ⚠️  Permission denied. Going back...")
            current_path = current_path.parent
            selected_idx = 0
            scroll_offset = 0
            continue
        
        # Build menu items
        items = []
        if selected_set:
            action_label = f"✅ DONE — {len(selected_set)} folder(s) selected"
        else:
            action_label = "✅ SELECT THIS FOLDER"
        items.append((action_label, 'select'))
        if current_path != Path(start_path):
            items.append(('⬆️  ../ (go up)', 'up'))
        for d in subdirs:
            marker = " [✓]" if str(d) in selected_set else ""
            items.append((f"📁 {d.name}{marker}", str(d)))
        
        # Clamp selection
        if selected_idx >= len(items):
            selected_idx = len(items) - 1
        if selected_idx < 0:
            selected_idx = 0
        
        # Adjust scroll so selected item is visible
        if selected_idx < scroll_offset:
            scroll_offset = selected_idx
        if selected_idx >= scroll_offset + page_size:
            scroll_offset = selected_idx - page_size + 1
        
        visible_items = items[scroll_offset:scroll_offset + page_size]
        
        # Render
        lines = []
        rel_path = str(current_path)
        try:
            rel_path = str(current_path.relative_to(start_path))
            if rel_path == '.':
                rel_path = '(library root)'
            else:
                rel_path = f"/{rel_path}"
        except ValueError:
            pass
        
        lines.append(f"\n   📂 Browsing: {rel_path}")
        lines.append(f"   📍 Full path: {current_path}")
        lines.append("   ↑↓ navigate | Enter = open folder | Space = select/deselect | 'q' cancel")
        lines.append("   " + "─" * 50)
        
        for i, (label, _action) in enumerate(visible_items):
            global_idx = i + scroll_offset
            if global_idx == selected_idx:
                lines.append(f"   ▶ {label}")
            else:
                lines.append(f"     {label}")
        
        if scroll_offset > 0:
            lines.append(f"   ↑ ({scroll_offset} more above)")
        remaining_below = len(items) - scroll_offset - page_size
        if remaining_below > 0:
            lines.append(f"   ↓ ({remaining_below} more below)")
        
        lines.append("")
        
        output = '\n'.join(lines)
        sys.stdout.write(output)
        sys.stdout.flush()
        
        # Read key
        key = _read_key()
        
        # Clear the rendered block before re-rendering
        line_count = len(lines)
        _clear_lines(line_count)
        
        if key == 'up':
            if selected_idx > 0:
                selected_idx -= 1
        elif key == 'down':
            if selected_idx < len(items) - 1:
                selected_idx += 1
        elif key == ' ':
            # Space toggles selection on a folder item (not on 'select' or 'up')
            _label, action = items[selected_idx]
            if action not in ('select', 'up'):
                if action in selected_set:
                    selected_set.remove(action)
                else:
                    selected_set.append(action)
        elif key == 'enter':
            _label, action = items[selected_idx]
            if action == 'select':
                if selected_set:
                    return selected_set  # multi-select confirmed
                else:
                    return [str(current_path)]  # single: current folder
            elif action == 'up':
                current_path = current_path.parent
                selected_idx = 0
                scroll_offset = 0
            else:
                # Navigate into subfolder
                current_path = Path(action)
                selected_idx = 0
                scroll_offset = 0
        elif key == 'q':
            return None
        elif key == 'backspace':
            if current_path != Path(start_path):
                current_path = current_path.parent
                selected_idx = 0
                scroll_offset = 0


# ─────────────────────────────────────────────────────────────────────────────
# MODEL DOWNLOAD / SELECTION TUI
# ─────────────────────────────────────────────────────────────────────────────

def prompt_download_models():
    """Show model status and let the user download additional models.
    
    Returns set of model IDs that are now available on disk.
    """
    downloaded = show_model_status()
    
    not_downloaded = [mid for mid in MODEL_REGISTRY if mid not in downloaded]
    
    if not not_downloaded:
        print("   All models are already downloaded!\n")
        return downloaded
    
    print(f"\n   {len(not_downloaded)} model(s) available to download.")
    choice = input("   Download additional models? [y/N]: ").strip().lower()
    if choice not in ('y', 'yes'):
        return downloaded
    
    # Multi-select which models to download
    print("\n   Select models to download (enter numbers separated by commas, or 'all'):")
    for i, mid in enumerate(not_downloaded, 1):
        info = MODEL_REGISTRY[mid]
        print(f"     {i:2d}. [{info['category']}] {info['display_name']}")
    
    selection = input("\n   Models to download [all]: ").strip().lower()
    
    if selection in ('', 'all'):
        to_download = not_downloaded
    else:
        to_download = []
        for part in selection.split(','):
            part = part.strip()
            try:
                idx = int(part) - 1
                if 0 <= idx < len(not_downloaded):
                    to_download.append(not_downloaded[idx])
            except ValueError:
                pass
    
    if not to_download:
        print("   No models selected.")
        return downloaded
    
    print(f"\n   Downloading {len(to_download)} model(s)...")
    newly_downloaded = download_models(to_download)
    downloaded.update(newly_downloaded)
    
    print(f"   ✅ {len(newly_downloaded)} model(s) downloaded successfully.")
    if newly_downloaded:
        print(f"   Total models available: {len(downloaded)}/{len(MODEL_REGISTRY)}")
    
    return downloaded


def select_models_interactive(downloaded_models):
    """Interactive multi-select for which models to run.
    
    Uses arrow keys + space for selection, grouped by category.
    
    Args:
        downloaded_models: set of model IDs that are available
    
    Returns:
        list of selected model IDs, or None if cancelled
    """
    if not downloaded_models:
        print("\n   ❌ No models are downloaded. Please download models first.")
        return None
    
    # Build ordered items list: (model_id, display_text, category_header_or_None)
    items = []  # list of (model_id, display_text)
    category_headers = {}  # index -> category name (for display)
    
    for cat in MODEL_CATEGORIES:
        models_in_cat = [
            (mid, MODEL_REGISTRY[mid]) 
            for mid in MODEL_REGISTRY 
            if MODEL_REGISTRY[mid]['category'] == cat and mid in downloaded_models
        ]
        if models_in_cat:
            category_headers[len(items)] = cat
            for mid, info in models_in_cat:
                items.append((mid, info['display_name']))
    
    if not items:
        print("\n   ❌ No downloaded models available.")
        return None
    
    selected = set(mid for mid, _ in items)  # all selected by default
    cursor = 0
    page_size = 20
    scroll_offset = 0
    
    while True:
        # Figure out which line indices are category headers
        # We need to build a display list that intersperses headers
        display_lines = []
        item_idx_map = {}  # display_line_index -> items_index
        header_lines = set()
        
        for i, (mid, display) in enumerate(items):
            if i in category_headers:
                display_lines.append(f"── {category_headers[i]} ──")
                header_lines.add(len(display_lines) - 1)
            
            marker = '[✓]' if mid in selected else '[ ]'
            display_lines.append(f"  {marker} {display}")
            item_idx_map[len(display_lines) - 1] = i
        
        # Translate cursor (item index) to display line index
        cursor_display = None
        for dl_idx, item_i in item_idx_map.items():
            if item_i == cursor:
                cursor_display = dl_idx
                break
        
        if cursor_display is None:
            cursor_display = 0
        
        # Scroll
        if cursor_display < scroll_offset:
            scroll_offset = cursor_display
        if cursor_display >= scroll_offset + page_size:
            scroll_offset = cursor_display - page_size + 1
        
        visible = display_lines[scroll_offset:scroll_offset + page_size]
        
        lines = []
        lines.append(f"\n   🎯 SELECT MODELS TO RUN ({len(selected)}/{len(items)} selected)")
        lines.append("   ↑↓ navigate | Space = toggle | a = all | n = none | Enter = confirm | q = cancel")
        lines.append("   " + "─" * 55)
        
        for vi, line_text in enumerate(visible):
            global_idx = vi + scroll_offset
            if global_idx == cursor_display:
                lines.append(f"   ▶{line_text}")
            elif global_idx in header_lines:
                lines.append(f"    {line_text}")
            else:
                lines.append(f"    {line_text}")
        
        if scroll_offset > 0:
            lines.append(f"   ↑ ({scroll_offset} more above)")
        remaining = len(display_lines) - scroll_offset - page_size
        if remaining > 0:
            lines.append(f"   ↓ ({remaining} more below)")
        
        lines.append("")
        
        output = '\n'.join(lines)
        sys.stdout.write(output)
        sys.stdout.flush()
        
        key = _read_key()
        _clear_lines(len(lines))
        
        if key == 'up':
            if cursor > 0:
                cursor -= 1
        elif key == 'down':
            if cursor < len(items) - 1:
                cursor += 1
        elif key == ' ':
            mid, _ = items[cursor]
            if mid in selected:
                selected.discard(mid)
            else:
                selected.add(mid)
        elif key == 'a':
            selected = set(mid for mid, _ in items)
        elif key == 'n':
            selected.clear()
        elif key == 'enter':
            if not selected:
                # Show brief message then continue selection
                print("   ⚠️  Please select at least one model.")
                continue
            return [mid for mid, _ in items if mid in selected]
        elif key == 'q':
            return None
    
    return None


def get_music_path(config):
    """Prompt user for music directory path, with optional library browsing"""
    print("\n" + "=" * 70)
    print("🎸 ESSENTIA MUSIC TAGGER - INTERACTIVE MODE")
    print("=" * 70)
    print("\nThis tool will recursively analyze ALL audio files")
    print("in the directory you specify and its subdirectories.\n")
    
    library_path = config.default_library_path
    
    if library_path and not os.path.isdir(library_path):
        print(f"⚠️  Default library path no longer exists: {library_path}")
        library_path = None
        print()
    
    if not library_path:
        # No library path set yet — offer to set one now
        print("💡 TIP: You can set a default library path for quick access on future runs.")
        set_now = input("   Set a default library path now? [y/N]: ").strip().lower()
        if set_now in ('y', 'yes'):
            new_path = input("   Enter library path: ").strip().strip('\'"')
            new_path = os.path.expanduser(new_path)
            if os.path.isdir(new_path):
                saved = load_settings()
                saved['default_library_path'] = new_path
                save_settings(saved)
                config.default_library_path = new_path
                library_path = new_path
                print(f"   ✅ Library path saved: {library_path}")
            else:
                print(f"   ❌ Path does not exist: {new_path}")
        print()
    
    # Show library scan options if a library path is known
    if library_path and os.path.isdir(library_path):
        print(f"📚 Default library: {library_path}")
        print()
        print("How would you like to choose the scan path?")
        print("   1 = Scan entire library (default)")
        print("   2 = Browse & select a folder within library")
        print("   3 = Enter a custom path")
        print("   4 = Change/clear default library path")
        print()
        while True:
            choice = input("Select option [1]: ").strip()
            if choice in ('', '1'):
                path = Path(library_path)
                sample_files = list(path.rglob('*'))
                audio_count = len([f for f in sample_files if f.suffix.lower() in AUDIO_EXTENSIONS])
                print(f"\n📂 Directory: {path}")
                print(f"🎵 Found ~{audio_count} audio files")
                confirm = input("\nProceed with this directory? [Y/n]: ").strip().lower()
                if confirm in ('', 'y', 'yes'):
                    return [str(path)]
                else:
                    print("Cancelled.\n")
                    continue
            elif choice == '2':
                print("\n📂 Opening folder browser...")
                selected = browse_directory(library_path)  # list[str] or None
                if selected:
                    total_audio = sum(
                        len([f for f in Path(p).rglob('*') if f.suffix.lower() in AUDIO_EXTENSIONS])
                        for p in selected
                    )
                    if len(selected) == 1:
                        print(f"\n📂 Selected: {selected[0]}")
                    else:
                        print(f"\n📂 Selected {len(selected)} folders:")
                        for p in selected:
                            print(f"   • {p}")
                    print(f"🎵 Found ~{total_audio} audio files")
                    confirm = input("\nProceed with this selection? [Y/n]: ").strip().lower()
                    if confirm in ('', 'y', 'yes'):
                        return selected
                    else:
                        print("Cancelled. Let's try again.\n")
                        continue
                else:
                    print("\nBrowsing cancelled. Let's try again.\n")
                    continue
            elif choice == '3':
                break  # Fall through to manual path entry
            elif choice == '4':
                print(f"\n   Current: {library_path}")
                print("   c = Change path  |  x = Clear/remove  |  Enter = Cancel")
                mgmt = input("   Action: ").strip().lower()
                if mgmt == 'c':
                    new_path = input("   New library path: ").strip().strip('\'\'"')
                    new_path = os.path.expanduser(new_path)
                    if os.path.isdir(new_path):
                        s = load_settings()
                        s['default_library_path'] = new_path
                        save_settings(s)
                        config.default_library_path = new_path
                        library_path = new_path
                        print(f"   ✅ Saved: {library_path}")
                    else:
                        print(f"   ❌ Does not exist: {new_path}")
                elif mgmt == 'x':
                    s = load_settings()
                    s.pop('default_library_path', None)
                    save_settings(s)
                    config.default_library_path = None
                    print("   ✅ Library path cleared")
                    break  # Fall through to manual path entry
                print()
                continue
            else:
                print("   ⚠️  Please enter 1, 2, 3, or 4")
    
    # Manual path entry (original flow)
    print("Example paths:")
    print("  • /srv/.../Music/Sources/Clean/2Pac")
    print("  • /srv/.../Music/Sources/Clean/2Pac/Me Against the World")
    print("  • /srv/.../Music/Sources/Clean")
    print()
    
    while True:
        path_input = input("Enter the path to analyze (or 'q' to quit): ").strip()
        
        if path_input.lower() in ['q', 'quit', 'exit']:
            print("👋 Exiting...")
            sys.exit(0)
        
        # Expand ~ and handle quotes
        path_input = path_input.strip('\'"')
        path = Path(os.path.expanduser(path_input))
        
        if not path.exists():
            print(f"❌ Path does not exist: {path}")
            print("Please try again.\n")
            continue
        
        if not path.is_dir():
            print(f"❌ Path is not a directory: {path}")
            print("Please try again.\n")
            continue
        
        # Preview what will be scanned
        sample_files = list(path.rglob('*'))
        audio_count = len([f for f in sample_files if f.suffix.lower() in AUDIO_EXTENSIONS])
        
        print(f"\n📂 Directory: {path}")
        print(f"🎵 Found ~{audio_count} audio files")
        
        confirm = input("\nProceed with this directory? [Y/n]: ").strip().lower()
        if confirm in ['', 'y', 'yes']:
            return [str(path)]
        else:
            print("Cancelled. Let's try again.\n")


def get_int_input(prompt, default, min_val=None, max_val=None):
    """Get integer input with validation"""
    while True:
        user_input = input(f"{prompt} [{default}]: ").strip()
        if not user_input:
            return default
        try:
            value = int(user_input)
            if min_val is not None and value < min_val:
                print(f"   ⚠️  Must be at least {min_val}")
                continue
            if max_val is not None and value > max_val:
                print(f"   ⚠️  Must be at most {max_val}")
                continue
            return value
        except ValueError:
            print("   ⚠️  Please enter a valid number")


def get_float_input(prompt, default, min_val=None, max_val=None):
    """Get float input with validation"""
    while True:
        user_input = input(f"{prompt} [{default}]: ").strip()
        if not user_input:
            return default
        try:
            value = float(user_input)
            if min_val is not None and value < min_val:
                print(f"   ⚠️  Must be at least {min_val}")
                continue
            if max_val is not None and value > max_val:
                print(f"   ⚠️  Must be at most {max_val}")
                continue
            return value
        except ValueError:
            print("   ⚠️  Please enter a valid number")


def get_yes_no(prompt, default=True):
    """Get yes/no input"""
    default_str = "Y/n" if default else "y/N"
    user_input = input(f"{prompt} [{default_str}]: ").strip().lower()
    if not user_input:
        return default
    return user_input in ['y', 'yes']


def configure_settings(selected_models):
    """Interactive configuration (adapts to selected models)."""
    config = Config()
    config.selected_models = selected_models
    
    saved = load_settings()
    config.default_library_path = saved.get('default_library_path')
    
    print("\n" + "=" * 70)
    print("⚙️  CONFIGURATION")
    print("=" * 70)
    print("\nPress Enter to accept defaults shown in [brackets]\n")

    # Dry run mode
    print("─" * 70)
    print("🧪 DRY RUN MODE")
    print("   Test mode - analyzes files but doesn't write tags")
    print("   Recommended: Enable for first run to see results")
    config.dry_run = get_yes_no("Enable dry run mode?", default=True)
    
    # Genre settings (only if genre_discogs400 is selected)
    if 'genre_discogs400' in selected_models:
        print("\n" + "─" * 70)
        print("🎸 GENRE (DISCOGS 400) SETTINGS")
        print("   How many genre tags to write per song")
        print("   • 1 = Only top genre")
        print("   • 3 = Balanced (good variety)")
        print("   • 5 = Comprehensive (may include less relevant)")
        config.top_n_genres = get_int_input("Number of genres to write", default=3, min_val=1, max_val=10)
        
        print("\n   Genre confidence threshold (as percentage)")
        print("   • 5% = Very inclusive  |  15% = Balanced  |  25% = Strict")
        threshold_pct = get_float_input("Genre threshold (%)", default=15, min_val=1, max_val=50)
        config.genre_threshold = threshold_pct / 100.0
        
        print("\n   Genre tag formatting ('Rock---Alternative Rock')")
        print("   • 1 = 'Rock - Alternative Rock'  (parent - child)")
        print("   • 2 = 'Alternative Rock - Rock'  (child - parent)")
        print("   • 3 = 'Alternative Rock'          (child only)")
        print("   • 4 = 'Rock---Alternative Rock'   (raw)")
        format_choice = get_int_input("Genre format", default=1, min_val=1, max_val=4)
        config.genre_format = {1: 'parent_child', 2: 'child_parent', 3: 'child_only', 4: 'raw'}[format_choice]
    
    # Multi-label threshold (for any multi-label model except genre_discogs400)
    has_multi_label = any(
        MODEL_REGISTRY[mid]['multi_label'] and mid != 'genre_discogs400'
        for mid in selected_models
    )
    if has_multi_label:
        print("\n" + "─" * 70)
        print("🏷️  MULTI-LABEL THRESHOLD")
        print("   Confidence threshold for multi-label models (mood/theme, instrument, etc.)")
        print("   These typically have much lower confidence scores than genre.")
        print("   • 0.1% = Very inclusive  |  0.5% = Balanced  |  1% = Strict")
        ml_pct = get_float_input("Multi-label threshold (%)", default=0.5, min_val=0.01, max_val=20)
        config.multi_label_threshold = ml_pct / 100.0
    
    # Write confidence scores
    print("\n" + "─" * 70)
    print("📊 CONFIDENCE SCORES")
    print("   Write confidence percentages to additional tags")
    config.write_confidence_tags = get_yes_no("Write confidence score tags?", default=True)
    
    # Overwrite existing
    print("\n" + "─" * 70)
    print("♻️  EXISTING TAGS")
    print("   What to do if files already have existing tags")
    config.overwrite_existing = get_yes_no("Overwrite existing tags?", default=False)
    
    # Verbose output
    print("\n" + "─" * 70)
    print("📢 VERBOSE OUTPUT")
    config.verbose = get_yes_no("Enable verbose output?", default=True)
    
    return config


def display_config_summary(config, music_path, selected_models):
    """Display final configuration before processing."""
    print("\n" + "=" * 70)
    print("📋 FINAL SETTINGS")
    print("=" * 70)
    if isinstance(music_path, list) and len(music_path) > 1:
        print(f"📂 Target folders ({len(music_path)}):")
        for p in music_path:
            print(f"   • {p}")
    else:
        target = music_path[0] if isinstance(music_path, list) else music_path
        print(f"📂 Target directory: {target}")
    print(f"📁 Model directory: {MODEL_DIR}")
    
    print(f"\n🎯 Selected models ({len(selected_models)}):")
    for mid in selected_models:
        info = MODEL_REGISTRY[mid]
        print(f"   • {info['display_name']} → {info['tag_field']}")
    
    if 'genre_discogs400' in selected_models:
        print(f"\n🎸 Genre Settings:")
        print(f"   • Number of genres: {config.top_n_genres}")
        print(f"   • Confidence threshold: {config.genre_threshold:.2%}")
        print(f"   • Format style: {config.genre_format}")
    
    print(f"\n📊 Other Settings:")
    print(f"   • Multi-label threshold: {config.multi_label_threshold:.2%}")
    print(f"   • Dry run mode: {config.dry_run}")
    print(f"   • Write confidence tags: {config.write_confidence_tags}")
    print(f"   • Overwrite existing: {config.overwrite_existing}")
    print(f"   • Verbose output: {config.verbose}")
    
    if config.dry_run:
        print(f"\n⚠️  DRY RUN MODE - No files will be modified!")
    else:
        print(f"\n⚠️  LIVE MODE - Files WILL be modified!")
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    config.log_file = f"essentia_tagger_{timestamp}.log"
    print(f"\n📝 Log file: {config.log_file}")
    
    print("=" * 70 + "\n")
    
    if not config.dry_run:
        confirm = input("Ready to proceed? [Y/n]: ").strip().lower()
        if confirm not in ['', 'y', 'yes']:
            print("Cancelled.")
            sys.exit(0)


def parse_arguments():
    """Parse command-line arguments for automated/non-interactive mode."""
    all_model_ids = ', '.join(MODEL_REGISTRY.keys())
    parser = argparse.ArgumentParser(
        description='Analyze music files with Essentia and write metadata tags',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  # Interactive mode (no arguments)
  python tag_music.py
  
  # Automated mode — all downloaded models
  python tag_music.py /path/to/music --auto
  
  # Automated mode — specific models only
  python tag_music.py /path/to/music --auto --models genre_discogs400 mood_happy danceability
  
  # List available model IDs
  python tag_music.py --list-models
  
  # Watch a specific file (for file watcher integration)
  python tag_music.py /path/to/song.flac --auto --single-file
  
  # Dry run to test
  python tag_music.py /path/to/music --auto --dry-run

Available model IDs:
  {all_model_ids}

Genre format styles:
  parent_child: "Rock - Alternative Rock" (default)
  child_parent: "Alternative Rock - Rock"
  child_only:   "Alternative Rock"
  raw:          "Rock---Alternative Rock"
"""
    )
    
    parser.add_argument(
        'path', nargs='?',
        help='Path to music file or directory to analyze'
    )
    parser.add_argument(
        '--auto', '-a', action='store_true',
        help='Run in automated (non-interactive) mode'
    )
    parser.add_argument(
        '--single-file', '-f', action='store_true',
        help='Process a single file (for file watcher integration)'
    )
    parser.add_argument(
        '--models', '-m', nargs='+', default=None, metavar='ID',
        help='Model IDs to use (default: all downloaded models)'
    )
    parser.add_argument(
        '--list-models', action='store_true',
        help='List all available model IDs and exit'
    )
    parser.add_argument(
        '--download', nargs='*', default=None, metavar='ID',
        help='Download models. No IDs = download all. Otherwise list specific model IDs.'
    )
    parser.add_argument(
        '--genres', '-g', type=int, default=3, metavar='N',
        help='Number of Discogs genres to write (default: 3)'
    )
    parser.add_argument(
        '--genre-threshold', '-gt', type=float, default=15.0, metavar='PCT',
        help='Genre confidence threshold %% (default: 15)'
    )
    parser.add_argument(
        '--genre-format', '-gf',
        choices=['parent_child', 'child_parent', 'child_only', 'raw'],
        default='parent_child',
        help='Genre tag format style (default: parent_child)'
    )
    parser.add_argument(
        '--multi-label-threshold', '-mlt', type=float, default=0.5, metavar='PCT',
        help='Multi-label confidence threshold %% (default: 0.5)'
    )
    parser.add_argument(
        '--dry-run', '-d', action='store_true',
        help='Analyze files but do not write tags'
    )
    parser.add_argument(
        '--no-confidence-tags', action='store_true',
        help='Do not write confidence score tags'
    )
    parser.add_argument(
        '--overwrite', '-o', action='store_true',
        help='Overwrite existing tags'
    )
    parser.add_argument(
        '--quiet', '-q', action='store_true',
        help='Minimal output'
    )
    parser.add_argument(
        '--log-dir', type=str, default=None, metavar='DIR',
        help='Directory for log files (default: current directory)'
    )
    parser.add_argument(
        '--model-dir', type=str, default=None, metavar='DIR',
        help='Directory containing Essentia models (default: ~/essentia_models)'
    )
    parser.add_argument(
        '--library', type=str, default=None, metavar='DIR',
        help='Default music library path (saved for future runs)'
    )
    
    return parser.parse_args()


def config_from_args(args, selected_models):
    """Create Config object from command-line arguments."""
    config = Config()
    config.dry_run = args.dry_run
    config.top_n_genres = args.genres
    config.genre_threshold = args.genre_threshold / 100.0
    config.multi_label_threshold = args.multi_label_threshold / 100.0
    config.write_confidence_tags = not args.no_confidence_tags
    config.overwrite_existing = args.overwrite
    config.verbose = not args.quiet
    config.genre_format = args.genre_format
    config.selected_models = selected_models
    
    if args.library:
        lib_path = os.path.expanduser(args.library)
        if os.path.isdir(lib_path):
            saved = load_settings()
            saved['default_library_path'] = lib_path
            save_settings(saved)
            config.default_library_path = lib_path
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f"essentia_tagger_{timestamp}.log"
    if args.log_dir:
        log_dir = Path(args.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        config.log_file = str(log_dir / log_filename)
    else:
        config.log_file = log_filename
    
    return config


def resolve_models_for_auto(args):
    """Determine which models to use in automated mode."""
    downloaded = get_downloaded_models()
    
    if args.models:
        # User specified exact models
        selected = []
        for mid in args.models:
            if mid not in MODEL_REGISTRY:
                print(f"❌ Unknown model ID: {mid}")
                print(f"   Available: {', '.join(MODEL_REGISTRY.keys())}")
                sys.exit(1)
            if mid not in downloaded:
                print(f"❌ Model not downloaded: {mid}")
                print(f"   Run: python tag_music.py --download {mid}")
                sys.exit(1)
            selected.append(mid)
        return selected
    else:
        # Use all downloaded models
        if not downloaded:
            print("❌ No models downloaded. Run: python tag_music.py --download")
            sys.exit(1)
        # Return in registry order
        return [mid for mid in MODEL_REGISTRY if mid in downloaded]


def process_single_file(filepath, analyzer, tag_writer, config, logger):
    """Process a single audio file (for file watcher integration)."""
    filepath = Path(filepath)
    
    if not filepath.exists():
        logger.log(f"❌ File not found: {filepath}")
        return False
    
    if filepath.suffix.lower() not in AUDIO_EXTENSIONS:
        logger.log(f"⏭️ Skipping non-audio file: {filepath}")
        return False
    
    logger.log(f"🎵 Processing: {filepath.name}")
    
    results = analyzer.analyze_file(filepath)
    
    if results:
        for model_id, model_results in results.items():
            info = MODEL_REGISTRY[model_id]
            if info['multi_label']:
                if model_results.get('formatted_tags'):
                    logger.log(f"   🏷️  {info['display_name']}: {', '.join(model_results['formatted_tags'][:3])}")
            else:
                if model_results.get('formatted_winner'):
                    w = model_results['winner']
                    logger.log(f"   🏷️  {info['display_name']}: {model_results['formatted_winner']} ({w['confidence']:.0%})")
        
        logger.log_analysis(filepath, results, filepath.name)
        tag_writer.write_tags(filepath, results)
        
        if not config.dry_run:
            logger.log("   ✅ Tags written")
        
        return True
    else:
        logger.log(f"   ❌ Analysis failed")
        return False


def main():
    """Main entry point."""
    args = parse_arguments()
    logger = None
    
    # Handle --list-models
    if args.list_models:
        print("\nAvailable Essentia models:")
        downloaded = get_downloaded_models()
        for cat in MODEL_CATEGORIES:
            models = [(mid, m) for mid, m in MODEL_REGISTRY.items() if m['category'] == cat]
            if models:
                print(f"\n  {cat}:")
                for mid, info in models:
                    status = '✅' if mid in downloaded else '  '
                    print(f"    {status} {mid:30s}  {info['display_name']:30s}  → {info['tag_field']}")
        print(f"\n  ✅ = downloaded ({len(downloaded)}/{len(MODEL_REGISTRY)})")
        sys.exit(0)
    
    # Handle --download
    if args.download is not None:
        if len(args.download) == 0:
            # Download all
            to_download = list(MODEL_REGISTRY.keys())
        else:
            to_download = []
            for mid in args.download:
                if mid not in MODEL_REGISTRY:
                    print(f"❌ Unknown model ID: {mid}")
                    sys.exit(1)
                to_download.append(mid)
        
        print(f"📦 Downloading {len(to_download)} model(s)...")
        downloaded = download_models(to_download)
        print(f"✅ {len(downloaded)} model(s) downloaded successfully.")
        sys.exit(0)
    
    # Update model directory if specified
    global MODEL_DIR
    if args.model_dir:
        MODEL_DIR = os.path.expanduser(args.model_dir)
    
    if args.auto or args.single_file:
        # ── AUTOMATED MODE ──────────────────────────────────────────────
        if not args.path:
            print("❌ Error: Path is required in automated mode")
            print("   Use: python tag_music.py /path/to/music --auto")
            sys.exit(1)
        
        music_path = os.path.expanduser(args.path)
        selected_models = resolve_models_for_auto(args)
        config = config_from_args(args, selected_models)
        
        try:
            logger = Logger(config.log_file)
            logger.log_config(config, music_path, selected_models)
            
            mode_str = "DRY RUN" if config.dry_run else "LIVE"
            logger.log(f"🎸 Essentia Tagger [{mode_str}]")
            logger.log(f"   Path: {music_path}")
            logger.log(f"   Models: {', '.join(selected_models)}")
            logger.log("")
            
            analyzer = EssentiaAnalyzer(config, logger, selected_models)
            tag_writer = TagWriter(config, logger)
            
            if args.single_file:
                success = process_single_file(music_path, analyzer, tag_writer, config, logger)
                sys.exit(0 if success else 1)
            else:
                if not os.path.isdir(music_path):
                    logger.log(f"❌ Error: Not a directory: {music_path}")
                    sys.exit(1)
                scan_library(music_path, analyzer, tag_writer, config, logger)
            
            logger.log("\n✅ Processing complete!")
            logger.log(f"📝 Log: {config.log_file}")
            
        except KeyboardInterrupt:
            print("\n⚠️ Interrupted")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Error: {e}")
            sys.exit(1)
        finally:
            if logger:
                logger.close()
    
    else:
        # ── INTERACTIVE MODE ─────────────────────────────────────────────
        try:
            saved = load_settings()
            temp_config = Config()
            temp_config.default_library_path = saved.get('default_library_path')
            
            # Step 1: Show models & offer download
            downloaded = prompt_download_models()
            
            if not downloaded:
                print("\n❌ No models available. Please download at least one model.")
                print("   Run: python tag_music.py --download")
                sys.exit(1)
            
            # Step 2: Select which models to run
            selected_models = select_models_interactive(downloaded)
            if not selected_models:
                print("\n👋 Cancelled.")
                sys.exit(0)
            
            print(f"\n✅ Selected {len(selected_models)} model(s)")
            
            # Step 3: Get scan path
            music_paths = get_music_path(temp_config)
            
            # Step 4: Configure settings
            config = configure_settings(selected_models)
            
            # Step 5: Show summary and confirm
            display_config_summary(config, music_paths, selected_models)
            
            # Step 6: Process
            logger = Logger(config.log_file)
            logger.log_config(config, music_paths, selected_models)
            
            analyzer = EssentiaAnalyzer(config, logger, selected_models)
            tag_writer = TagWriter(config, logger)
            
            for music_path in music_paths:
                if len(music_paths) > 1:
                    logger.log(f"\n{'=' * 70}")
                    logger.log(f"📂 Processing: {music_path}")
                    logger.log(f"{'=' * 70}")
                scan_library(music_path, analyzer, tag_writer, config, logger)
            
            logger.log("\n" + "=" * 70)
            logger.log("✅ PROCESSING COMPLETE!")
            logger.log("=" * 70)
            logger.log(f"\n📝 Full log saved to: {config.log_file}")
            
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted by user. Exiting...")
            sys.exit(1)
        finally:
            if logger:
                logger.close()


if __name__ == '__main__':
    main()
