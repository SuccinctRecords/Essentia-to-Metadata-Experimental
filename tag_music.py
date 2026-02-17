#!/usr/bin/env python3
"""
Analyze music library with Essentia and write genre/mood to tags
Interactive version with logging and tag formatting
"""
import os
import json
import sys
import re
from pathlib import Path
from datetime import datetime
import numpy as np
from essentia.standard import MonoLoader, TensorflowPredictEffnetDiscogs, TensorflowPredict2D
import mutagen
from mutagen.flac import FLAC
from mutagen.mp3 import MP3
from mutagen.id3 import ID3, TCON, COMM

# Model directory (fixed)
MODEL_DIR = os.path.expanduser('~/essentia_models')

# Model files
EMBEDDING_MODEL = f"{MODEL_DIR}/discogs-effnet-bs64-1.pb"
GENRE_MODEL = f"{MODEL_DIR}/genre_discogs400-discogs-effnet-1.pb"
GENRE_METADATA = f"{MODEL_DIR}/genre_discogs400-discogs-effnet-1.json"
MOOD_MODEL = f"{MODEL_DIR}/mtg_jamendo_moodtheme-discogs-effnet-1.pb"
MOOD_METADATA = f"{MODEL_DIR}/mtg_jamendo_moodtheme-discogs-effnet-1.json"


def format_genre_tag(raw_genre, style='parent_child'):
    """
    Format genre tags for better readability
    
    Args:
        raw_genre: Raw genre string like "Rock---Alternative Rock"
        style: Formatting style
            - 'parent_child': "Rock - Alternative Rock" (default)
            - 'child_parent': "Alternative Rock - Rock"
            - 'child_only': "Alternative Rock"
            - 'raw': "Rock---Alternative Rock" (no formatting)
    
    Returns:
        Formatted genre string
    """
    if style == 'raw':
        return raw_genre
    
    # Split on triple dash
    if '---' in raw_genre:
        parts = raw_genre.split('---')
        parent = parts[0].strip()
        child = parts[1].strip() if len(parts) > 1 else ''
        
        if style == 'parent_child':
            # "Rock - Alternative Rock"
            if child:
                return f"{parent} - {child}"
            else:
                return parent
        
        elif style == 'child_parent':
            # "Alternative Rock - Rock"
            if child:
                return f"{child} - {parent}"
            else:
                return parent
        
        elif style == 'child_only':
            # "Alternative Rock" (just the specific genre)
            return child if child else parent
    
    # No triple dash, return as-is
    return raw_genre


def format_mood_tag(raw_mood):
    """
    Format mood tags for better readability
    
    Args:
        raw_mood: Raw mood string
    
    Returns:
        Formatted mood string (capitalized, etc.)
    """
    # Capitalize first letter of each word
    return raw_mood.title()


class Logger:
    """Dual output to console and log file"""
    def __init__(self, log_file):
        self.log_file = log_file
        self.file_handle = open(log_file, 'w', encoding='utf-8')
        self.write_header()
    
    def write_header(self):
        """Write log file header"""
        header = f"""
{'=' * 80}
ESSENTIA MUSIC TAGGER - LOG FILE
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'=' * 80}

"""
        self.file_handle.write(header)
        self.file_handle.flush()
    
    def log(self, message, console=True, file=True):
        """Write to console and/or file"""
        if console:
            print(message)
        if file:
            self.file_handle.write(message + '\n')
            self.file_handle.flush()
    
    def log_config(self, config, music_path):
        """Log configuration"""
        config_text = f"""
CONFIGURATION:
{'-' * 80}
Target Directory: {music_path}
Model Directory: {MODEL_DIR}

Genre Settings:
  - Number of genres: {config.top_n_genres}
  - Confidence threshold: {config.genre_threshold:.1%}
  - Genre format: {config.genre_format}

Mood Settings:
  - Enable moods: {config.enable_moods}
  - Confidence threshold: {config.mood_threshold:.1%}

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
        """Log detailed analysis results"""
        log_entry = f"""
FILE: {relative_path}
{'-' * 80}
"""
        
        # Genre results (raw)
        if results.get('genres'):
            log_entry += "GENRES (raw predictions):\n"
            for g in results['genres']:
                log_entry += f"  • {g['label']}: {g['confidence']:.2%}\n"
        else:
            log_entry += "GENRES: None passed threshold\n"
        
        # Formatted genres
        if results.get('formatted_genres'):
            log_entry += "\nGENRES (formatted for tags):\n"
            for fg in results['formatted_genres']:
                log_entry += f"  • {fg}\n"
        
        # All genre predictions (top 10)
        if results.get('all_genres_debug'):
            log_entry += "\nALL GENRE PREDICTIONS (top 10):\n"
            for label, conf in results['all_genres_debug']:
                log_entry += f"  • {label}: {conf:.2%}\n"
        
        # Mood results (raw)
        if results.get('moods'):
            log_entry += f"\nMOODS (raw predictions - {len(results['moods'])} total):\n"
            for m in results['moods']:
                log_entry += f"  • {m['label']}: {m['confidence']:.2%}\n"
        else:
            log_entry += "\nMOODS: None passed threshold\n"
        
        # Formatted moods
        if results.get('formatted_moods'):
            log_entry += "\nMOODS (formatted for tags):\n"
            for fm in results['formatted_moods']:
                log_entry += f"  • {fm}\n"
        
        # All mood predictions (top 10)
        if results.get('all_moods_debug'):
            log_entry += "\nALL MOOD PREDICTIONS (top 10):\n"
            for label, conf in results['all_moods_debug'][:10]:
                log_entry += f"  • {label}: {conf:.2%}\n"
        
        log_entry += f"\n{'=' * 80}\n"
        
        self.file_handle.write(log_entry)
        self.file_handle.flush()
    
    def log_summary(self, processed, errors, skipped):
        """Log final summary"""
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
        """Close log file"""
        self.file_handle.close()


class Config:
    """Runtime configuration from user prompts"""
    def __init__(self):
        self.dry_run = True
        self.top_n_genres = 3
        self.genre_threshold = 0.15
        self.mood_threshold = 0.01
        self.enable_moods = True
        self.write_confidence_tags = True
        self.overwrite_existing = False
        self.verbose = True
        self.log_file = None
        self.genre_format = 'parent_child'  # New: genre formatting style


class EssentiaAnalyzer:
    """Analyze audio files with Essentia models"""
    
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        
        logger.log("\n🔄 Loading models...")
        
        self.embedding_model = TensorflowPredictEffnetDiscogs(
            graphFilename=EMBEDDING_MODEL,
            output="PartitionedCall:1"
        )
        
        # Load genre model
        self.genre_model = TensorflowPredict2D(
            graphFilename=GENRE_MODEL,
            input="serving_default_model_Placeholder",
            output="PartitionedCall"
        )
        
        # Load genre class labels
        with open(GENRE_METADATA, 'r') as f:
            metadata = json.load(f)
            self.genre_labels = metadata['classes']
        
        logger.log(f"   ✅ Loaded {len(self.genre_labels)} genre classes")
        
        # Optionally load mood model
        self.mood_model = None
        self.mood_labels = None
        if config.enable_moods and os.path.exists(MOOD_MODEL):
            logger.log("   Loading mood model...")
            try:
                self.mood_model = TensorflowPredict2D(
                    graphFilename=MOOD_MODEL,
                    input="model/Placeholder",
                    output="model/Sigmoid"
                )
                with open(MOOD_METADATA, 'r') as f:
                    metadata = json.load(f)
                    self.mood_labels = metadata['classes']
                logger.log(f"   ✅ Loaded {len(self.mood_labels)} mood classes")
            except Exception as e:
                logger.log(f"   ⚠️  Could not load mood model: {e}")
                logger.log("      Continuing without mood analysis...")
                self.mood_model = None
        elif not config.enable_moods:
            logger.log("   ⏭️  Mood analysis disabled by user")
        else:
            logger.log("   ⚠️  Mood model not found")
        
        logger.log("   ✅ Models loaded successfully!\n")
    
    def analyze_file(self, filepath):
        """Analyze a single audio file"""
        try:
            # Load audio (resampled to 16kHz)
            audio = MonoLoader(
                filename=str(filepath),
                sampleRate=16000,
                resampleQuality=4
            )()
            
            # Get embeddings
            embeddings = self.embedding_model(audio)
            
            # Predict genres
            genre_predictions = self.genre_model(embeddings)
            genre_activations = np.mean(genre_predictions, axis=0)
            
            # Get top genres with threshold
            top_indices = np.argsort(genre_activations)[::-1][:self.config.top_n_genres * 2]
            genres = []
            for idx in top_indices:
                if len(genres) >= self.config.top_n_genres:
                    break
                if genre_activations[idx] >= self.config.genre_threshold:
                    genres.append({
                        'label': self.genre_labels[idx],
                        'confidence': float(genre_activations[idx])
                    })
            
            # If no genres pass threshold, take top 1 anyway
            if not genres:
                top_idx = np.argmax(genre_activations)
                genres.append({
                    'label': self.genre_labels[top_idx],
                    'confidence': float(genre_activations[top_idx])
                })
            
            results = {'genres': genres}
            
            # Format genres for tag writing
            results['formatted_genres'] = [
                format_genre_tag(g['label'], style=self.config.genre_format) 
                for g in genres
            ]
            
            # Store all genre activations for logging
            all_top_indices = np.argsort(genre_activations)[::-1][:10]
            results['all_genres_debug'] = [
                (self.genre_labels[idx], float(genre_activations[idx])) 
                for idx in all_top_indices
            ]
            
            # Predict moods (if model loaded)
            if self.mood_model:
                mood_predictions = self.mood_model(embeddings)
                mood_activations = np.mean(mood_predictions, axis=0)
                
                # Log raw mood activation stats
                max_mood = np.max(mood_activations)
                mean_mood = np.mean(mood_activations)
                
                self.logger.log(f"     [MOOD DEBUG] Max: {max_mood:.4f}, Mean: {mean_mood:.4f}, Threshold: {self.config.mood_threshold:.4f}", console=False)
                
                moods = []
                for idx, activation in enumerate(mood_activations):
                    if activation >= self.config.mood_threshold:
                        moods.append({
                            'label': self.mood_labels[idx],
                            'confidence': float(activation)
                        })
                
                # Sort by confidence
                moods = sorted(moods, key=lambda x: x['confidence'], reverse=True)
                results['moods'] = moods[:5]  # Limit to top 5 moods
                
                # Format moods for tag writing
                results['formatted_moods'] = [
                    format_mood_tag(m['label']) 
                    for m in results['moods']
                ]
                
                # Store all mood activations for logging
                results['all_moods_debug'] = sorted(
                    [(self.mood_labels[idx], float(mood_activations[idx])) 
                     for idx in range(len(mood_activations))],
                    key=lambda x: x[1], reverse=True
                )
                
                self.logger.log(f"     [MOOD DEBUG] Found {len(moods)} moods above threshold", console=False)
            
            return results
            
        except Exception as e:
            self.logger.log(f"     ⚠️  Error analyzing: {e}")
            return None


class TagWriter:
    """Write analysis results to audio file tags"""
    
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
    
    def write_tags(self, filepath, results):
        """Write genre/mood to file tags"""
        if self.config.dry_run:
            genre_info = []
            if results.get('formatted_genres'):
                genre_info.append(f"Genres: {', '.join(results['formatted_genres'])}")
            if results.get('formatted_moods'):
                genre_info.append(f"Moods: {', '.join(results['formatted_moods'][:3])}")
            self.logger.log(f"     [DRY RUN] Would write: {' | '.join(genre_info)}")
            return
        
        try:
            file_ext = filepath.suffix.lower()
            
            if file_ext == '.flac':
                self._write_flac(filepath, results)
            elif file_ext == '.mp3':
                self._write_mp3(filepath, results)
            else:
                self.logger.log(f"     ⚠️  Unsupported format: {file_ext}")
                
        except Exception as e:
            self.logger.log(f"     ⚠️  Error writing tags: {e}")
    
    def _write_flac(self, filepath, results):
        """Write to FLAC tags"""
        audio = FLAC(filepath)
        
        # Check if we should skip existing tags
        if not self.config.overwrite_existing and 'GENRE' in audio:
            self.logger.log(f"     ⏭️  Skipping (already has GENRE tag)")
            return
        
        tags_written = []
        
        # Write genres (using formatted versions)
        if results.get('formatted_genres'):
            genre_str = '; '.join(results['formatted_genres'])
            audio['GENRE'] = genre_str
            tags_written.append(f"GENRE={genre_str}")
            
            # Store confidence scores if enabled (use raw labels for clarity)
            if self.config.write_confidence_tags and results.get('genres'):
                genre_details = [f"{g['label']}: {g['confidence']:.2%}" for g in results['genres']]
                confidence_str = ', '.join(genre_details)
                audio['ESSENTIA_GENRE'] = f"Essentia: {confidence_str}"
                tags_written.append(f"ESSENTIA_GENRE={confidence_str}")
        
        # Write moods (using formatted versions)
        if results.get('formatted_moods'):
            mood_str = '; '.join(results['formatted_moods'][:3])
            audio['MOOD'] = mood_str
            tags_written.append(f"MOOD={mood_str}")
            
            if self.config.write_confidence_tags and results.get('moods'):
                mood_details = [f"{m['label']}: {m['confidence']:.2%}" for m in results['moods'][:3]]
                mood_conf_str = ', '.join(mood_details)
                audio['ESSENTIA_MOOD'] = f"Essentia: {mood_conf_str}"
                tags_written.append(f"ESSENTIA_MOOD={mood_conf_str}")
        
        audio.save()
        
        # Log what was written
        self.logger.log(f"     ✅ Written tags: {', '.join(tags_written)}", console=False)
    
    def _write_mp3(self, filepath, results):
        """Write to MP3 ID3 tags"""
        try:
            audio = ID3(filepath)
        except:
            audio = ID3()
        
        # Check if we should skip existing tags
        if not self.config.overwrite_existing:
            try:
                existing = audio.getall('TCON')
                if existing:
                    self.logger.log(f"     ⏭️  Skipping (already has GENRE tag)")
                    return
            except:
                pass
        
        tags_written = []
        
        # Write genres (using formatted versions)
        if results.get('formatted_genres'):
            genre_str = '; '.join(results['formatted_genres'])
            audio.delall('TCON')
            audio.add(TCON(encoding=3, text=genre_str))
            tags_written.append(f"TCON={genre_str}")
            
            # Store confidence in comment if enabled (use raw labels)
            if self.config.write_confidence_tags and results.get('genres'):
                genre_details = [f"{g['label']}: {g['confidence']:.2%}" for g in results['genres']]
                confidence_str = ', '.join(genre_details)
                audio.delall('COMM::eng')
                audio.add(COMM(
                    encoding=3,
                    lang='eng',
                    desc='Essentia Genre',
                    text=confidence_str
                ))
                tags_written.append(f"COMM(genre)={confidence_str}")
            
            # Write moods in a separate comment if available (using formatted versions)
            if results.get('formatted_moods') and self.config.write_confidence_tags:
                mood_str = '; '.join(results['formatted_moods'][:3])
                audio.add(COMM(
                    encoding=3,
                    lang='eng',
                    desc='Essentia Mood',
                    text=mood_str
                ))
                tags_written.append(f"COMM(mood)={mood_str}")
        
        audio.save(filepath)
        
        # Log what was written
        self.logger.log(f"     ✅ Written tags: {', '.join(tags_written)}", console=False)


def scan_library(root_path, analyzer, tag_writer, config, logger):
    """Recursively scan and process music library"""
    root = Path(root_path)
    audio_extensions = {'.flac', '.mp3', '.ogg', '.m4a', '.wav'}
    
    logger.log("\n🔍 Scanning for audio files...")
    files = list(root.rglob('*'))
    audio_files = [f for f in files if f.suffix.lower() in audio_extensions]
    
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
        
        # Analyze
        results = analyzer.analyze_file(filepath)
        
        if results:
            # Print results to console (formatted versions)
            if results.get('genres'):
                genre_list = [f"{g['label']} ({g['confidence']:.1%})" for g in results['genres']]
                logger.log(f"     🎸 Raw: {', '.join(genre_list)}")
            
            if results.get('formatted_genres'):
                logger.log(f"     🎸 Formatted: {', '.join(results['formatted_genres'])}")
            
            if results.get('moods'):
                mood_list = [f"{m['label']} ({m['confidence']:.1%})" for m in results['moods'][:3]]
                logger.log(f"     😊 Raw: {', '.join(mood_list)}")
            
            if results.get('formatted_moods'):
                logger.log(f"     😊 Formatted: {', '.join(results['formatted_moods'][:3])}")
            elif results.get('moods') is not None and len(results.get('moods', [])) == 0:
                logger.log(f"     😊 Moods: None above threshold ({config.mood_threshold:.2%})")
            
            # Show debug info if verbose
            if config.verbose and results.get('all_genres_debug'):
                top_5 = ', '.join([f"{label} ({conf:.1%})" for label, conf in results['all_genres_debug'][:5]])
                logger.log(f"     📊 Top 5 genres: {top_5}", console=False)
            
            # Log detailed analysis to file
            logger.log_analysis(filepath, results, relative_path)
            
            # Write tags
            tag_writer.write_tags(filepath, results)
            
            if not config.dry_run:
                logger.log("     ✅ Tags written")
            
            processed += 1
        else:
            errors += 1
        
        logger.log("")
    
    # Summary
    logger.log(f"\n{'=' * 70}")
    logger.log(f"📊 SUMMARY")
    logger.log(f"{'=' * 70}")
    logger.log(f"✅ Processed: {processed}")
    logger.log(f"❌ Errors: {errors}")
    logger.log(f"⏭️  Skipped: {skipped}")
    
    logger.log_summary(processed, errors, skipped)


def get_music_path():
    """Prompt user for music directory path"""
    print("\n" + "=" * 70)
    print("🎸 ESSENTIA MUSIC TAGGER - INTERACTIVE MODE")
    print("=" * 70)
    print("\nThis tool will recursively analyze ALL audio files")
    print("in the directory you specify and its subdirectories.\n")
    
    # Show some example paths to help user
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
        audio_extensions = {'.flac', '.mp3', '.ogg', '.m4a', '.wav'}
        sample_files = list(path.rglob('*'))
        audio_count = len([f for f in sample_files if f.suffix.lower() in audio_extensions])
        
        print(f"\n📂 Directory: {path}")
        print(f"🎵 Found ~{audio_count} audio files")
        
        confirm = input("\nProceed with this directory? [Y/n]: ").strip().lower()
        if confirm in ['', 'y', 'yes']:
            return str(path)
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


def configure_settings():
    """Interactive configuration"""
    config = Config()
    
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
    
    # Number of genres
    print("\n" + "─" * 70)
    print("🎸 GENRE SETTINGS")
    print("   How many genre tags to write per song")
    print("   Recommended: 2-4 genres")
    print("   • 1 = Only top genre")
    print("   • 3 = Balanced (good variety)")
    print("   • 5 = Comprehensive (may include less relevant)")
    config.top_n_genres = get_int_input("Number of genres to write", default=3, min_val=1, max_val=10)
    
    # Genre threshold
    print("\n   Genre confidence threshold (as percentage)")
    print("   Only include genres above this confidence level")
    print("   • 5% = Very inclusive (more genres)")
    print("   • 15% = Balanced (recommended)")
    print("   • 25% = Strict (fewer, higher confidence)")
    print("   • 35% = Very strict (may get 0-1 genres)")
    threshold_pct = get_float_input("Genre threshold (%)", default=15, min_val=1, max_val=50)
    config.genre_threshold = threshold_pct / 100.0
    
    # Genre formatting
    print("\n   Genre tag formatting")
    print("   How to format genre tags like 'Rock---Alternative Rock'")
    print("   • 1 = 'Rock - Alternative Rock' (parent - child)")
    print("   • 2 = 'Alternative Rock - Rock' (child - parent)")
    print("   • 3 = 'Alternative Rock' (child only)")
    print("   • 4 = 'Rock---Alternative Rock' (raw/no formatting)")
    format_choice = get_int_input("Genre format", default=1, min_val=1, max_val=4)
    format_map = {
        1: 'parent_child',
        2: 'child_parent',
        3: 'child_only',
        4: 'raw'
    }
    config.genre_format = format_map[format_choice]
    
    # Enable moods
    print("\n" + "─" * 70)
    print("😊 MOOD ANALYSIS")
    print("   Analyze and tag moods/themes (e.g., energetic, dark, happy)")
    print("   Note: Mood predictions are typically MUCH lower confidence than genres")
    print("   Often in the 0.01% - 5% range!")
    config.enable_moods = get_yes_no("Enable mood analysis?", default=True)
    
    if config.enable_moods:
        print("\n   Mood confidence threshold (as percentage)")
        print("   Moods naturally have VERY low confidence")
        print("   • 0.1% = Very inclusive (will get many moods)")
        print("   • 0.5% = Inclusive (recommended to start)")
        print("   • 1% = Balanced")
        print("   • 3% = Strict (may get few/no moods)")
        mood_threshold_pct = get_float_input("Mood threshold (%)", default=0.5, min_val=0.01, max_val=20)
        config.mood_threshold = mood_threshold_pct / 100.0
    
    # Write confidence scores
    print("\n" + "─" * 70)
    print("📊 CONFIDENCE SCORES")
    print("   Write confidence percentages to additional tags")
    print("   Example: 'ESSENTIA_GENRE: Alternative Rock: 32%, Indie Rock: 23%'")
    config.write_confidence_tags = get_yes_no("Write confidence score tags?", default=True)
    
    # Overwrite existing
    print("\n" + "─" * 70)
    print("♻️  EXISTING TAGS")
    print("   What to do if files already have genre tags")
    print("   • Overwrite: Replace existing tags")
    print("   • Skip: Leave files with existing tags untouched")
    config.overwrite_existing = get_yes_no("Overwrite existing genre tags?", default=False)
    
    # Verbose output
    print("\n" + "─" * 70)
    print("📢 VERBOSE OUTPUT")
    print("   Show detailed analysis info (top 10 predictions, etc.)")
    config.verbose = get_yes_no("Enable verbose output?", default=True)
    
    return config


def display_config_summary(config, music_path):
    """Display final configuration before processing"""
    print("\n" + "=" * 70)
    print("📋 FINAL SETTINGS")
    print("=" * 70)
    print(f"📂 Target directory: {music_path}")
    print(f"📁 Model directory: {MODEL_DIR}")
    print(f"\n🎸 Genre Settings:")
    print(f"   • Number of genres: {config.top_n_genres}")
    print(f"   • Confidence threshold: {config.genre_threshold:.2%}")
    print(f"   • Format style: {config.genre_format}")
    print(f"\n😊 Mood Settings:")
    print(f"   • Enable moods: {config.enable_moods}")
    if config.enable_moods:
        print(f"   • Confidence threshold: {config.mood_threshold:.2%}")
    print(f"\n📊 Other Settings:")
    print(f"   • Dry run mode: {config.dry_run}")
    print(f"   • Write confidence tags: {config.write_confidence_tags}")
    print(f"   • Overwrite existing: {config.overwrite_existing}")
    print(f"   • Verbose output: {config.verbose}")
    
    if config.dry_run:
        print(f"\n⚠️  DRY RUN MODE - No files will be modified!")
    else:
        print(f"\n⚠️  LIVE MODE - Files WILL be modified!")
    
    # Set log file path
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    config.log_file = f"essentia_tagger_{timestamp}.log"
    print(f"\n📝 Log file: {config.log_file}")
    
    print("=" * 70 + "\n")
    
    if not config.dry_run:
        confirm = input("Ready to proceed? [Y/n]: ").strip().lower()
        if confirm not in ['', 'y', 'yes']:
            print("Cancelled.")
            sys.exit(0)


def main():
    """Main entry point"""
    logger = None
    try:
        # Get path from user
        music_path = get_music_path()
        
        # Configure settings interactively
        config = configure_settings()
        
        # Show summary and confirm
        display_config_summary(config, music_path)
        
        # Initialize logger
        logger = Logger(config.log_file)
        logger.log_config(config, music_path)
        
        # Initialize
        analyzer = EssentiaAnalyzer(config, logger)
        tag_writer = TagWriter(config, logger)
        
        # Process library
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
