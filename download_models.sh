#!/bin/bash
# Download Essentia pre-trained models (all discogs-effnet classifier heads)
# Models are ~250MB total

set -e

MODEL_DIR="${HOME}/essentia_models"
BASE="https://essentia.upf.edu/models"

echo "🎵 Essentia Model Downloader"
echo "================================"
echo ""
echo "Downloading models to: ${MODEL_DIR}"
echo ""

# Create directory
mkdir -p "${MODEL_DIR}"
cd "${MODEL_DIR}"

download() {
    local url="$1"
    local file
    file=$(basename "$url")
    if [ -f "$file" ]; then
        echo "   ⏭️  Already exists: $file"
    else
        wget -q --show-progress "$url"
    fi
}

# ── Embedding model (required) ──────────────────────────────────────────────
echo "📦 Embedding model..."
download "${BASE}/music-style-classification/discogs-effnet/discogs-effnet-bs64-1.pb"

# ── Genre ────────────────────────────────────────────────────────────────────
echo ""
echo "📦 Genre models..."
download "${BASE}/classification-heads/genre_discogs400/genre_discogs400-discogs-effnet-1.pb"
download "${BASE}/classification-heads/genre_discogs400/genre_discogs400-discogs-effnet-1.json"
download "${BASE}/classification-heads/mtg_jamendo_genre/mtg_jamendo_genre-discogs-effnet-1.pb"
download "${BASE}/classification-heads/mtg_jamendo_genre/mtg_jamendo_genre-discogs-effnet-1.json"

# ── Mood / Theme ─────────────────────────────────────────────────────────────
echo ""
echo "📦 Mood/Theme models..."
download "${BASE}/classification-heads/mtg_jamendo_moodtheme/mtg_jamendo_moodtheme-discogs-effnet-1.pb"
download "${BASE}/classification-heads/mtg_jamendo_moodtheme/mtg_jamendo_moodtheme-discogs-effnet-1.json"
download "${BASE}/classification-heads/mood_aggressive/mood_aggressive-discogs-effnet-1.pb"
download "${BASE}/classification-heads/mood_aggressive/mood_aggressive-discogs-effnet-1.json"
download "${BASE}/classification-heads/mood_happy/mood_happy-discogs-effnet-1.pb"
download "${BASE}/classification-heads/mood_happy/mood_happy-discogs-effnet-1.json"
download "${BASE}/classification-heads/mood_party/mood_party-discogs-effnet-1.pb"
download "${BASE}/classification-heads/mood_party/mood_party-discogs-effnet-1.json"
download "${BASE}/classification-heads/mood_relaxed/mood_relaxed-discogs-effnet-1.pb"
download "${BASE}/classification-heads/mood_relaxed/mood_relaxed-discogs-effnet-1.json"
download "${BASE}/classification-heads/mood_sad/mood_sad-discogs-effnet-1.pb"
download "${BASE}/classification-heads/mood_sad/mood_sad-discogs-effnet-1.json"
download "${BASE}/classification-heads/mood_acoustic/mood_acoustic-discogs-effnet-1.pb"
download "${BASE}/classification-heads/mood_acoustic/mood_acoustic-discogs-effnet-1.json"
download "${BASE}/classification-heads/mood_electronic/mood_electronic-discogs-effnet-1.pb"
download "${BASE}/classification-heads/mood_electronic/mood_electronic-discogs-effnet-1.json"

# ── Context ──────────────────────────────────────────────────────────────────
echo ""
echo "📦 Context models..."
download "${BASE}/classification-heads/danceability/danceability-discogs-effnet-1.pb"
download "${BASE}/classification-heads/danceability/danceability-discogs-effnet-1.json"
download "${BASE}/classification-heads/voice_instrumental/voice_instrumental-discogs-effnet-1.pb"
download "${BASE}/classification-heads/voice_instrumental/voice_instrumental-discogs-effnet-1.json"
download "${BASE}/classification-heads/gender/gender-discogs-effnet-1.pb"
download "${BASE}/classification-heads/gender/gender-discogs-effnet-1.json"
download "${BASE}/classification-heads/tonal_atonal/tonal_atonal-discogs-effnet-1.pb"
download "${BASE}/classification-heads/tonal_atonal/tonal_atonal-discogs-effnet-1.json"
download "${BASE}/classification-heads/timbre/timbre-discogs-effnet-1.pb"
download "${BASE}/classification-heads/timbre/timbre-discogs-effnet-1.json"
download "${BASE}/classification-heads/approachability/approachability_2c-discogs-effnet-1.pb"
download "${BASE}/classification-heads/approachability/approachability_2c-discogs-effnet-1.json"
download "${BASE}/classification-heads/engagement/engagement_2c-discogs-effnet-1.pb"
download "${BASE}/classification-heads/engagement/engagement_2c-discogs-effnet-1.json"

# ── Instrument ───────────────────────────────────────────────────────────────
echo ""
echo "📦 Instrument model..."
download "${BASE}/classification-heads/mtg_jamendo_instrument/mtg_jamendo_instrument-discogs-effnet-1.pb"
download "${BASE}/classification-heads/mtg_jamendo_instrument/mtg_jamendo_instrument-discogs-effnet-1.json"

# ── Auto-tagging ─────────────────────────────────────────────────────────────
echo ""
echo "📦 Auto-tagging models..."
download "${BASE}/classification-heads/mtg_jamendo_top50tags/mtg_jamendo_top50tags-discogs-effnet-1.pb"
download "${BASE}/classification-heads/mtg_jamendo_top50tags/mtg_jamendo_top50tags-discogs-effnet-1.json"
download "${BASE}/classification-heads/mtt/mtt-discogs-effnet-1.pb"
download "${BASE}/classification-heads/mtt/mtt-discogs-effnet-1.json"

echo ""
echo "✅ Download complete!"
echo ""
echo "Models installed:"
ls -lh "${MODEL_DIR}"

echo ""
echo "🎸 Ready to run: python tag_music.py"
