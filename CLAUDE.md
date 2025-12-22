# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research project that uses OpenAI's Batch API to analyze sentiment in Federal Reserve and ECB speeches. The project extracts 5 dimensions of sentiment (hawkish/dovish, uncertainty, forward guidance, topic emphasis, market impact predictions) from central bank communications.

**Key Innovation**: Uses OpenAI Batch API with chunking to process hundreds of speeches cost-effectively (50% discount vs real-time API) while staying under the 90,000 enqueued token limit.

## Critical Architecture Concepts

### 1. Three-Phase Pipeline Architecture

The project follows a strict sequential pipeline:

**Phase 1: Data Preparation**
- `phase1_data_prep.py` → Downloads Hugging Face dataset, samples speeches
- `phase1_batch_prep.py` → Creates JSONL batch files with sentiment prompts
- Output: `data/batch_input/batch_sample_2022_2023.jsonl`

**Phase 2: Batch Processing with Chunking**
- `phase2_batch_submit.py` → **Automatically splits into 17 chunks** (~18-20 speeches each)
- Chunking is MANDATORY due to OpenAI's 90,000 enqueued token limit
- `phase2_check_status.py` → Monitors 17 separate batch jobs
- `phase2_download_results.py` → Downloads and combines all chunks
- Output: `data/results/sentiment_results_2022_2023.csv` (combined from all chunks)

**Phase 3: Index Building & Visualization**
- `phase3_build_indices.py` → Creates forward-filled daily time series
- `phase3_build_indices_no_fill.py` → Creates sparse (speech-dates only) time series
- Three visualization scripts create 18 total charts (3 types × 6 metrics × Fed/ECB)
- `validate_hawkish_dovish.py` → Validates LLM scores against actual speeches

### 2. Chunking Strategy (CRITICAL)

**Why chunking exists**: OpenAI enforces a 90,000 enqueued token limit per organization. Without chunking, batch submission fails with:
```
Error: Enqueued token limit reached for gpt-4o-2024-08-06
Limit: 90,000 enqueued tokens per organization
```

**How it works**:
- 311 speeches = ~5 million tokens total
- Split into 17 chunks of ~18-20 speeches each
- Each chunk ≈ 5,000 tokens (well under 90,000 limit)
- Chunks process in parallel on OpenAI servers
- Results automatically combined into single CSV

**Files involved**:
- Input: `data/batch_input/batch_sample_2022_2023.jsonl` (single file)
- Chunk outputs: `data/batch_output/chunk01_results.jsonl` through `chunk17_results.jsonl`
- Chunk parsed: `data/results/chunk01_parsed.csv` through `chunk17_parsed.csv`
- Combined: `data/results/sentiment_results_2022_2023.csv`

### 3. Two Data Versions: Forward-Filled vs Non-Filled

The project creates TWO versions of daily indices:

**Forward-filled** (`phase3_build_indices.py`):
- Continuous daily time series with no gaps
- Uses pandas forward-fill: last value carried forward
- Purpose: Area plots showing continuous trends
- Files: `fed_daily_indices.csv`, `ecb_daily_indices.csv`

**Non-filled** (`phase3_build_indices_no_fill.py`):
- Sparse: only dates with actual speeches
- No gap filling, NaN on non-speech dates
- Purpose: Bar charts and calendar heatmaps (accurate speech-date representation)
- Files: `fed_daily_indices_no_fill.csv`, `ecb_daily_indices_no_fill.csv`

**Why both versions**:
- Bar charts need accurate "this speech on this date" representation
- Calendar heatmaps show sparse data naturally (grey = no speech)
- Area plots need continuous lines (gaps would break visualization)

### 4. Visualization Architecture

Three distinct visualization types, each using appropriate data:

**Bar Charts** (`phase3_visualize_indices_bars.py`):
- Uses: Non-filled data (sparse)
- Shows: Individual speeches as bars on specific dates
- Special handling:
  - Diverging bars for hawkish/dovish (centered on 0)
  - Diverging bars for market impact (centered on 50)
  - Market impact charts have no spines/grid

**Area Plots** (`phase3_visualize_indices_area.py`):
- Uses: Forward-filled data (continuous)
- Shows: Trends over time with filled areas
- Special handling:
  - Hawkish/dovish: Diverging colors (red above 0, blue below 0)
  - Neutral grey line for hawkish/dovish
  - All others: Colored fill + matching line

**Calendar Heatmaps** (`phase3_visualize_indices_dayplot.py`):
- Uses: Non-filled data (sparse)
- Library: `dayplot` (not standard matplotlib)
- Special handling:
  - Grey cells = no speech on that date
  - Zero values replaced with 0.01 AND vmin=-1 (prevents grey coloring)
  - Diverging colormaps for hawkish/dovish and market impact
  - 5 legend bins, no numeric labels

### 5. Configuration Architecture

`src/config.py` is the central configuration hub:
- All paths (uses `pathlib.Path`)
- API credentials (loaded from `.env`)
- Model settings (GPT-4o)
- **Prompt template** (via `get_sentiment_prompt()`)

**Important**: The sentiment extraction prompt is in `Config.get_sentiment_prompt()`. When modifying what the LLM extracts, edit this method.

## Common Commands

### Setup
```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Unix

# Install dependencies
pip install -r requirements.txt

# Configure API keys
cp .env.example .env
# Edit .env and add OPENAI_API_KEY and HF_TOKEN
```

### Running the Pipeline

**Phase 1: Prepare data and batch files**
```bash
python phase1_data_prep.py      # Downloads dataset, creates sample
python phase1_batch_prep.py     # Creates JSONL batch file
```

**Phase 2: Submit and process batches**
```bash
# Submit all chunks (automatic chunking)
python phase2_batch_submit.py

# Check status of all chunks
python phase2_check_status.py

# Download and combine results
python phase2_download_results.py
```

**Phase 3: Build indices and visualizations**
```bash
# Build daily indices
python phase3_build_indices.py          # Forward-filled version
python phase3_build_indices_no_fill.py  # Sparse version

# Create visualizations
python phase3_visualize_indices_bars.py     # Bar charts
python phase3_visualize_indices_area.py     # Area plots
python phase3_visualize_indices_dayplot.py  # Calendar heatmaps

# Validate results
python validate_hawkish_dovish.py
```

### Utility Scripts
```bash
# List all batch jobs
python list_batches.py

# Check for batch errors
python check_batch_errors.py

# Verify configuration
python src/config.py
```

## Key Data Files

**Input Data**:
- `data/batch_input/batch_sample_2022_2023.jsonl` - Single batch file (311 speeches)

**Intermediate Outputs** (from chunking):
- `data/batch_output/chunk01_results.jsonl` ... `chunk17_results.jsonl` - Raw LLM outputs
- `data/results/chunk01_parsed.csv` ... `chunk17_parsed.csv` - Parsed scores

**Combined Results**:
- `data/results/sentiment_results_2022_2023.csv` - All speeches combined
- `data/results/phase3_prepared_data.csv` - With metadata (author, title, speech_id)

**Daily Indices**:
- `data/results/fed_daily_indices.csv` - Fed forward-filled
- `data/results/fed_daily_indices_no_fill.csv` - Fed sparse
- `data/results/ecb_daily_indices.csv` - ECB forward-filled
- `data/results/ecb_daily_indices_no_fill.csv` - ECB sparse

**Visualizations**:
- `reports/*_bars.png` - 6 bar chart files (3 metrics × 2 institutions)
- `reports/*_area.png` - 6 area plot files
- `reports/*_calendar.png` - 6 calendar heatmap files
- `reports/hawkish_dovish_validation.txt` - Validation report

## Important Implementation Details

### Sentiment Extraction
The LLM extracts these dimensions from each speech:
1. **Hawkish/Dovish Score** (-100 to +100): Policy tightening vs easing stance
2. **Topic Emphasis** (0-100 each): inflation, growth, financial_stability, labor_market, international
3. **Uncertainty Level** (0-100): Confidence about economic outlook
4. **Forward Guidance Strength** (0-100): Explicitness of future policy signals
5. **Market Impact** (rise/fall/neutral): Predictions for stocks, bonds, currency

### Market Impact Diffusion Index
Market impact predictions are converted to diffusion indices (0-100 scale):
- Formula: `(% rise) + (0.5 * % neutral)` × 100
- 100 = all predictions "rise"
- 50 = neutral/mixed
- 0 = all predictions "fall"

### Dayplot Zero-Value Handling
The `dayplot` library treats 0 as "no contribution" and colors it grey. To fix:
1. Replace 0 → 0.01 in data
2. Set `vmin=-1` (so 0.01 is clearly above minimum)
3. This makes 0 values map to yellow (lowest color) instead of grey

### Speech ID Mapping
- Batch input uses `custom_id` (e.g., "speech_0", "speech_1")
- Phase 3 prepared data includes `speech_id` column
- Validation script uses `speech_id` to match scores with original speeches

## Module Responsibilities

**src/config.py**: Central configuration, prompt template, directory setup
**src/data_loader.py**: Hugging Face dataset loading and caching
**src/batch_builder.py**: Creates JSONL batch files from speeches
**src/batch_processor.py**: Uploads, submits, monitors, downloads batch jobs
**src/output_validator.py**: Validates LLM outputs (ranges, required fields)
**src/utils.py**: Token estimation, formatting, JSON utilities

## Project Status

- ✅ Phase 1: Complete (data prep, batch file creation)
- ✅ Phase 2: Complete (batch processing with chunking)
- ✅ Phase 3: Complete (indices, visualizations, validation)
- ⏳ Phase 4: Next (market correlation analysis)

## Common Gotchas

1. **Chunking is mandatory**: Don't try to submit the full batch file without chunking - it will fail with enqueued token limit error

2. **Two data versions matter**: Use forward-filled for area plots, non-filled for bar charts and calendars

3. **Dayplot zero handling**: When working with calendar heatmaps, remember the 0→0.01 replacement AND vmin=-1 requirement

4. **Custom_id vs speech_id**: Batch API uses `custom_id`, internal processing uses `speech_id` - they map 1:1 but have different names

5. **Market impact is categorical**: Don't treat "rise/fall/neutral" as numeric - convert to diffusion index first

6. **API keys required**: Both OPENAI_API_KEY and HF_TOKEN must be in .env file

7. **Dataset caching**: First run downloads ~2-3GB Hugging Face dataset, subsequent runs use cache
