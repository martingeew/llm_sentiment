# LLM Sentiment Analysis of Central Bank Communications

A research project using Large Language Models (LLMs) to analyze sentiment in Federal Reserve and European Central Bank speeches, leveraging OpenAI's Batch API for cost-effective processing at scale.

## 📋 Table of Contents

- [Overview](#overview)
- [Project Phases](#project-phases)
- [Current Phase: Phase 1](#phase-1-setup--batch-processing-demo)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [For Beginners](#for-beginners)
- [Cost Estimates](#cost-estimates)
- [Documentation](#documentation)

---

## Overview

### What This Project Does

This project analyzes speeches from central bank officials (Federal Reserve and ECB) to extract:

1. **Hawkish/Dovish Score**: Is the central bank leaning toward tightening (raising rates) or easing (lowering rates)?
2. **Topic Emphasis**: What are they focused on? (inflation, growth, financial stability, labor markets, international issues)
3. **Uncertainty Level**: How confident are they about the economic outlook?
4. **Forward Guidance**: How clearly are they signaling future policy actions?
5. **Market Impact**: What might financial markets do in response?

### Why This Matters

- **For Economics Research**: Better measurement tools for studying central bank communication
- **For Financial Markets**: Automated analysis of policy signals that can move markets
- **For Policy Analysis**: Cross-country comparisons of central bank messaging
- **For Education**: Practical demonstration of LLMs in economic research

### Key Innovation

We use OpenAI's **Batch API** instead of real-time API calls:
- ✅ **50% cost savings** (same quality, half the price)
- ✅ **Processes hundreds of speeches** overnight
- ✅ **No rate limits** - process large datasets at once
- ✅ **Reproducible** - same analysis applied consistently

---

## Project Phases

This project is organized into 5 phases, each building on the previous one:

### ✅ **Phase 1** (Complete): Setup & Batch Processing Demo
- Load ECB-FED speeches dataset
- Sample 2 years of data (2022-2023)
- Create batch processing files
- Compare costs: Batch API vs Real-time API

### ✅ **Phase 2** (Complete): Batch Submission & Output Validation
- Submit batch job to OpenAI API
- Monitor processing progress
- Download and parse results
- Validate LLM output quality
- Verify scores are sensible

### ✅ **Phase 3** (Complete): Build Sentiment Indices & Visualizations
- Create daily time series indices (hawkish/dovish, uncertainty, etc.)
- Generate forward-filled and non-filled versions
- Create multiple visualization types:
  - Bar charts for policy metrics, topics, and market impact
  - Area plots for continuous time series
  - Calendar heatmaps for sparse data visualization
- Validate hawkish/dovish scores against actual speech content

### 📈 **Phase 4**: Market Correlation Analysis
- Correlate sentiment indices with market variables:
  - Stock market returns (S&P 500, EURO STOXX)
  - Bond yields (10-year Treasury)
  - Exchange rates (USD/EUR)
  - Market volatility (VIX)
- Test combined indices (e.g., uncertainty + hawkishness)
- Compare Fed vs ECB predictive power

### 🏆 **Phase 5**: Benchmark Against Literature
- Compare with published indices (FinBERT-FOMC, dictionary methods)
- Validate against known events (2008 crisis, COVID, 2022 inflation)
- Document findings

Each phase will be completed with a Pull Request (PR) for review before moving to the next.

---

## Phase 1: Setup & Batch Processing Demo (✅ Complete)

### Goals

1. ✅ Demonstrate the project works end-to-end
2. ✅ Understand costs before processing full dataset
3. ✅ Validate data quality and batch processing setup
4. ✅ Show 50% cost savings from Batch API

### What You Get

After running Phase 1:

- **Sample dataset**: 311 speeches from 2022-2023 (14 MB)
- **Batch file**: 311 requests ready for OpenAI (4.9 MB)
- **Cost estimates**: $2.32 (batch) vs $4.64 (real-time)
- **Validation**: Confirmed everything works correctly

### Files Created

```
data/
├── processed/
│   └── sample_2022_2023.csv          # 311 speeches
├── batch_input/
│   └── batch_sample_2022_2023.jsonl  # Ready for OpenAI
└── results/
    └── phase1_statistics.json        # Cost breakdown
```

---

## Phase 2: Batch Submission & Output Validation (✅ Complete)

### Goals

1. ✅ Submit batch job to OpenAI (with chunking strategy)
2. ✅ Monitor processing progress
3. ✅ Download and parse results from all chunks
4. ✅ Validate LLM output quality
5. ✅ Verify sentiment scores are sensible

### Why Chunking is Needed

The batch input is split into **multiple chunks** rather than submitting as a single large file. This is **essential** due to OpenAI's API limits.

**Primary Reason - Enqueued Token Limit:**

OpenAI enforces a limit on how many tokens can be "enqueued" (waiting to be processed) at once:

```
Error: Enqueued token limit reached for gpt-4o-2024-08-06
Limit: 90,000 enqueued tokens per organization
```

**What this means:**
- You cannot submit a batch with more than 90,000 input tokens at once
- For this project, 311 speeches ≈ 5 million tokens total
- Without chunking: Would hit the limit immediately
- With chunking: Each chunk stays well under 90,000 tokens

**Additional Benefits of Chunking:**
1. **Error Isolation**: If one chunk fails, others complete successfully
2. **Parallel Processing**: Multiple chunks can process simultaneously
3. **Resume Capability**: Completed chunks don't need reprocessing
4. **Memory Management**: Easier to load and process smaller result files
5. **Progress Tracking**: Can monitor completion of individual chunks

**For this project:**
- 311 speeches split into **17 chunks** (~18-20 speeches per chunk)
- Each chunk ≈ 5,000 enqueued tokens (well under the 90,000 limit)
- Chunks submitted sequentially to stay under token limit
- Results downloaded and combined automatically

### What You'll Get

After running Phase 2:

- **Sentiment scores**: All 5 dimensions for each speech
- **Validation report**: Quality metrics and error analysis
- **Distribution analysis**: Check scores look reasonable
- **Parsed CSV**: Ready for Phase 3 analysis

### Files Created

```
data/
├── batch_output/
│   ├── chunk01_results.jsonl          # Raw LLM outputs (chunk 1)
│   ├── chunk02_results.jsonl          # Raw LLM outputs (chunk 2)
│   ├── ...                            # (chunks 3-16)
│   └── chunk17_results.jsonl          # Raw LLM outputs (chunk 17)
└── results/
    ├── chunk01_parsed.csv             # Parsed scores (chunk 1)
    ├── chunk02_parsed.csv             # Parsed scores (chunk 2)
    ├── ...                            # (chunks 3-16)
    ├── chunk17_parsed.csv             # Parsed scores (chunk 17)
    ├── sentiment_results_2022_2023.csv # Combined results from all chunks
    ├── phase2_batch_info.json          # Batch tracking info
    └── phase2_validation_report.json   # Quality report
```

**Note**: Individual chunk files are intermediate outputs. The combined `sentiment_results_2022_2023.csv` is what gets used in Phase 3.

### How to Run

**Option 1: Submit and monitor**
```bash
python phase2_batch_submit.py
```
*This script automatically handles chunking - it will submit all 17 chunks and monitor their progress.*

**Option 2: Submit and check later**
```bash
# Submit
python phase2_batch_submit.py
# (say 'n' when asked to monitor)

# Check later
python phase2_check_status.py
```

**Option 3: Download results**
```bash
# After all chunks complete, download and combine results
python phase2_download_results.py
```

**Expected Cost:** ~$2.32 for 2022-2023 sample (all chunks combined)
**Expected Time:** 30 minutes to 4 hours (chunks may complete at different times)

**Note**: The scripts handle all chunking automatically. You don't need to manually split files or track individual chunks - just run the scripts and they manage the chunking for you.

### Chunking Workflow

```
Phase 1: Create batch file
└─> batch_sample_2022_2023.jsonl (311 speeches, 4.9 MB)

Phase 2a: Submit (automatic chunking)
├─> Chunk 1: speeches 1-18   → Batch Job 1
├─> Chunk 2: speeches 19-37  → Batch Job 2
├─> ...
└─> Chunk 17: speeches 295-311 → Batch Job 17

Phase 2b: Process (parallel)
├─> All 17 chunks process simultaneously on OpenAI servers
└─> Each completes independently (30 min - 4 hours)

Phase 2c: Download (automatic combination)
├─> Download chunk01_results.jsonl → Parse → chunk01_parsed.csv
├─> Download chunk02_results.jsonl → Parse → chunk02_parsed.csv
├─> ...
├─> Download chunk17_results.jsonl → Parse → chunk17_parsed.csv
└─> Combine all chunks → sentiment_results_2022_2023.csv

Phase 3: Use combined results
└─> sentiment_results_2022_2023.csv (all 311 speeches together)
```

### Validation Checks

The validator automatically checks:
- ✅ All required fields present
- ✅ Scores within valid ranges (-100 to +100, 0 to 100, etc.)
- ✅ No invalid values or nonsense
- ✅ Distributions look reasonable

**Target validation rate:** >95%

See [docs/PHASE2.md](docs/PHASE2.md) for detailed documentation.

---

## Phase 3: Build Sentiment Indices & Visualizations (✅ Complete)

### Goals

1. ✅ Build daily time series indices from parsed sentiment data
2. ✅ Create both forward-filled (continuous) and non-filled (sparse) versions
3. ✅ Generate comprehensive visualizations for analysis
4. ✅ Validate hawkish/dovish scores against actual speeches

### What You Get

After running Phase 3:

- **Daily Indices**: Time series for all sentiment metrics
  - Forward-filled: Continuous daily series (no gaps)
  - Non-filled: Only dates with actual speeches
- **Visualizations**: Three distinct chart types
  - Bar charts: Compare metrics on speech dates
  - Area plots: Show trends over time with continuous data
  - Calendar heatmaps: Visualize sparse data in calendar format
- **Validation Report**: Assessment of score accuracy vs actual speech content

### Files Created

```
data/results/
├── phase3_prepared_data.csv          # Combined data with speech IDs
├── fed_daily_indices.csv             # Fed indices (forward-filled)
├── fed_daily_indices_no_fill.csv     # Fed indices (sparse)
├── ecb_daily_indices.csv             # ECB indices (forward-filled)
└── ecb_daily_indices_no_fill.csv     # ECB indices (sparse)

reports/
├── fed_policy_metrics_bars.png       # Fed bar charts
├── fed_topic_indices_bars.png
├── fed_market_impact_bars.png
├── fed_policy_metrics_area.png       # Fed area plots
├── fed_topic_indices_area.png
├── fed_policy_metrics_calendar.png   # Fed calendar heatmaps
├── fed_topic_indices_calendar.png
├── fed_market_impact_calendar.png
├── ecb_*.png                         # Corresponding ECB charts
└── hawkish_dovish_validation.txt     # Validation report
```

### How to Run

**Step 1: Build Daily Indices**
```bash
# Build forward-filled indices (continuous time series)
python phase3_build_indices.py

# Build non-filled indices (sparse, speech dates only)
python phase3_build_indices_no_fill.py
```

**Step 2: Create Visualizations**
```bash
# Bar charts (uses non-filled data)
python phase3_visualize_indices_bars.py

# Area plots (uses forward-filled data)
python phase3_visualize_indices_area.py

# Calendar heatmaps (uses non-filled data)
python phase3_visualize_indices_dayplot.py
```

**Step 3: Validate Scores**
```bash
# Compare hawkish/dovish scores with actual speeches
python validate_hawkish_dovish.py
```

### Visualization Details

**1. Bar Charts** (`phase3_visualize_indices_bars.py`)
- Shows values only on dates with speeches
- Diverging bars for metrics with neutral values:
  - Hawkish/Dovish (centered on 0): Blue=hawkish, Red=dovish
  - Market Impact (centered on 50): Green=bullish, Red=bearish
- Standard bars for other metrics (topics, uncertainty, forward guidance)
- Clean styling with removed spines for market impact charts

**2. Area Plots** (`phase3_visualize_indices_area.py`)
- Continuous time series using forward-filled data
- Filled areas with colored lines on top
- Special diverging colors for hawkish/dovish (red above 0, blue below 0)
- Y-axis: 0-100 for most metrics, -100 to 100 for hawkish/dovish

**3. Calendar Heatmaps** (`phase3_visualize_indices_dayplot.py`)
- Full calendar years (2022, 2023) with sparse data
- Uses dayplot library for professional calendar layouts
- Diverging colormaps (coolwarm) for:
  - Hawkish/Dovish centered on 0
  - Market diffusion indices centered on 50
- Sequential colormaps (YlOrRd, Greens) for other metrics
- Grey cells indicate no speeches on that date

### Validation Results

The validation script checks hawkish/dovish scores by:
1. Finding outlier (most extreme) and random samples for Fed and ECB
2. Loading actual speech text from batch input files
3. Comparing scores against speech content
4. Generating assessment report

**Key Findings:**
- Score of 85.0 is REASONABLE for highly hawkish speeches (e.g., explicit 75bp rate hike commitments)
- Score of 50.0 captures balanced/neutral stances accurately
- LLM scoring captures policy actions and context, not just keywords
- Simple keyword counting can miss nuanced hawkish/dovish signals

See `reports/hawkish_dovish_validation.txt` for full validation report.

---

## Installation

### Prerequisites

- **Python 3.9 or higher**
- **OpenAI API Key** ([Get one here](https://platform.openai.com/api-keys))
- **Hugging Face Token** ([Get one here](https://huggingface.co/settings/tokens))
- **Git** (for version control)

### Step-by-Step Setup

**1. Clone the repository**

```bash
git clone <repository-url>
cd llm_sentiment
```

**2. Create a virtual environment** (recommended)

```bash
# On macOS/Linux:
python3 -m venv venv
source venv/bin/activate

# On Windows:
python -m venv venv
venv\Scripts\activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**Key dependencies include:**
- `openai` - OpenAI API client
- `datasets` - Hugging Face datasets library
- `pandas` - Data manipulation
- `matplotlib`, `seaborn` - Visualizations
- `dayplot` - Calendar heatmap visualizations
- `python-dotenv` - Environment variable management

**4. Set up your API keys**

```bash
# Copy the example file
cp .env.example .env

# Edit .env and add your API keys
# On macOS/Linux:
nano .env

# On Windows:
notepad .env
```

Add **both** keys to the `.env` file:
```
OPENAI_API_KEY=sk-your-actual-openai-key-here
HF_TOKEN=hf_your-actual-huggingface-token-here
```

**Where to get tokens:**
- OpenAI: https://platform.openai.com/api-keys
- Hugging Face: https://huggingface.co/settings/tokens (select "Read" access)

**5. Verify installation**

```bash
python src/config.py
```

You should see: `✅ Configuration is ready to use!`

---

## Quick Start

### Run Phase 1 (Two-Step Process)

Phase 1 is split into two scripts for flexibility:

**Step 1: Data Preparation**
```bash
python phase1_data_prep.py
```

This will:
1. Download the ECB-FED speeches dataset (~2-3 minutes first run)
2. Sample speeches from 2022-2023
3. Save sample data to CSV

**Step 2: Batch File Creation**
```bash
python phase1_batch_prep.py
```

This will:
1. Load the sample data (from Step 1)
2. Create batch processing file
3. Show cost comparison

**Why two scripts?**
- Run data prep once, create multiple batch variations
- Iterate on prompts without re-downloading data
- Clearer separation of concerns

**Example Output:**

```
========================================
BATCH PREPARATION COMPLETE ✅
========================================

📁 Files created:
   1. Batch file:   data/batch_input/batch_sample_2022_2023.jsonl
   2. Statistics:   data/results/phase1_statistics.json

💰 Cost Estimates:
   Batch API:     $12.50
   Real-time API: $25.00
   Savings:       $12.50 (50% discount)
```

**Legacy:** You can still run `python phase1_demo.py` to do everything in one go.

### Run Phase 3 (Visualizations)

After completing Phase 2 (batch processing), create visualizations:

**Step 1: Build Daily Indices**
```bash
python phase3_build_indices.py
python phase3_build_indices_no_fill.py
```

**Step 2: Create All Visualizations**
```bash
python phase3_visualize_indices_bars.py
python phase3_visualize_indices_area.py
python phase3_visualize_indices_dayplot.py
```

**Step 3: Validate Results**
```bash
python validate_hawkish_dovish.py
```

**What You'll See:**
- 18 visualization files in `reports/` directory
- Daily indices CSV files in `data/results/`
- Validation report comparing scores to actual speeches

### Inspect the Results

**View the sample data:**
```bash
# On macOS/Linux:
head -20 data/processed/sample_2022_2023.csv

# On Windows:
type data\processed\sample_2022_2023.csv | more
```

**View cost statistics:**
```bash
cat data/results/phase1_statistics.json
```

---

## Project Structure

```
llm_sentiment/
│
├── README.md                              # This file
├── requirements.txt                       # Python dependencies
├── .env.example                          # Template for API keys
├── .gitignore                            # Files to exclude from Git
│
├── phase1_data_prep.py                   # Phase 1: Data preparation
├── phase1_batch_prep.py                  # Phase 1: Batch file creation
├── phase1_demo.py                        # Phase 1: Combined (legacy)
│
├── phase2_batch_submit.py                # Phase 2: Submit batch jobs
├── phase2_check_status.py                # Phase 2: Check batch status
├── phase2_download_results.py            # Phase 2: Download results
│
├── phase3_build_indices.py               # Phase 3: Build daily indices (forward-filled)
├── phase3_build_indices_no_fill.py       # Phase 3: Build daily indices (sparse)
├── phase3_visualize_indices_bars.py      # Phase 3: Bar chart visualizations
├── phase3_visualize_indices_area.py      # Phase 3: Area plot visualizations
├── phase3_visualize_indices_dayplot.py   # Phase 3: Calendar heatmap visualizations
│
├── validate_hawkish_dovish.py            # Validation: Check scores vs speeches
│
├── src/                                  # Source code
│   ├── config.py                         # Configuration settings
│   ├── data_loader.py                    # Load Hugging Face dataset
│   ├── batch_builder.py                  # Create batch API files
│   ├── batch_processor.py                # Submit & monitor batch jobs
│   └── utils.py                          # Helper functions
│
├── data/                                 # Data files (created when you run)
│   ├── raw/                              # Downloaded datasets
│   ├── processed/                        # Cleaned/sampled data
│   ├── batch_input/                      # Files to upload to OpenAI
│   ├── batch_output/                     # Results from OpenAI
│   └── results/                          # Daily indices and parsed data
│
├── reports/                              # Visualization outputs
│   ├── *_bars.png                        # Bar chart reports
│   ├── *_area.png                        # Area plot reports
│   ├── *_calendar.png                    # Calendar heatmap reports
│   └── hawkish_dovish_validation.txt     # Validation report
│
├── notebooks/                            # Jupyter notebooks (for exploration)
│
└── config/                               # Configuration files
```

---

## For Beginners

### What is a "Batch API"?

Think of it like this:

**Real-time API** (traditional):
- You: "Analyze this speech" → OpenAI: "Here's the result" (instant)
- Like ordering fast food: immediate, but you pay full price
- Good for: One or a few speeches

**Batch API**:
- You: "Here are 100 speeches, analyze them all" → OpenAI: "I'll work on it and email you when done" (24 hours max)
- Like ordering catering: takes longer, but you get a bulk discount
- Good for: Hundreds or thousands of speeches

### Key Concepts

**Tokens**
- How OpenAI measures text length
- Roughly: 1 token ≈ 4 characters ≈ 0.75 words
- Example: "Hello world" ≈ 2 tokens

**Prompt**
- The instructions you give to GPT
- Like hiring an analyst: the better your instructions, the better the analysis

**JSONL**
- JSON Lines format
- One JSON object per line
- Required format for Batch API

**Sentiment**
- The overall tone or stance
- In central banking: "Hawkish" (tightening) vs "Dovish" (easing)

**Chunking**
- Breaking a large job into smaller pieces
- **Why necessary**: OpenAI limits how much work you can queue at once (90,000 tokens)
- Like a restaurant kitchen that can only accept 10 orders at a time - you need to submit in batches
- In this project: 311 speeches split into 17 chunks (~18 speeches each)
- Each chunk stays under the 90,000 token limit

### Common Issues & Solutions

**Issue**: `OPENAI_API_KEY not found`
- **Solution**: Make sure you created `.env` file and added your API key

**Issue**: `ModuleNotFoundError: No module named 'datasets'`
- **Solution**: Run `pip install -r requirements.txt`

**Issue**: `Permission denied` when running script
- **Solution**: On macOS/Linux, run `chmod +x phase1_demo.py` first

**Issue**: Download is slow
- **Solution**: First download takes time (downloading dataset). Subsequent runs use cached version.

**Issue**: `Enqueued token limit reached for gpt-4o-2024-08-06`
- **Solution**: This is why we use chunking! The scripts automatically split your data into chunks to stay under the 90,000 token limit. If you see this error, your batch file is too large - increase the number of chunks or reduce batch size.

---

## Cost Estimates

### Phase 1 (2 years of data)

Typical costs for 2022-2023 sample (estimates may vary):

| API Type | Input Tokens | Output Tokens | Total Cost |
|----------|-------------|---------------|------------|
| Batch API | 5,000,000 | 1,000,000 | ~$11.25 |
| Real-time API | 5,000,000 | 1,000,000 | ~$22.50 |
| **Savings** | - | - | **~$11.25 (50%)** |

*Based on GPT-4o pricing as of 2024. Actual costs depend on speech length.*

### Full Dataset (1996-2025)

Estimated costs for complete analysis:

- **Batch API**: ~$75-150
- **Real-time API**: ~$150-300
- **Savings**: ~$75-150

**Why the range?**
- Speeches vary in length
- Some years have more speeches than others
- Actual token usage may differ from estimates

### Cost Control Tips

✅ **Always test with Phase 1 first** - See actual costs before scaling up
✅ **Check estimates** - The script shows estimates before uploading
✅ **Use batch API** - Automatic 50% discount
✅ **Monitor usage** - Check your OpenAI dashboard regularly

---

## Documentation

### Module Documentation

Each Python module has detailed documentation:

```python
# To see documentation for a module:
python -c "import src.data_loader; help(src.data_loader.DataLoader)"
```

### Code Comments

Every function includes:
- **Purpose**: What it does
- **Parameters**: What inputs it needs
- **Returns**: What it gives back
- **For beginners** section: Plain English explanation

Example:
```python
def load_from_huggingface(self) -> pd.DataFrame:
    """
    Load the dataset from Hugging Face.

    Returns:
        DataFrame containing all speeches with metadata

    For beginners:
    - This function first checks if we already have the data saved locally
    - If yes, it loads from local file (much faster!)
    - If no, it downloads from Hugging Face and saves for next time
    """
```

### Getting Help

**In the code:**
- Read the comments (lines starting with `#`)
- Read the docstrings (text in `"""triple quotes"""`)
- Look for "For beginners:" sections

**External resources:**
- [OpenAI Batch API Documentation](https://platform.openai.com/docs/guides/batch)
- [Hugging Face Datasets](https://huggingface.co/docs/datasets/)
- [Pandas Tutorial](https://pandas.pydata.org/docs/getting_started/intro_tutorials/index.html)

---

## Research Background

### Literature This Project Builds On

1. **Liao et al. (2024)** - Comparison of LLMs for central bank sentiment
2. **IMF (2025)** - Large-scale analysis of 169 central banks
3. **ECB Working Papers** - Hawkish/dovish classification methods
4. **Federal Reserve (2024)** - Using GPT for FOMC minutes analysis

See the full literature review in the project documentation.

### Novel Contributions

This project advances research by:

✅ **Democratization**: Making advanced analysis accessible without expensive training data
✅ **Multi-dimensional**: Extracting 5+ dimensions vs. single hawkish/dovish score
✅ **Cross-country**: Consistent methodology for Fed AND ECB
✅ **Reproducible**: All code and data sources are open

---

## Contributing

This is a research project organized in phases. Each phase has:

1. **Clear objectives** - What we're trying to accomplish
2. **Implementation** - Code to achieve objectives
3. **Validation** - Tests to ensure quality
4. **Pull Request** - Review before moving forward

### Current Phase Checklist

**Phase 1 - Setup & Batch Processing** ✅
- [x] Project structure created
- [x] Data loader implemented
- [x] Batch processing modules created
- [x] Cost comparison functional
- [x] Documentation complete
- [x] Run demo and validate results

**Phase 2 - Batch Submission & Output** ✅
- [x] Batch submission implemented
- [x] Status monitoring functional
- [x] Results download and parsing
- [x] Output validation

**Phase 3 - Indices & Visualizations** ✅
- [x] Daily indices generation (forward-filled and sparse)
- [x] Bar chart visualizations
- [x] Area plot visualizations
- [x] Calendar heatmap visualizations
- [x] Hawkish/dovish score validation
- [x] Documentation updated

**Phase 4 - Market Correlation Analysis** (Next)
- [ ] Market data collection
- [ ] Correlation analysis
- [ ] Predictive model testing
- [ ] Results documentation

---

## License

See [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- **Dataset**: ECB-FED speeches from [istat-ai/ECB-FED-speeches](https://huggingface.co/datasets/istat-ai/ECB-FED-speeches)
- **API**: OpenAI's GPT-4o and Batch API
- **Inspiration**: Academic research on central bank communication analysis

---

## Contact

For questions about this research project, please open an issue on GitHub.

---

**Last Updated**: Phase 3 Complete - December 2025
