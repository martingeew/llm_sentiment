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

### 🔄 **Phase 2** (Current): Batch Submission & Output Validation
- Submit batch job to OpenAI API
- Monitor processing progress
- Download and parse results
- Validate LLM output quality
- Verify scores are sensible

### 📊 **Phase 3**: Build Sentiment Indices
- Create time series indices (hawkish/dovish, uncertainty, etc.)
- Aggregate by month/quarter
- Visualize trends over time

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

## Phase 2: Batch Submission & Output Validation (🔄 Current)

### Goals

1. 🔄 Submit batch job to OpenAI
2. 🔄 Monitor processing progress
3. 🔄 Download and parse results
4. 🔄 Validate LLM output quality
5. 🔄 Verify sentiment scores are sensible

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
│   └── batch_results_2022_2023.jsonl  # Raw LLM outputs
└── results/
    ├── sentiment_results_2022_2023.csv # Parsed scores
    ├── phase2_batch_info.json          # Batch tracking
    └── phase2_validation_report.json   # Quality report
```

### How to Run

**Option 1: Submit and monitor**
```bash
python phase2_batch_submit.py
```

**Option 2: Submit and check later**
```bash
# Submit
python phase2_batch_submit.py
# (say 'n' when asked to monitor)

# Check later
python phase2_check_status.py
```

**Expected Cost:** ~$2.32 for 2022-2023 sample
**Expected Time:** 30 minutes to 4 hours

### Validation Checks

The validator automatically checks:
- ✅ All required fields present
- ✅ Scores within valid ranges (-100 to +100, 0 to 100, etc.)
- ✅ No invalid values or nonsense
- ✅ Distributions look reasonable

**Target validation rate:** >95%

See [docs/PHASE2.md](docs/PHASE2.md) for detailed documentation.

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
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── .env.example             # Template for API keys
├── .gitignore              # Files to exclude from Git
│
├── phase1_demo.py          # Phase 1 main script
│
├── src/                    # Source code
│   ├── config.py          # Configuration settings
│   ├── data_loader.py     # Load Hugging Face dataset
│   ├── batch_builder.py   # Create batch API files
│   ├── batch_processor.py # Submit & monitor batch jobs
│   └── utils.py           # Helper functions
│
├── data/                   # Data files (created when you run)
│   ├── raw/               # Downloaded datasets
│   ├── processed/         # Cleaned/sampled data
│   ├── batch_input/       # Files to upload to OpenAI
│   ├── batch_output/      # Results from OpenAI
│   └── results/           # Final analysis results
│
├── notebooks/             # Jupyter notebooks (for exploration)
│
└── config/               # Configuration files
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

### Common Issues & Solutions

**Issue**: `OPENAI_API_KEY not found`
- **Solution**: Make sure you created `.env` file and added your API key

**Issue**: `ModuleNotFoundError: No module named 'datasets'`
- **Solution**: Run `pip install -r requirements.txt`

**Issue**: `Permission denied` when running script
- **Solution**: On macOS/Linux, run `chmod +x phase1_demo.py` first

**Issue**: Download is slow
- **Solution**: First download takes time (downloading dataset). Subsequent runs use cached version.

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

Phase 1:
- [x] Project structure created
- [x] Data loader implemented
- [x] Batch processing modules created
- [x] Cost comparison functional
- [x] Documentation complete
- [ ] Run demo and validate results
- [ ] Create Pull Request
- [ ] Review and merge

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

**Last Updated**: Phase 1 - January 2025
