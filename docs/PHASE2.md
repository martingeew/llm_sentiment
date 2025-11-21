# Phase 2: Batch Job Submission & LLM Output Validation

## Overview

Phase 2 takes the batch file created in Phase 1 and:
1. **Submits** it to OpenAI's Batch API
2. **Monitors** processing progress
3. **Downloads** results when complete
4. **Parses** LLM outputs into structured data
5. **Validates** output quality

**Expected Duration:** 30 minutes to 4 hours (typically)
**Expected Cost:** ~$2.32 for 2022-2023 sample (311 speeches)

---

## 🚨 Important Notes

### This Phase Makes Real API Calls

- ⚠️ **Charges will apply** - Review cost estimates from Phase 1 first
- ⚠️ **Processing takes time** - Up to 24 hours (usually 30min-4hrs)
- ✅ **Can stop and resume** - Batch continues processing on OpenAI's servers

### Safety Features

- Confirmation prompt before submission
- Batch ID saved for resuming later
- No charges if you abort before submitting

---

## 📋 Prerequisites

Before running Phase 2:

1. ✅ **Phase 1 completed** - Batch file must exist at:
   - `data/batch_input/batch_sample_2022_2023.jsonl`

2. ✅ **API keys configured** - `.env` file must contain:
   - `OPENAI_API_KEY`
   - `HF_TOKEN`

3. ✅ **Budget approved** - You're comfortable with the cost estimate

---

## 🚀 Running Phase 2

### Option 1: Submit and Monitor (Recommended for Small Batches)

```bash
python phase2_batch_submit.py
```

**What happens:**
1. Asks for confirmation ("Type 'YES' to proceed")
2. Uploads batch file to OpenAI
3. Submits batch job
4. Asks if you want to monitor now
5. If yes: Monitors progress until complete
6. Downloads and validates results

**When to use:**
- Small batches (expected completion < 1 hour)
- You can keep terminal open
- You want immediate results

### Option 2: Submit and Check Later (Recommended for Large Batches)

**Step 1: Submit the batch**
```bash
python phase2_batch_submit.py
```

When asked "Monitor now? (Y/n)", type `n`

**Step 2: Check status later**
```bash
python phase2_check_status.py
```

Run this as many times as you want to check progress.

**When to use:**
- Large batches (expected completion > 1 hour)
- You want to close terminal
- You'll check back later

---

## 📁 Files Created

After Phase 2 completes:

```
data/
├── batch_output/
│   └── batch_results_2022_2023.jsonl     # Raw results from OpenAI
└── results/
    ├── sentiment_results_2022_2023.csv   # Parsed sentiment scores
    ├── phase2_batch_info.json            # Batch ID and status
    └── phase2_validation_report.json     # Quality validation report
```

---

## 📊 Understanding the Results

### Raw Results (`batch_results_2022_2023.jsonl`)

- JSONL format (one result per line)
- Contains complete LLM responses
- Useful for debugging

### Parsed Results (`sentiment_results_2022_2023.csv`)

**Columns:**
- `speech_id` - Unique identifier
- `hawkish_dovish_score` - Policy stance (-100 to +100)
- `topic_inflation` - Inflation emphasis (0-100)
- `topic_growth` - Growth emphasis (0-100)
- `topic_financial_stability` - Financial stability (0-100)
- `topic_labor_market` - Labor market (0-100)
- `topic_international` - International issues (0-100)
- `uncertainty` - Uncertainty level (0-100)
- `forward_guidance_strength` - Guidance clarity (0-100)
- `market_impact_stocks` - Predicted stock reaction
- `market_impact_bonds` - Predicted bond reaction
- `market_impact_currency` - Predicted currency reaction
- `market_impact_reasoning` - Brief explanation
- `summary` - One-paragraph summary
- `key_sentences` - Important quotes (pipe-separated)

### Validation Report (`phase2_validation_report.json`)

**What it checks:**
- ✅ All required fields present
- ✅ Scores within valid ranges
- ✅ No invalid values
- ✅ Distributions look reasonable

**Example:**
```json
{
  "validation_results": {
    "total_speeches": 311,
    "valid_speeches": 308,
    "invalid_speeches": 3,
    "validation_rate": 99.0
  },
  "distributions": {
    "hawkish_dovish": {
      "mean": 15.3,
      "median": 12.0,
      "very_dovish": 12,
      "dovish": 45,
      "neutral": 89,
      "hawkish": 123,
      "very_hawkish": 42
    }
  }
}
```

---

## 🔍 Validation Checks

### Automatic Validation

The validator checks:

**1. Field Presence**
- All required fields exist
- No missing data

**2. Score Ranges**
- Hawkish/dovish score: -100 to +100
- Topic scores: 0 to 100
- Uncertainty: 0 to 100
- Forward guidance: 0 to 100

**3. Value Types**
- Numeric fields are numbers
- Text fields are strings
- Market impact uses valid categories

**4. Distribution Sanity**
- Scores have reasonable variation
- Not all extremely hawkish or dovish
- No obvious patterns suggesting errors

### Expected Validation Rate

- ✅ **>95%**: Excellent quality
- ⚠️ **85-95%**: Good, minor issues
- ❌ **<85%**: Review required

---

## 🐛 Troubleshooting

### "Batch file not found"

**Problem:** Phase 1 not completed

**Solution:**
```bash
# Run Phase 1 first
python phase1_data_prep.py
python phase1_batch_prep.py
```

### "OPENAI_API_KEY not found"

**Problem:** API key not configured

**Solution:**
```bash
# Create .env file with your keys
cp .env.example .env
nano .env
```

### Batch job fails

**Problem:** OpenAI API error

**Solution:**
1. Check OpenAI dashboard for error details
2. Verify API key is valid and has credits
3. Check if service is down: https://status.openai.com/
4. Wait a few minutes and try again

### Low validation rate

**Problem:** Many outputs don't pass validation

**Solution:**
1. Check `phase2_validation_report.json` for common errors
2. Review a few failed speeches in the CSV
3. May indicate prompt needs refinement
4. Or dataset has unusual characteristics

---

## ⏱️ Timing Expectations

Based on OpenAI's Batch API:

| Batch Size | Typical Time | Maximum Time |
|------------|--------------|--------------|
| 100 speeches | 20-40 min | 4 hours |
| 300 speeches | 30-90 min | 6 hours |
| 1000 speeches | 1-4 hours | 12 hours |
| 5000+ speeches | 4-12 hours | 24 hours |

**Note:** These are estimates. Actual time varies based on:
- OpenAI's current load
- Time of day
- Speech length
- Prompt complexity

---

## 💰 Cost Tracking

### Where to Check Costs

1. **Before submission:** Phase 1 statistics
   - `data/results/phase1_statistics.json`

2. **After completion:** OpenAI dashboard
   - https://platform.openai.com/usage

3. **Calculate actual cost:**
   ```python
   # From the CSV
   import pandas as pd
   df = pd.read_csv('data/results/sentiment_results_2022_2023.csv')
   num_speeches = len(df)

   # Estimate (actual usage shown in OpenAI dashboard)
   print(f"Processed {num_speeches} speeches")
   ```

### Cost Breakdown

For 311 speeches (2022-2023 sample):

```
Input tokens:  1,235,379 × $0.00125/1K = $1.54
Output tokens:   155,500 × $0.00500/1K = $0.78
---------------------------------------------------
Total:                                    $2.32
```

---

## 📈 Next Steps

After Phase 2 completes successfully:

1. **Review validation report**
   - Check validation rate
   - Review any errors

2. **Inspect sample results**
   - Open `sentiment_results_2022_2023.csv`
   - Read a few speeches and their scores
   - Verify they make sense

3. **If quality is good → Phase 3**
   - Build time series indices
   - Aggregate by month/quarter
   - Visualize trends

4. **If quality issues → Refine**
   - Adjust prompts
   - Re-run Phase 1 and 2
   - May need to iterate

---

## 🔄 Re-running Phase 2

If you need to re-run Phase 2:

**Option 1: New batch submission**
```bash
# This will create a NEW batch job
python phase2_batch_submit.py
```

**Option 2: Use existing batch**
```bash
# If batch is still processing or complete
python phase2_check_status.py
```

**Costs:**
- Checking status: **FREE**
- New submission: **~$2.32 per batch**

---

## 📞 Getting Help

**If results look wrong:**
1. Check validation report for clues
2. Review Phase 2 outputs in `reports/` directory
3. Inspect a few speeches manually
4. Compare with known events (e.g., 2022 inflation → should be hawkish)

**If batch fails:**
1. Check OpenAI status page
2. Verify API key and credits
3. Try again after a few minutes
4. Check batch status with `phase2_check_status.py`

---

## ✅ Phase 2 Success Criteria

Phase 2 is successful when:

- ✅ Batch completes without errors
- ✅ Validation rate >95%
- ✅ Score distributions look reasonable
- ✅ Sample reviews confirm quality
- ✅ Ready to proceed to Phase 3

---

**Last Updated:** Phase 2 - January 2025
