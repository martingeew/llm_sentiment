# Setup Instructions for Phase 1

## Quick Setup (5 minutes)

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- `pandas` - Data manipulation
- `datasets` - Hugging Face dataset library
- `openai` - OpenAI API client
- `python-dotenv` - Environment variable management
- And other supporting libraries

### 2. Configure OpenAI API Key

**Create your `.env` file:**

```bash
cp .env.example .env
```

**Edit `.env` and add your API key:**

```bash
# On macOS/Linux:
nano .env

# On Windows:
notepad .env
```

**Add your key:**
```
OPENAI_API_KEY=sk-proj-your-actual-key-here
```

**Get an API key:**
- Go to: https://platform.openai.com/api-keys
- Click "Create new secret key"
- Copy and paste into your `.env` file

**Important**: Never commit your `.env` file to git (it's in `.gitignore`)

### 3. Verify Setup

```bash
python src/config.py
```

Expected output:
```
✓ Configuration validated successfully
✓ Project root: /path/to/llm_sentiment
✓ Using model: gpt-4o-2024-08-06
✓ Sample period: 2022-2023

✅ Configuration is ready to use!
```

### 4. Run Phase 1 Demo

```bash
python phase1_demo.py
```

This will:
1. Download the ECB-FED dataset (first run takes 2-3 minutes)
2. Sample 2022-2023 data
3. Create batch processing files
4. Show cost estimates

**No charges yet!** Phase 1 only prepares files, it doesn't submit to OpenAI.

---

## Troubleshooting

### "OPENAI_API_KEY not found"

**Problem**: The `.env` file wasn't created or the key wasn't added.

**Solution**:
```bash
# Check if .env exists
ls -la .env

# If not, create it
cp .env.example .env

# Edit and add your key
nano .env
```

### "ModuleNotFoundError"

**Problem**: Dependencies not installed.

**Solution**:
```bash
# Make sure you're in the project directory
cd llm_sentiment

# Install dependencies
pip install -r requirements.txt

# If using a virtual environment, make sure it's activated
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows
```

### "Failed to download dataset"

**Problem**: Network issues or Hugging Face is down.

**Solution**:
- Check your internet connection
- Try again later
- The dataset is cached after first download

### "Permission denied"

**Problem**: Script doesn't have execute permissions.

**Solution**:
```bash
# On macOS/Linux:
chmod +x phase1_demo.py
```

---

## First Run Expectations

### What Happens

1. **Configuration Check** (~1 second)
   - Validates your API key exists
   - Checks directory structure

2. **Dataset Download** (~2-3 minutes first time)
   - Downloads ECB-FED speeches from Hugging Face
   - ~50-100 MB download
   - Cached for future runs (subsequent runs ~5 seconds)

3. **Data Sampling** (~5 seconds)
   - Filters 2022-2023 speeches
   - Typically 50-150 speeches

4. **Batch File Creation** (~10 seconds)
   - Creates JSONL file
   - Estimates costs

5. **Summary** (~1 second)
   - Shows statistics
   - Displays cost comparison

**Total time (first run)**: ~3-4 minutes
**Total time (subsequent runs)**: ~20 seconds

### What Gets Created

```
data/
├── raw/
│   └── ecb_fed_speeches.parquet         # Cached dataset
├── processed/
│   └── sample_2022_2023.csv             # Your sample
├── batch_input/
│   └── batch_sample_2022_2023.jsonl     # Ready for OpenAI
└── results/
    └── phase1_statistics.json           # Cost estimates
```

---

## Costs (Phase 1 Only)

### Phase 1: NO CHARGES

Phase 1 **prepares files** but **does not submit to OpenAI**.

You can:
✅ Run the demo multiple times
✅ Inspect the batch files
✅ Review cost estimates
✅ Validate everything works

**No API calls are made in Phase 1.**

### Future Phases

When you're ready to actually process the data:

**Estimated Phase 1 cost** (2 years of data):
- Batch API: ~$10-15
- Real-time API: ~$20-30
- **Savings: ~$10-15 (50% discount)**

You'll be prompted before any charges occur.

---

## Next Steps After Phase 1

Once Phase 1 is complete and validated:

1. **Review the batch file**
   ```bash
   head data/batch_input/batch_sample_2022_2023.jsonl
   ```

2. **Check the cost estimates**
   ```bash
   cat data/results/phase1_statistics.json
   ```

3. **If everything looks good**, proceed to actually submitting the batch
   (This will be a separate script in Phase 2)

---

## Getting Help

**Check the logs**: The demo script prints detailed progress

**Validate modules work**:
```bash
# Test each module individually
python src/config.py
python src/data_loader.py
python src/utils.py
```

**Python version**:
```bash
python --version
# Should be 3.9 or higher
```

**Dependencies**:
```bash
pip list | grep -E "pandas|datasets|openai"
```

---

**Ready to start?**

```bash
pip install -r requirements.txt
cp .env.example .env
# (Edit .env with your API key)
python phase1_demo.py
```
