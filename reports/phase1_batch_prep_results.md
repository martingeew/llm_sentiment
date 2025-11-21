# Phase 1 Batch Preparation - Results

======================================================================
               PHASE 1 - PART 2: BATCH FILE PREPARATION
            Central Bank Communication Sentiment Analysis
======================================================================

============================================================
               STEP 1: VALIDATE CONFIGURATION
============================================================
✓ Configuration validated successfully
✓ Project root: C:\Users\marti\py_projects\llm_sentiment
✓ Using model: gpt-4o-2024-08-06
✓ Sample period: 2022-2023
✓ Hugging Face authentication: Configured

============================================================
                    LOADING SAMPLE DATA
============================================================
📂 Loading sample data from: C:\Users\marti\py_projects\llm_sentiment\data\processed\sample_2022_2023.csv
✓ Loaded 311 speeches

============================================================
              STEP 4: BUILD BATCH REQUEST FILE
============================================================

📋 Dataset columns available:
   - date
   - author
   - country
   - title
   - description
   - text
   - mistral_ocr
   - clean_text
   - url
   - year

✓ Using 'clean_text' column for speech content
✓ Created 'id' column

📋 Column mapping:
   Speech text:  clean_text
   Speaker:      author
   Institution:  country
   Date:         date
   ID:           id

============================================================
                BUILDING BATCH REQUEST FILE
============================================================

📝 Creating batch requests for 311 speeches...
Creating requests: |███-------------------------------------| 8% 26/311 speeches   ⚠️  Truncated speech speech_26 (was 9905 tokens)
Creating requests: |████████████----------------------------| 31% 98/311 speeches   ⚠️  Truncated speech speech_98 (was 11273 tokens)
Creating requests: |███████████████████---------------------| 49% 155/311 speeches   ⚠️  Truncated speech speech_155 (was 12410 tokens)
Creating requests: |█████████████████████████████████████---| 92% 289/311 speeches   ⚠️  Truncated speech speech_289 (was 9472 tokens)
Creating requests: |████████████████████████████████████████| 100% 311/311 speeches

💾 Writing 311 requests to C:\Users\marti\py_projects\llm_sentiment\data\batch_input\batch_sample_2022_2023.jsonl...
✓ Batch file created successfully!

============================================================
BATCH FILE SUMMARY
============================================================

📊 Requests: 311
📏 Total input tokens: 1,235,379
📏 Estimated output tokens: 155,500

💰 Cost Estimates:
   Batch API:    $2.32
   Real-time API: $4.64
   💵 Savings:    $2.32 (50% discount)

📁 File: C:\Users\marti\py_projects\llm_sentiment\data\batch_input\batch_sample_2022_2023.jsonl
============================================================

============================================================
                   VALIDATING BATCH FILE
============================================================

🔍 Validating batch file: C:\Users\marti\py_projects\llm_sentiment\data\batch_input\batch_sample_2022_2023.jsonl
✓ Batch file is valid! (311 requests)

============================================================
                  STEP 5: COST COMPARISON
============================================================

💰 DETAILED COST BREAKDOWN
============================================================

📊 Processing 311 speeches
   Input tokens:  1,235,379
   Output tokens: 155,500 (estimated)

💵 BATCH API (50% discount):
   Input cost:  $1.54
   Output cost: $0.78
   TOTAL:       $2.32

💵 REAL-TIME API (standard pricing):
   Input cost:  $3.09
   Output cost: $1.55
   TOTAL:       $4.64

✅ SAVINGS WITH BATCH API:
   $2.32 (50% discount)
✓ Saved JSON to: C:\Users\marti\py_projects\llm_sentiment\data\results\phase1_statistics.json

============================================================
                BATCH PREPARATION COMPLETE ✅
============================================================

📁 Files created:
   1. Batch file:   C:\Users\marti\py_projects\llm_sentiment\data\batch_input\batch_sample_2022_2023.jsonl
   2. Statistics:   C:\Users\marti\py_projects\llm_sentiment\data\results\phase1_statistics.json

🎯 What we've demonstrated:
   ✓ Created batch processing file with 311 requests
   ✓ Compared costs: $2.32 (batch) vs $4.64 (real-time)
   ✓ Validated batch file format

📊 Cost savings: $2.32 (50% discount)

📝 Next Steps:
   1. Review the batch file to ensure it looks correct
   2. Check the cost estimates are within your budget
   3. If ready, proceed to actually submit the batch job
      (This will be in Phase 2 - actual batch submission)

======================================================================