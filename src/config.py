"""
Configuration module for the LLM Sentiment Analysis project.

This module manages all configuration settings including:
- API credentials
- File paths
- Model parameters
- Prompt templates

For beginners:
- Configuration keeps all settings in one place
- This makes it easy to change settings without editing multiple files
- It also keeps secrets (like API keys) separate from code
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
# This is where we store secret information like API keys
load_dotenv()


class Config:
    """
    Configuration class that stores all project settings.

    Think of this like a settings menu for the entire project.
    Instead of hardcoding values throughout the code, we put them here.
    """

    # ==================== DIRECTORIES ====================
    # These are the folder paths where we store different types of files

    # Get the project root directory (the main folder)
    PROJECT_ROOT = Path(__file__).parent.parent

    # Data directories
    DATA_DIR = PROJECT_ROOT / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    BATCH_INPUT_DIR = DATA_DIR / "batch_input"
    BATCH_OUTPUT_DIR = DATA_DIR / "batch_output"
    RESULTS_DIR = DATA_DIR / "results"

    # Create directories if they don't exist
    for directory in [
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        BATCH_INPUT_DIR,
        BATCH_OUTPUT_DIR,
        RESULTS_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)

    # ==================== OPENAI API SETTINGS ====================
    # Settings for connecting to OpenAI's API

    # Your OpenAI API key (loaded from .env file for security)
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    # Which GPT model to use
    # gpt-4o-2024-08-06 is good for structured outputs (JSON format)
    MODEL_NAME = "gpt-4o-2024-08-06"

    # Temperature controls randomness (0 = deterministic, 1 = creative)
    # We use 0.3 for consistency while allowing some flexibility
    TEMPERATURE = 0.3

    # ==================== HUGGING FACE SETTINGS ====================
    # Settings for accessing Hugging Face datasets

    # Your Hugging Face token (loaded from .env file for security)
    # Required to access gated/private datasets like ECB-FED speeches
    HF_TOKEN = os.getenv("HF_TOKEN")

    # ==================== DATASET SETTINGS ====================
    # Settings for the Hugging Face dataset we're using

    # The dataset on Hugging Face with ECB and Fed speeches
    DATASET_NAME = "istat-ai/ECB-FED-speeches"

    # For Phase 1: which years to sample for testing
    # We'll use 2022-2023 (2 years) as a manageable test sample
    SAMPLE_START_YEAR = 2022
    SAMPLE_END_YEAR = 2023

    # ==================== BATCH PROCESSING SETTINGS ====================
    # Settings specific to OpenAI's Batch API

    # How long to wait between checking if batch is complete (in seconds)
    BATCH_CHECK_INTERVAL = 60  # Check every 60 seconds

    # Maximum time to wait for batch completion (in seconds)
    # 24 hours = 86400 seconds
    BATCH_TIMEOUT = 86400

    # ==================== PROMPT TEMPLATES ====================

    @staticmethod
    def get_sentiment_prompt(
        speech_text: str, speaker: str, institution: str, date: str
    ) -> str:
        """
        Creates the prompt that tells GPT-4 how to analyze a speech.

        This is like writing instructions for a research assistant.
        The better the instructions, the better the analysis.

        Args:
            speech_text: The actual speech text to analyze
            speaker: Who gave the speech (e.g., "Jerome Powell")
            institution: Which central bank (e.g., "Federal Reserve")
            date: When the speech was given (e.g., "2023-05-15")

        Returns:
            A detailed prompt string for GPT-4
        """

        prompt = f"""You are an expert in central bank communication analysis. Your task is to analyze speeches from Federal Reserve and ECB officials and extract monetary policy sentiment indicators used by financial market participants and economists.

**Speech Information:**
- Speaker: {speaker}
- Institution: {institution}
- Date: {date}

**Speech Text:**
{speech_text}

**Your Task:**
Analyze this speech and extract the following information. Be objective and base your analysis on the actual content.

**1. Hawkish/Dovish Score (-100 to +100):**
Rate the overall monetary policy stance:
- -100 = Extremely dovish (strongly favoring lower rates, more accommodation)
- 0 = Neutral (balanced, no clear directional bias)
- +100 = Extremely hawkish (strongly favoring higher rates, less accommodation)

Consider:
- Language about inflation (concerns = hawkish, moderating = dovish)
- Interest rate intentions (raising = hawkish, cutting = dovish)
- Economic risks (upside inflation risks = hawkish, downside growth risks = dovish)
- Urgency of action (immediate = more extreme, gradual = more moderate)

**2. Topic Emphasis (0-100 for each):**
How much emphasis does the speaker place on each topic?
- Inflation concerns: 0 (not mentioned) to 100 (major focus)
- Economic growth outlook: 0 to 100
- Financial stability risks: 0 to 100
- Labor market conditions: 0 to 100
- International/trade issues: 0 to 100

**3. Uncertainty Level (0-100):**
How certain is the speaker about the economic outlook and policy path?
- 0 = Very confident (clear statements, definitive language)
- 100 = Highly uncertain (many conditional statements, "depends on", "might", "could")

**4. Forward Guidance Strength (0-100):**
How explicit is guidance about future policy actions?
- 0 = No forward guidance (only describes current conditions)
- 50 = Implicit guidance (hints at future direction)
- 100 = Explicit commitment with timeline (e.g., "rates will remain elevated through 2024")

**5. Key Sentences:**
Extract 2-3 direct quotes that best support your hawkish/dovish score.

**6. Market Impact Prediction:**
Based on this speech, predict the likely immediate market reaction:
- Stock market: EXACTLY "rise" or "fall" or "neutral" (no other words allowed)
- Bond yields: EXACTLY "rise" or "fall" or "neutral" (no other words allowed)
- Currency (USD for Fed, EUR for ECB): EXACTLY "rise" or "fall" or "neutral"

CRITICAL: For currency, you MUST use ONLY these three exact words: "rise", "fall", or "neutral"
DO NOT use words like "strengthen", "weaken", "appreciate", "depreciate", "up", "down", etc.
ONLY use: "rise" (for strengthening), "fall" (for weakening), or "neutral"

- Brief reasoning: One sentence explaining your prediction

**Output Format:**
Respond ONLY with valid JSON in this exact format (no markdown, no code blocks):

{{
  "hawkish_dovish_score": <number from -100 to 100>,
  "topics": {{
    "inflation": <number 0-100>,
    "growth": <number 0-100>,
    "financial_stability": <number 0-100>,
    "labor_market": <number 0-100>,
    "international": <number 0-100>
  }},
  "uncertainty": <number 0-100>,
  "forward_guidance_strength": <number 0-100>,
  "key_sentences": ["sentence 1", "sentence 2", "sentence 3"],
  "market_impact": {{
    "stocks": "rise",
    "bonds": "fall",
    "currency": "rise"
  }},
  "reasoning": "brief explanation",
  "summary": "One paragraph summarizing the main policy message and stance"
}}

IMPORTANT: The market_impact values MUST be exactly "rise", "fall", or "neutral" - no other words!"""

        return prompt

    # ==================== COST CALCULATION ====================
    # Pricing for GPT-4o (as of 2024)
    # These prices are per 1,000 tokens

    # Real-time API pricing
    REALTIME_INPUT_PRICE_PER_1K = 0.0025  # $0.0025 per 1K input tokens
    REALTIME_OUTPUT_PRICE_PER_1K = 0.010  # $0.010 per 1K output tokens

    # Batch API pricing (50% discount)
    BATCH_INPUT_PRICE_PER_1K = 0.00125  # $0.00125 per 1K input tokens
    BATCH_OUTPUT_PRICE_PER_1K = 0.005  # $0.005 per 1K output tokens

    @staticmethod
    def validate():
        """
        Check if configuration is valid before running the project.

        This is like a pre-flight checklist - making sure everything
        is set up correctly before we start.

        Raises:
            ValueError: If configuration is invalid
        """

        if not Config.OPENAI_API_KEY:
            raise ValueError(
                "OPENAI_API_KEY not found in environment variables. "
                "Please create a .env file with your API key. "
                "See .env.example for the format."
            )

        if not Config.HF_TOKEN:
            raise ValueError(
                "HF_TOKEN not found in environment variables. "
                "Please add your Hugging Face token to the .env file. "
                "Get a token from: https://huggingface.co/settings/tokens"
            )

        print("✓ Configuration validated successfully")
        print(f"✓ Project root: {Config.PROJECT_ROOT}")
        print(f"✓ Using model: {Config.MODEL_NAME}")
        print(f"✓ Sample period: {Config.SAMPLE_START_YEAR}-{Config.SAMPLE_END_YEAR}")
        print(f"✓ Hugging Face authentication: Configured")


# When this file is run directly, validate the configuration
if __name__ == "__main__":
    try:
        Config.validate()
        print("\n✅ Configuration is ready to use!")
    except ValueError as e:
        print(f"\n❌ Configuration error: {e}")
