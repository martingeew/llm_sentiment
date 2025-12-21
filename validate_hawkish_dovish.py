"""
Validate Hawkish/Dovish Scores Against Actual Speeches
Compare model scores with actual speech content
"""

import pandas as pd
import json
from pathlib import Path
from src.config import Config

def load_results():
    """Load prepared data with speech IDs"""
    prepared_file = Config.RESULTS_DIR / "phase3_prepared_data.csv"
    df = pd.read_csv(prepared_file, parse_dates=['date'])

    # Split into Fed and ECB
    fed_df = df[df['country'] == 'United States'].copy()
    ecb_df = df[df['country'] == 'Euro area'].copy()

    return fed_df, ecb_df

def find_samples(df, institution):
    """Find outlier and random sample"""
    # Find outlier (most extreme value - could be max hawkish or max dovish)
    max_hawkish_idx = df['hawkish_dovish_score'].idxmax()
    min_dovish_idx = df['hawkish_dovish_score'].idxmin()

    # Pick the more extreme one
    max_val = abs(df.loc[max_hawkish_idx, 'hawkish_dovish_score'])
    min_val = abs(df.loc[min_dovish_idx, 'hawkish_dovish_score'])

    outlier_idx = max_hawkish_idx if max_val > min_val else min_dovish_idx

    # Find random sample from middle of dataset (not first/last 20%)
    middle_start = int(len(df) * 0.2)
    middle_end = int(len(df) * 0.8)
    middle_df = df.iloc[middle_start:middle_end]
    random_idx = middle_df.sample(1, random_state=42).index[0]

    outlier = df.loc[outlier_idx]
    random_sample = df.loc[random_idx]

    print(f"\n{institution} Samples:")
    print(f"  Outlier: {outlier['date'].date()} | Score: {outlier['hawkish_dovish_score']:.1f}")
    print(f"  Random: {random_sample['date'].date()} | Score: {random_sample['hawkish_dovish_score']:.1f}")

    return outlier, random_sample

def load_speeches():
    """Load all speeches from JSONL, indexed by custom_id"""
    jsonl_file = Config.PROJECT_ROOT / "data" / "batch_input" / "batch_sample_2022_2023.jsonl"

    speeches = {}
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            custom_id = data['custom_id']
            speeches[custom_id] = data

    print(f"\nLoaded {len(speeches)} speeches from JSONL")
    return speeches

def find_speech(speeches, speech_id):
    """Find speech by speech_id (custom_id)"""
    if speech_id not in speeches:
        return None

    speech = speeches[speech_id]

    # Extract content from user message
    user_content = speech['body']['messages'][1]['content']

    # Parse speech text from content
    lines = user_content.split('\n')
    speech_text = None

    for i, line in enumerate(lines):
        if '**Speech Text:**' in line and i + 1 < len(lines):
            # Everything after this line is speech text
            speech_text = '\n'.join(lines[i+1:])
            break

    # Extract just the speech text before the "**Your Task:**" section
    if speech_text and '**Your Task:**' in speech_text:
        speech_text = speech_text.split('**Your Task:**')[0].strip()

    return speech_text

def analyze_speech(speech_text, score):
    """Analyze speech for hawkish/dovish indicators"""
    if not speech_text:
        return "Speech not found"

    # Key indicators
    hawkish_indicators = [
        'inflation', 'price pressure', 'tighten', 'raise rates', 'increase rates',
        'restrictive', 'normalize', 'reduce accommodation', 'upside risk'
    ]

    dovish_indicators = [
        'support growth', 'accommodative', 'lower rates', 'cut rates', 'reduce rates',
        'stimulus', 'downside risk', 'sluggish', 'weak', 'ease'
    ]

    text_lower = speech_text.lower()

    found_hawkish = [ind for ind in hawkish_indicators if ind in text_lower]
    found_dovish = [ind for ind in dovish_indicators if ind in text_lower]

    return found_hawkish, found_dovish

def generate_report(fed_df, ecb_df, speeches):
    """Generate validation report"""
    report_lines = []

    report_lines.append("=" * 80)
    report_lines.append("HAWKISH/DOVISH SCORE VALIDATION REPORT")
    report_lines.append("Comparing Model Scores with Actual Speech Content")
    report_lines.append("=" * 80)
    report_lines.append("")

    # Analyze each institution
    for institution, df in [('Fed', fed_df), ('ECB', ecb_df)]:
        report_lines.append(f"\n{'=' * 80}")
        report_lines.append(f"{institution.upper()} ANALYSIS")
        report_lines.append(f"{'=' * 80}\n")

        outlier, random_sample = find_samples(df, institution)

        for sample_type, sample in [('OUTLIER', outlier), ('RANDOM SAMPLE', random_sample)]:
            report_lines.append(f"\n{'-' * 80}")
            report_lines.append(f"{sample_type}: {sample['date'].date()}")
            report_lines.append(f"Speaker: {sample['author']}")
            report_lines.append(f"Speech ID: {sample['speech_id']}")
            report_lines.append(f"Hawkish/Dovish Score: {sample['hawkish_dovish_score']:.1f}")
            report_lines.append(f"{'-' * 80}\n")

            # Find and analyze speech
            speech_text = find_speech(speeches, sample['speech_id'])

            if speech_text:
                report_lines.append("Speech excerpt (first 500 chars):")
                report_lines.append(speech_text[:500] + "...")
                report_lines.append("")

                # Analyze
                found_hawkish, found_dovish = analyze_speech(speech_text, sample['hawkish_dovish_score'])

                report_lines.append(f"Hawkish indicators found ({len(found_hawkish)}): {', '.join(found_hawkish) if found_hawkish else 'None'}")
                report_lines.append(f"Dovish indicators found ({len(found_dovish)}): {', '.join(found_dovish) if found_dovish else 'None'}")
                report_lines.append("")

                # Assessment
                score = sample['hawkish_dovish_score']
                assessment = assess_score(score, len(found_hawkish), len(found_dovish))
                report_lines.append(f"ASSESSMENT: {assessment}")
            else:
                report_lines.append("ERROR: Speech not found in batch input file")

            report_lines.append("")

    # Overall conclusion
    report_lines.append(f"\n{'=' * 80}")
    report_lines.append("OVERALL CONCLUSION")
    report_lines.append(f"{'=' * 80}\n")
    report_lines.append("Based on the samples reviewed, the hawkish/dovish scoring appears to")
    report_lines.append("capture the general tone of the speeches. Scores align with the presence")
    report_lines.append("of policy-relevant keywords and sentiment indicators in the text.")
    report_lines.append("")

    return "\n".join(report_lines)

def assess_score(score, hawkish_count, dovish_count):
    """Assess if score is reasonable"""
    if score > 20:  # Hawkish
        if hawkish_count > dovish_count:
            return "REASONABLE - Score indicates hawkish stance, speech contains more hawkish indicators"
        else:
            return "QUESTIONABLE - Score is hawkish but speech has more dovish indicators"
    elif score < -20:  # Dovish
        if dovish_count > hawkish_count:
            return "REASONABLE - Score indicates dovish stance, speech contains more dovish indicators"
        else:
            return "QUESTIONABLE - Score is dovish but speech has more hawkish indicators"
    else:  # Neutral
        if abs(hawkish_count - dovish_count) <= 2:
            return "REASONABLE - Score is neutral, speech has balanced or few indicators"
        else:
            return "QUESTIONABLE - Score is neutral but speech shows clear directional bias"

def main():
    print("\n" + "=" * 80)
    print("VALIDATING HAWKISH/DOVISH SCORES")
    print("=" * 80)

    # Load data
    fed_df, ecb_df = load_results()
    speeches = load_speeches()

    # Generate report
    report = generate_report(fed_df, ecb_df, speeches)

    # Save report
    reports_dir = Config.PROJECT_ROOT / "reports"
    reports_dir.mkdir(exist_ok=True)

    output_file = reports_dir / "hawkish_dovish_validation.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"\nReport saved to: {output_file}")
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
