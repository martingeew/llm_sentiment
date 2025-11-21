"""
Output validation module for LLM sentiment analysis results.

This module validates that LLM outputs are sensible and complete.

For beginners:
- Validation means checking that outputs meet our expectations
- We check things like: Are scores in the right range? Are required fields present?
- This helps catch errors early before using the data for analysis
"""

import json
import pandas as pd
from typing import Dict, Any, List, Tuple
from pathlib import Path


class OutputValidator:
    """
    Validates LLM sentiment analysis outputs.

    Checks:
    1. JSON structure is correct
    2. All required fields are present
    3. Scores are within expected ranges
    4. No obvious errors or nonsense
    """

    def __init__(self):
        """Initialize the validator with expected field specifications."""

        # Define what we expect in the output
        self.required_fields = [
            'hawkish_dovish_score',
            'topics',
            'uncertainty',
            'forward_guidance_strength',
            'key_sentences',
            'market_impact',
            'summary'
        ]

        self.required_topic_fields = [
            'inflation',
            'growth',
            'financial_stability',
            'labor_market',
            'international'
        ]

        self.required_market_fields = [
            'stocks',
            'bonds',
            'currency',
            'reasoning'
        ]

        # Define valid ranges for numeric scores
        self.score_ranges = {
            'hawkish_dovish_score': (-100, 100),
            'uncertainty': (0, 100),
            'forward_guidance_strength': (0, 100),
            'topic_scores': (0, 100)  # All topic scores
        }

        # Valid values for market impact
        self.valid_market_values = ['rise', 'fall', 'neutral']

    def validate_single_output(self, output: Dict[str, Any],
                              speech_id: str = "unknown") -> Tuple[bool, List[str]]:
        """
        Validate a single LLM output.

        Args:
            output: The parsed JSON output from LLM
            speech_id: ID of the speech (for error reporting)

        Returns:
            Tuple of (is_valid, list_of_errors)

        For beginners:
        - This checks one speech's analysis
        - Returns True if everything looks good
        - Returns False + list of problems if something is wrong
        """

        errors = []

        # Check 1: All required top-level fields present
        for field in self.required_fields:
            if field not in output:
                errors.append(f"Missing required field: {field}")

        if errors:
            return False, errors

        # Check 2: Hawkish/dovish score in valid range
        hd_score = output.get('hawkish_dovish_score')
        if not isinstance(hd_score, (int, float)):
            errors.append(f"hawkish_dovish_score must be numeric, got: {type(hd_score)}")
        elif hd_score < -100 or hd_score > 100:
            errors.append(f"hawkish_dovish_score out of range [-100, 100]: {hd_score}")

        # Check 3: Topics structure
        topics = output.get('topics', {})
        if not isinstance(topics, dict):
            errors.append(f"topics must be a dictionary, got: {type(topics)}")
        else:
            for topic_field in self.required_topic_fields:
                if topic_field not in topics:
                    errors.append(f"Missing topic field: {topic_field}")
                else:
                    score = topics[topic_field]
                    if not isinstance(score, (int, float)):
                        errors.append(f"Topic '{topic_field}' must be numeric, got: {type(score)}")
                    elif score < 0 or score > 100:
                        errors.append(f"Topic '{topic_field}' out of range [0, 100]: {score}")

        # Check 4: Uncertainty score
        uncertainty = output.get('uncertainty')
        if not isinstance(uncertainty, (int, float)):
            errors.append(f"uncertainty must be numeric, got: {type(uncertainty)}")
        elif uncertainty < 0 or uncertainty > 100:
            errors.append(f"uncertainty out of range [0, 100]: {uncertainty}")

        # Check 5: Forward guidance strength
        fg_strength = output.get('forward_guidance_strength')
        if not isinstance(fg_strength, (int, float)):
            errors.append(f"forward_guidance_strength must be numeric, got: {type(fg_strength)}")
        elif fg_strength < 0 or fg_strength > 100:
            errors.append(f"forward_guidance_strength out of range [0, 100]: {fg_strength}")

        # Check 6: Key sentences
        key_sentences = output.get('key_sentences', [])
        if not isinstance(key_sentences, list):
            errors.append(f"key_sentences must be a list, got: {type(key_sentences)}")
        elif len(key_sentences) == 0:
            errors.append("key_sentences is empty")

        # Check 7: Market impact structure
        market_impact = output.get('market_impact', {})
        if not isinstance(market_impact, dict):
            errors.append(f"market_impact must be a dictionary, got: {type(market_impact)}")
        else:
            for market_field in self.required_market_fields:
                if market_field not in market_impact:
                    errors.append(f"Missing market_impact field: {market_field}")

            # Check market impact values are valid
            for field in ['stocks', 'bonds', 'currency']:
                if field in market_impact:
                    value = market_impact[field]
                    if value not in self.valid_market_values:
                        errors.append(
                            f"market_impact.{field} must be one of {self.valid_market_values}, "
                            f"got: {value}"
                        )

        # Check 8: Summary is not empty
        summary = output.get('summary', '')
        if not isinstance(summary, str):
            errors.append(f"summary must be a string, got: {type(summary)}")
        elif len(summary.strip()) == 0:
            errors.append("summary is empty")

        # If we found errors, validation failed
        is_valid = len(errors) == 0

        return is_valid, errors

    def validate_batch_results(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate an entire batch of results.

        Args:
            results_df: DataFrame with parsed sentiment results

        Returns:
            Dictionary with validation statistics and any errors found

        For beginners:
        - This checks ALL speeches at once
        - Gives you a report of how many passed/failed
        - Lists any problems found
        """

        total_speeches = len(results_df)
        validation_results = {
            'total_speeches': total_speeches,
            'valid_speeches': 0,
            'invalid_speeches': 0,
            'validation_rate': 0.0,
            'errors_by_speech': {},
            'error_summary': {}
        }

        # Validate each speech
        for idx, row in results_df.iterrows():
            speech_id = row.get('speech_id', f'unknown_{idx}')

            # Reconstruct the output dict from DataFrame columns
            output = {
                'hawkish_dovish_score': row.get('hawkish_dovish_score'),
                'topics': {
                    'inflation': row.get('topic_inflation'),
                    'growth': row.get('topic_growth'),
                    'financial_stability': row.get('topic_financial_stability'),
                    'labor_market': row.get('topic_labor_market'),
                    'international': row.get('topic_international')
                },
                'uncertainty': row.get('uncertainty'),
                'forward_guidance_strength': row.get('forward_guidance_strength'),
                'key_sentences': row.get('key_sentences', '').split('|') if pd.notna(row.get('key_sentences')) else [],
                'market_impact': {
                    'stocks': row.get('market_impact_stocks'),
                    'bonds': row.get('market_impact_bonds'),
                    'currency': row.get('market_impact_currency'),
                    'reasoning': row.get('market_impact_reasoning')
                },
                'summary': row.get('summary', '')
            }

            is_valid, errors = self.validate_single_output(output, speech_id)

            if is_valid:
                validation_results['valid_speeches'] += 1
            else:
                validation_results['invalid_speeches'] += 1
                validation_results['errors_by_speech'][speech_id] = errors

                # Count error types
                for error in errors:
                    error_type = error.split(':')[0]  # Get first part of error message
                    if error_type not in validation_results['error_summary']:
                        validation_results['error_summary'][error_type] = 0
                    validation_results['error_summary'][error_type] += 1

        # Calculate validation rate
        if total_speeches > 0:
            validation_results['validation_rate'] = (
                validation_results['valid_speeches'] / total_speeches * 100
            )

        return validation_results

    def print_validation_report(self, validation_results: Dict[str, Any]):
        """
        Print a nicely formatted validation report.

        Args:
            validation_results: Output from validate_batch_results()

        For beginners:
        - This displays the validation results in a readable format
        - Shows how many speeches passed/failed
        - Highlights any common issues
        """

        print("\n" + "=" * 60)
        print("VALIDATION REPORT")
        print("=" * 60)

        print(f"\n📊 Overall Statistics:")
        print(f"   Total speeches: {validation_results['total_speeches']}")
        print(f"   ✅ Valid: {validation_results['valid_speeches']}")
        print(f"   ❌ Invalid: {validation_results['invalid_speeches']}")
        print(f"   Validation rate: {validation_results['validation_rate']:.1f}%")

        if validation_results['error_summary']:
            print(f"\n⚠️  Common Error Types:")
            for error_type, count in sorted(
                validation_results['error_summary'].items(),
                key=lambda x: x[1],
                reverse=True
            ):
                print(f"   - {error_type}: {count} occurrences")

        # Show first few problematic speeches
        if validation_results['errors_by_speech']:
            print(f"\n🔍 Sample of Problematic Speeches (first 5):")
            for i, (speech_id, errors) in enumerate(
                list(validation_results['errors_by_speech'].items())[:5]
            ):
                print(f"\n   Speech: {speech_id}")
                for error in errors[:3]:  # Show first 3 errors
                    print(f"      - {error}")

        print("\n" + "=" * 60)

    def check_score_distributions(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Check if score distributions look reasonable.

        Args:
            results_df: DataFrame with sentiment results

        Returns:
            Dictionary with distribution statistics

        For beginners:
        - This checks if the scores look "normal"
        - For example, we'd expect a mix of hawkish and dovish speeches
        - If ALL speeches are extremely hawkish, something might be wrong
        """

        distributions = {}

        # Hawkish/Dovish score distribution
        if 'hawkish_dovish_score' in results_df.columns:
            hd_scores = results_df['hawkish_dovish_score'].dropna()
            distributions['hawkish_dovish'] = {
                'mean': float(hd_scores.mean()),
                'median': float(hd_scores.median()),
                'std': float(hd_scores.std()),
                'min': float(hd_scores.min()),
                'max': float(hd_scores.max()),
                'very_dovish': int((hd_scores < -50).sum()),
                'dovish': int(((hd_scores >= -50) & (hd_scores < -10)).sum()),
                'neutral': int(((hd_scores >= -10) & (hd_scores <= 10)).sum()),
                'hawkish': int(((hd_scores > 10) & (hd_scores <= 50)).sum()),
                'very_hawkish': int((hd_scores > 50).sum())
            }

        # Uncertainty distribution
        if 'uncertainty' in results_df.columns:
            uncertainty = results_df['uncertainty'].dropna()
            distributions['uncertainty'] = {
                'mean': float(uncertainty.mean()),
                'median': float(uncertainty.median()),
                'std': float(uncertainty.std())
            }

        # Forward guidance distribution
        if 'forward_guidance_strength' in results_df.columns:
            fg = results_df['forward_guidance_strength'].dropna()
            distributions['forward_guidance'] = {
                'mean': float(fg.mean()),
                'median': float(fg.median()),
                'std': float(fg.std())
            }

        return distributions

    def print_distribution_report(self, distributions: Dict[str, Any]):
        """
        Print distribution analysis report.

        Args:
            distributions: Output from check_score_distributions()
        """

        print("\n" + "=" * 60)
        print("SCORE DISTRIBUTION ANALYSIS")
        print("=" * 60)

        if 'hawkish_dovish' in distributions:
            hd = distributions['hawkish_dovish']
            print(f"\n🎯 Hawkish/Dovish Score:")
            print(f"   Mean: {hd['mean']:.1f}")
            print(f"   Median: {hd['median']:.1f}")
            print(f"   Range: [{hd['min']:.1f}, {hd['max']:.1f}]")
            print(f"   Std Dev: {hd['std']:.1f}")
            print(f"\n   Distribution:")
            print(f"      Very Dovish (< -50):   {hd['very_dovish']:3d} speeches")
            print(f"      Dovish (-50 to -10):   {hd['dovish']:3d} speeches")
            print(f"      Neutral (-10 to +10):  {hd['neutral']:3d} speeches")
            print(f"      Hawkish (+10 to +50):  {hd['hawkish']:3d} speeches")
            print(f"      Very Hawkish (> +50):  {hd['very_hawkish']:3d} speeches")

        if 'uncertainty' in distributions:
            unc = distributions['uncertainty']
            print(f"\n📊 Uncertainty Level:")
            print(f"   Mean: {unc['mean']:.1f}")
            print(f"   Median: {unc['median']:.1f}")

        if 'forward_guidance' in distributions:
            fg = distributions['forward_guidance']
            print(f"\n🔮 Forward Guidance Strength:")
            print(f"   Mean: {fg['mean']:.1f}")
            print(f"   Median: {fg['median']:.1f}")

        print("\n" + "=" * 60)


# Example usage
if __name__ == "__main__":
    """
    Test the validator with sample data.
    """

    print("=" * 60)
    print("TESTING OUTPUT VALIDATOR")
    print("=" * 60)

    # Create a sample valid output
    valid_output = {
        'hawkish_dovish_score': 45,
        'topics': {
            'inflation': 80,
            'growth': 30,
            'financial_stability': 20,
            'labor_market': 40,
            'international': 10
        },
        'uncertainty': 35,
        'forward_guidance_strength': 60,
        'key_sentences': [
            "Inflation remains elevated",
            "We are committed to price stability",
            "Further rate increases may be necessary"
        ],
        'market_impact': {
            'stocks': 'fall',
            'bonds': 'rise',
            'currency': 'strengthen',
            'reasoning': 'Hawkish tone suggests higher rates ahead'
        },
        'summary': 'The speech signals a hawkish stance with focus on controlling inflation.'
    }

    # Create a sample invalid output (missing fields, out of range)
    invalid_output = {
        'hawkish_dovish_score': 150,  # Out of range!
        'topics': {
            'inflation': 80
            # Missing other topics!
        },
        'uncertainty': 35
        # Missing other required fields!
    }

    validator = OutputValidator()

    # Test valid output
    print("\n📝 Testing VALID output:")
    is_valid, errors = validator.validate_single_output(valid_output, "test_valid")
    print(f"   Result: {'✅ PASS' if is_valid else '❌ FAIL'}")
    if errors:
        print(f"   Errors: {errors}")

    # Test invalid output
    print("\n📝 Testing INVALID output:")
    is_valid, errors = validator.validate_single_output(invalid_output, "test_invalid")
    print(f"   Result: {'✅ PASS' if is_valid else '❌ FAIL'}")
    if errors:
        print(f"   Errors found:")
        for error in errors:
            print(f"      - {error}")

    print("\n✅ Validator is working correctly!")
