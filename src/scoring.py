"""Scoring utilities for benchmarks."""

import logging
from typing import List, Dict, Any, Optional
import numpy as np

logger = logging.getLogger(__name__)


class PlaceholderScorer:
    """
    Placeholder scorer that returns deterministic proxy scores.
    
    This is clearly marked as a placeholder and raises NotImplementedError
    behind a flag when official scoring is unavailable.
    """
    
    def __init__(self, benchmark_name: str, use_placeholder: bool = True):
        self.benchmark_name = benchmark_name
        self.use_placeholder = use_placeholder
        if not use_placeholder:
            raise NotImplementedError(
                f"Official scoring for {benchmark_name} is not yet implemented. "
                "Set use_placeholder=True to use deterministic proxy scores."
            )
        logger.warning(
            f"Using PLACEHOLDER scorer for {benchmark_name}. "
            "This returns deterministic proxy scores, not official benchmark scores."
        )
    
    def score(self, prompts: List[str], model_outputs: List[str]) -> np.ndarray:
        """
        Compute scores for model outputs.
        
        Placeholder implementation returns deterministic proxy scores
        based on output hash for reproducibility.
        
        Args:
            prompts: List of input prompts
            model_outputs: List of model-generated outputs
        
        Returns:
            Array of scores in [0, 1] range
        """
        if not self.use_placeholder:
            raise NotImplementedError("Official scoring not implemented")
        
        scores = []
        for prompt, output in zip(prompts, model_outputs):
            # Deterministic proxy: hash-based score for reproducibility
            # This is a placeholder and should be replaced with actual scoring
            # TODO: Replace with actual benchmark scoring
            combined = f"{prompt}|||{output}"
            hash_val = hash(combined)
            # Convert hash to [0, 1] range deterministically
            score = abs(hash_val % 10000) / 10000.0
            scores.append(score)
        
        return np.array(scores)


def load_scorer(config: Dict[str, Any], benchmark_type: str) -> PlaceholderScorer:
    """
    Load scorer for a benchmark.
    
    Args:
        config: Experiment configuration
        benchmark_type: "on_target" or "off_target"
    
    Returns:
        Scorer object
    """
    # For now, always use placeholder
    # In practice, this would load official scorers for SALADBench, GPQA, etc.
    benchmark_name = f"{benchmark_type}_benchmark"
    
    # Check if config specifies to use official scoring
    use_placeholder = config.get("scoring", {}).get("use_placeholder", True)
    
    return PlaceholderScorer(benchmark_name, use_placeholder=use_placeholder)

