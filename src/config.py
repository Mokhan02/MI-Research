"""Configuration loading and resolution utilities."""

import os
import yaml
from pathlib import Path
from typing import Any, Dict
import logging

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    logger.info(f"Loaded config from {config_path}")
    return config


def resolve_config(config: Dict[str, Any], run_id: str) -> Dict[str, Any]:
    """
    Resolve configuration with run-specific paths and defaults.
    
    Ensures all paths are absolute and creates necessary directories.
    """
    resolved = config.copy()
    
    # Set up output directory
    output_dir = os.path.abspath(f"outputs/{run_id}")
    os.makedirs(output_dir, exist_ok=True)
    resolved["experiment"]["run_id"] = run_id
    resolved["experiment"]["output_dir"] = output_dir
    
    # Resolve data paths
    if "data" not in resolved:
        resolved["data"] = {}
    data_dir = os.path.abspath("data")
    resolved["data"]["data_dir"] = data_dir
    
    # Ensure steering parameters are present
    if "steering" not in resolved:
        raise ValueError("Config must contain 'steering' section")
    
    steering = resolved["steering"]
    required_params = ["alpha_grid", "threshold_T", "alpha0", "tau"]
    for param in required_params:
        if param not in steering:
            raise ValueError(f"Config must contain steering.{param}")
    
    # Validate alpha_grid
    alpha_grid = steering["alpha_grid"]
    if not isinstance(alpha_grid, list) or len(alpha_grid) == 0:
        raise ValueError("steering.alpha_grid must be a non-empty list")
    if 0.0 not in alpha_grid:
        logger.warning("alpha_grid should include 0.0 for baseline measurements")
    
    # Ensure deterministic generation settings
    if "model" in resolved:
        model_cfg = resolved["model"]
        if "temperature" not in model_cfg:
            model_cfg["temperature"] = 0.0
        if "do_sample" not in model_cfg:
            model_cfg["do_sample"] = False
        # Override if explicitly set in config
        if model_cfg.get("temperature", 0.0) != 0.0 or model_cfg.get("do_sample", False):
            logger.warning("Non-deterministic generation settings detected. Ensure this is intentional.")
    
    logger.info(f"Resolved config for run_id={run_id}")
    return resolved


def save_resolved_config(config: Dict[str, Any], run_id: str) -> None:
    """Save resolved configuration to outputs/<run_id>/config_resolved.yaml."""
    output_dir = config["experiment"]["output_dir"]
    config_path = os.path.join(output_dir, "config_resolved.yaml")
    
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    logger.info(f"Saved resolved config to {config_path}")

