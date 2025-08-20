"""
Utility modules for the configurable pipeline merger.

This package groups together functionality previously embedded directly
within the ``ConfigurablePipelineMerger`` class.  By separating these
concerns into dedicated modules we adhere to the Single Responsibility
Principle and make the codebase easier to maintain and extend.  Each
module exposes functions that operate on pandas DataFrames and, where
necessary, accept the pipeline configuration as an argument.  The
pipeline orchestrator can compose these functions without needing to
know their internal details.
"""

# Make submodules discoverable when imported from this package
from .feature_engineering import (
    calculate_equilibrium_temperature,
    calculate_escape_velocity,
    calculate_hz_positions,
    calculate_stellar_flux,
    calculate_surface_gravity,
    compute_distances,
    derive_features_with_uncertainty,
    monte_carlo_propagation,
)
from .missingness import (
    add_provenance_information,
    assign_exolife_ids,
    encode_missingness_patterns,
)
from .normalization import (
    get_pipeline_statistics,
    normalize_output_tables,
    save_normalized_results,
    save_results,
)
from .quality import (
    add_quality_indicators,
    assign_astrometry_quality_flag,
    assign_crossmatch_quality,
)
from .units import apply_physical_range_checks, standardize_units

__all__ = [
    # Feature engineering
    "calculate_equilibrium_temperature",
    "calculate_escape_velocity",
    "calculate_hz_positions",
    "calculate_stellar_flux",
    "calculate_surface_gravity",
    "compute_distances",
    "derive_features_with_uncertainty",
    "monte_carlo_propagation",
    # Missingness
    "add_provenance_information",
    "assign_exolife_ids",
    "encode_missingness_patterns",
    # Normalization
    "get_pipeline_statistics",
    "normalize_output_tables",
    "save_normalized_results",
    "save_results",
    # Quality
    "add_quality_indicators",
    "assign_astrometry_quality_flag",
    "assign_crossmatch_quality",
    # Units
    "apply_physical_range_checks",
    "standardize_units",
]
