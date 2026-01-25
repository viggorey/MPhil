"""
Factor Resolution Module for N×M Factorial Designs.

This module provides centralized factor resolution for datasets, replacing
hardcoded condition parsing with configuration-driven lookups.

The module requires dataset_links.json to contain explicit factor assignments
for each dataset. Legacy naming conventions (e.g., '11U', '21U') are NOT
supported - all datasets must be registered in dataset_links.json.

Usage:
    from factor_resolver import FactorResolver

    resolver = FactorResolver(project)
    factors = resolver.resolve_dataset("11U1")
    # Returns: {'species': 'WR', 'surface': 'Waxy', ...}
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
import pandas as pd


class FactorResolutionError(Exception):
    """Raised when a dataset cannot be resolved to factors."""
    pass


class FactorResolver:
    """
    Resolves dataset names to experimental factors using project configuration.

    This class provides the central mechanism for mapping dataset names to
    their factor values (species, surface, etc.) using the dataset_links.json
    file and project configuration.

    Attributes:
        project: The active Project instance.
        dataset_links: Dictionary of dataset name -> link info.
        species_registry: Dictionary of species abbreviation -> species info.
        surface_registry: Dictionary of surface abbreviation -> surface info.
        factors_config: Dictionary defining the experimental factors.
    """

    def __init__(self, project):
        """
        Initialize the factor resolver with a project.

        Args:
            project: Project instance with configuration and paths.

        Raises:
            ValueError: If project is None.
        """
        if project is None:
            raise ValueError("Project is required for factor resolution")

        self.project = project
        self._load_data()

    def _load_data(self):
        """Load dataset links and registries from project."""
        # Load dataset links
        self.dataset_links = {}
        if self.project.dataset_links_file.exists():
            with open(self.project.dataset_links_file, 'r', encoding='utf-8') as f:
                self.dataset_links = json.load(f)

        # Load registries from project config
        self.species_registry = self.project.get_species_registry()
        self.surface_registry = self.project.get_surface_registry()
        self.factors_config = self.project.get_factors()

    def reload(self):
        """Reload data from disk (useful after edits to dataset_links.json)."""
        self._load_data()

    def resolve_dataset(self, dataset_name: str) -> Dict[str, str]:
        """
        Resolve a dataset name to its factor values.

        Args:
            dataset_name: Name of the dataset (e.g., '11U1', 'WR_Waxy_001').

        Returns:
            Dictionary mapping factor names to their values for this dataset.
            Example: {'species': 'WR', 'surface': 'Waxy'}

        Raises:
            FactorResolutionError: If the dataset is not found in dataset_links.json
                                   or is missing required factor assignments.
        """
        # Clean up dataset name (remove _param suffix if present)
        clean_name = dataset_name.replace('_param', '').replace('_3D', '')

        if clean_name not in self.dataset_links:
            raise FactorResolutionError(
                f"Dataset '{clean_name}' not found in dataset_links.json. "
                f"All datasets must be registered with explicit factor assignments."
            )

        link_info = self.dataset_links[clean_name]
        factors = {}

        # Extract species factor
        if 'species' in link_info:
            factors['species'] = link_info['species']

        # Extract surface factor
        if 'surface' in link_info:
            factors['surface'] = link_info['surface']

        # Extract any additional custom factors
        for key, value in link_info.items():
            if key not in ['calibration_set', 'branch_set', 'species', 'surface']:
                factors[key] = value

        return factors

    def get_dataset_species(self, dataset_name: str) -> Optional[str]:
        """
        Get the species abbreviation for a dataset.

        Args:
            dataset_name: Name of the dataset.

        Returns:
            Species abbreviation (e.g., 'WR') or None if not found.
        """
        try:
            factors = self.resolve_dataset(dataset_name)
            return factors.get('species')
        except FactorResolutionError:
            return None

    def get_dataset_surface(self, dataset_name: str) -> Optional[str]:
        """
        Get the surface identifier for a dataset.

        Args:
            dataset_name: Name of the dataset.

        Returns:
            Surface identifier (e.g., 'Waxy', 'Smooth') or None if not found.
        """
        try:
            factors = self.resolve_dataset(dataset_name)
            return factors.get('surface')
        except FactorResolutionError:
            return None

    def get_factor_display_label(self, factor_name: str, factor_value: str) -> str:
        """
        Get the human-readable display label for a factor value.

        Args:
            factor_name: Name of the factor (e.g., 'species', 'surface').
            factor_value: Value of the factor (e.g., 'WR', 'Waxy').

        Returns:
            Display label (e.g., 'C. borneensis' for species='WR').
        """
        if factor_name == 'species' and factor_value in self.species_registry:
            species_info = self.species_registry[factor_value]
            return species_info.get('name', species_info.get('full_name', factor_value))

        if factor_name == 'surface' and factor_value in self.surface_registry:
            surface_info = self.surface_registry[factor_value]
            return surface_info.get('name', factor_value)

        # Check factors_config for custom labels
        if factor_name in self.factors_config:
            factor_config = self.factors_config[factor_name]
            labels = factor_config.get('labels', {})
            if factor_value in labels:
                return labels[factor_value]

        return factor_value

    def get_all_factor_labels(self, factors: Dict[str, str]) -> Dict[str, str]:
        """
        Get display labels for all factors in a dictionary.

        Args:
            factors: Dictionary of factor name -> factor value.

        Returns:
            Dictionary of factor name -> display label.
        """
        return {
            name: self.get_factor_display_label(name, value)
            for name, value in factors.items()
        }

    def get_condition_string(self, factors: Dict[str, str], use_labels: bool = True) -> str:
        """
        Create a condition string from factors.

        Args:
            factors: Dictionary of factor name -> factor value.
            use_labels: If True, use display labels instead of raw values.

        Returns:
            Condition string (e.g., 'C. borneensis_Waxy' or 'WR_Waxy').
        """
        if use_labels:
            values = [self.get_factor_display_label(k, v) for k, v in factors.items()]
        else:
            values = list(factors.values())
        return '_'.join(values)

    def get_all_species(self) -> List[str]:
        """
        Get all species abbreviations from the species registry.

        Returns:
            List of species abbreviations.
        """
        return list(self.species_registry.keys())

    def get_all_surfaces(self) -> List[str]:
        """
        Get all surface identifiers from the surface registry.

        Returns:
            List of surface identifiers.
        """
        return list(self.surface_registry.keys())

    def get_factor_levels(self, factor_name: str) -> List[str]:
        """
        Get all levels (values) for a factor.

        Args:
            factor_name: Name of the factor.

        Returns:
            List of factor levels.
        """
        if factor_name == 'species':
            return self.get_all_species()
        elif factor_name == 'surface':
            return self.get_all_surfaces()
        elif factor_name in self.factors_config:
            return self.factors_config[factor_name].get('levels', [])
        return []

    def get_datasets_for_condition(self, **factors) -> List[str]:
        """
        Get all dataset names matching specific factor values.

        Args:
            **factors: Factor name=value pairs to match.

        Returns:
            List of matching dataset names.
        """
        matches = []
        for dataset_name, link_info in self.dataset_links.items():
            match = True
            for factor_name, factor_value in factors.items():
                if link_info.get(factor_name) != factor_value:
                    match = False
                    break
            if match:
                matches.append(dataset_name)
        return sorted(matches)

    def validate_dataset_links(self) -> List[str]:
        """
        Validate that all datasets have required factor assignments.

        Required factors are determined from the project's factors_config,
        NOT hardcoded. This allows single-factor projects (e.g., only species
        without surface) to be valid.

        Returns:
            List of error messages (empty if all valid).
        """
        errors = []

        # Get required factors from project config, not hardcoded
        required_factors = list(self.factors_config.keys()) if self.factors_config else []

        for dataset_name, link_info in self.dataset_links.items():
            # Only validate factors that are defined in project config
            for factor in required_factors:
                if factor not in link_info or not link_info[factor]:
                    errors.append(
                        f"Dataset '{dataset_name}' is missing required factor '{factor}'"
                    )

            # Validate species exists in registry (if species is provided)
            if 'species' in link_info and link_info['species']:
                species = link_info['species']
                if species not in self.species_registry:
                    errors.append(
                        f"Dataset '{dataset_name}' references unknown species '{species}'"
                    )

            # Validate surface exists in registry (if surface is provided)
            if 'surface' in link_info and link_info['surface']:
                surface = link_info['surface']
                if surface not in self.surface_registry:
                    errors.append(
                        f"Dataset '{dataset_name}' references unknown surface '{surface}'"
                    )

        return errors

    def build_analysis_dataframe(
        self,
        datasets: List[str],
        values: Dict[str, float]
    ) -> pd.DataFrame:
        """
        Build a DataFrame with factor columns for statistical analysis.

        Args:
            datasets: List of dataset names.
            values: Dictionary mapping dataset name to metric value.

        Returns:
            DataFrame with columns for each factor plus 'value'.
        """
        rows = []
        for dataset_name in datasets:
            if dataset_name not in values:
                continue

            try:
                factors = self.resolve_dataset(dataset_name)
                row = {
                    'dataset': dataset_name,
                    'value': values[dataset_name],
                    **factors
                }
                rows.append(row)
            except FactorResolutionError:
                continue

        return pd.DataFrame(rows)

    def get_unique_conditions(self) -> List[Dict[str, str]]:
        """
        Get all unique factor combinations present in the dataset links.

        Returns:
            List of dictionaries, each representing a unique condition.
        """
        conditions_set = set()
        conditions_list = []

        for dataset_name in self.dataset_links:
            try:
                factors = self.resolve_dataset(dataset_name)
                # Create hashable key for deduplication
                key = tuple(sorted(factors.items()))
                if key not in conditions_set:
                    conditions_set.add(key)
                    conditions_list.append(factors)
            except FactorResolutionError:
                continue

        return conditions_list


def get_factor_resolver(project) -> FactorResolver:
    """
    Factory function to create a FactorResolver for a project.

    Args:
        project: Project instance.

    Returns:
        Configured FactorResolver instance.
    """
    return FactorResolver(project)


# =============================================================================
# Utility Functions for Direct Use
# =============================================================================

def resolve_factors_from_links(
    dataset_name: str,
    dataset_links: Dict[str, Any]
) -> Optional[Dict[str, str]]:
    """
    Resolve factors directly from a dataset_links dictionary.

    This is a convenience function for cases where a full FactorResolver
    is not needed.

    Args:
        dataset_name: Name of the dataset.
        dataset_links: Dictionary of dataset links.

    Returns:
        Dictionary of factors or None if not found.
    """
    clean_name = dataset_name.replace('_param', '').replace('_3D', '')

    if clean_name not in dataset_links:
        return None

    link_info = dataset_links[clean_name]
    factors = {}

    if 'species' in link_info:
        factors['species'] = link_info['species']
    if 'surface' in link_info:
        factors['surface'] = link_info['surface']

    # Include any additional factors
    for key, value in link_info.items():
        if key not in ['calibration_set', 'branch_set', 'species', 'surface']:
            factors[key] = value

    return factors if factors else None


def get_species_color_mapping(
    species_registry: Dict[str, Any],
    color_palette: Optional[List[str]] = None
) -> Dict[str, str]:
    """
    Generate a color mapping for species.

    Args:
        species_registry: Species registry dictionary.
        color_palette: Optional list of colors. If not provided, uses default.

    Returns:
        Dictionary mapping species abbreviation to color.
    """
    if color_palette is None:
        color_palette = [
            '#1f77b4',  # Blue
            '#2ca02c',  # Green
            '#ff7f0e',  # Orange
            '#d62728',  # Red
            '#9467bd',  # Purple
            '#8c564b',  # Brown
            '#e377c2',  # Pink
            '#7f7f7f',  # Gray
            '#bcbd22',  # Olive
            '#17becf',  # Cyan
        ]

    color_mapping = {}
    for i, species in enumerate(species_registry.keys()):
        color_mapping[species] = color_palette[i % len(color_palette)]

    return color_mapping


def get_surface_hatch_mapping(
    surface_registry: Dict[str, Any],
    hatch_patterns: Optional[List[str]] = None
) -> Dict[str, str]:
    """
    Generate a hatch pattern mapping for surfaces.

    Args:
        surface_registry: Surface registry dictionary.
        hatch_patterns: Optional list of hatch patterns. If not provided, uses default.

    Returns:
        Dictionary mapping surface identifier to hatch pattern.
    """
    if hatch_patterns is None:
        hatch_patterns = [
            '',       # Solid (no hatch)
            '///',    # Forward diagonal
            '\\\\\\', # Backward diagonal
            '+++',    # Plus
            'xxx',    # X pattern
            'ooo',    # Circles
            '...',    # Dots
        ]

    hatch_mapping = {}
    for i, surface in enumerate(surface_registry.keys()):
        hatch_mapping[surface] = hatch_patterns[i % len(hatch_patterns)]

    return hatch_mapping


def validate_factors_config(factors_config: Dict[str, Any]) -> List[str]:
    """
    Validate a factors configuration dictionary.

    Args:
        factors_config: Factors configuration to validate.

    Returns:
        List of error messages (empty if valid).
    """
    errors = []

    for factor_name, factor_config in factors_config.items():
        if not isinstance(factor_config, dict):
            errors.append(f"Factor '{factor_name}' must be a dictionary")
            continue

        if 'levels' not in factor_config:
            errors.append(f"Factor '{factor_name}' is missing 'levels' list")
        elif not isinstance(factor_config['levels'], list):
            errors.append(f"Factor '{factor_name}' levels must be a list")
        elif len(factor_config['levels']) < 1:
            errors.append(f"Factor '{factor_name}' must have at least 1 level")

    return errors
