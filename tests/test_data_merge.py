"""
Tests for the ExoLife data merge and preprocess modules.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, Mock

from exolife.data.merge.base_merger import BaseMerger
from exolife.data.preprocess.base_preprocessor import BasePreprocessor


class MockMerger(BaseMerger):
    """Mock merger for testing."""

    def merge(self, sources_data):
        """Simple mock merge that concatenates DataFrames."""
        if not sources_data:
            return pd.DataFrame()

        dfs = []
        for source_id, df in sources_data.items():
            df_copy = df.copy()
            df_copy['source'] = source_id
            dfs.append(df_copy)

        result = pd.concat(dfs, ignore_index=True)
        return result


class MockPreprocessor(BasePreprocessor):
    """Mock preprocessor for testing."""

    def preprocess(self, df):
        """Simple mock preprocessing that adds a processed flag."""
        df_copy = df.copy()
        df_copy['processed'] = True
        return df_copy


class TestBaseMerger:
    """Test cases for BaseMerger abstract base class."""

    def test_base_merger_interface(self):
        """Test that BaseMerger defines the required interface."""
        merger = MockMerger()

        # Should have merge method
        assert hasattr(merger, 'merge')
        assert callable(merger.merge)

    def test_mock_merger_functionality(self, sample_dataframe):
        """Test mock merger functionality."""
        merger = MockMerger()

        # Test with single source
        sources_data = {'nasa': sample_dataframe}
        result = merger.merge(sources_data)

        assert len(result) == len(sample_dataframe)
        assert 'source' in result.columns
        assert all(result['source'] == 'nasa')

    def test_mock_merger_multiple_sources(self, sample_dataframe):
        """Test mock merger with multiple data sources."""
        merger = MockMerger()

        # Create second DataFrame
        df2 = sample_dataframe.copy()
        df2['pl_name'] = df2['pl_name'] + ' Copy'

        sources_data = {
            'nasa': sample_dataframe,
            'phl': df2
        }

        result = merger.merge(sources_data)

        # Should have data from both sources
        assert len(result) == len(sample_dataframe) * 2
        assert 'source' in result.columns

        nasa_data = result[result['source'] == 'nasa']
        phl_data = result[result['source'] == 'phl']

        assert len(nasa_data) == len(sample_dataframe)
        assert len(phl_data) == len(sample_dataframe)

    def test_mock_merger_empty_input(self):
        """Test mock merger with empty input."""
        merger = MockMerger()

        result = merger.merge({})

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0


class TestBasePreprocessor:
    """Test cases for BasePreprocessor abstract base class."""

    def test_base_preprocessor_interface(self):
        """Test that BasePreprocessor defines the required interface."""
        preprocessor = MockPreprocessor()

        # Should have preprocess method
        assert hasattr(preprocessor, 'preprocess')
        assert callable(preprocessor.preprocess)

    def test_mock_preprocessor_functionality(self, sample_dataframe):
        """Test mock preprocessor functionality."""
        preprocessor = MockPreprocessor()

        result = preprocessor.preprocess(sample_dataframe)

        # Should have original data plus processed flag
        assert len(result) == len(sample_dataframe)
        assert 'processed' in result.columns
        assert all(result['processed'] == True)

        # Original columns should still be present
        for col in sample_dataframe.columns:
            assert col in result.columns

    def test_mock_preprocessor_preserves_data(self, sample_dataframe):
        """Test that mock preprocessor preserves original data."""
        preprocessor = MockPreprocessor()

        original_values = sample_dataframe['pl_name'].tolist()
        result = preprocessor.preprocess(sample_dataframe)

        # Check that original data is preserved
        assert result['pl_name'].tolist() == original_values

    def test_mock_preprocessor_empty_dataframe(self):
        """Test mock preprocessor with empty DataFrame."""
        preprocessor = MockPreprocessor()

        empty_df = pd.DataFrame()
        result = preprocessor.preprocess(empty_df)

        assert isinstance(result, pd.DataFrame)
        assert 'processed' in result.columns
        assert len(result) == 0


class TestDataQualityChecks:
    """Test cases for data quality checking functionality."""

    def test_dataframe_completeness_check(self, sample_dataframe):
        """Test data completeness checking."""
        df = sample_dataframe.copy()

        # Check completeness of clean data
        completeness = df.notna().sum() / len(df)
        assert all(completeness >= 0.9)  # At least 90% complete

        # Add some missing values
        df.loc[0, 'st_teff'] = np.nan
        df.loc[1, 'pl_rade'] = np.nan

        # Recalculate completeness
        completeness = df.notna().sum() / len(df)

        # Some columns should now have lower completeness
        assert completeness['st_teff'] < 1.0
        assert completeness['pl_rade'] < 1.0

    def test_data_type_validation(self, sample_dataframe):
        """Test data type validation."""
        df = sample_dataframe.copy()

        # Check expected data types
        assert df['pl_name'].dtype == object  # String-like
        assert np.issubdtype(df['st_teff'].dtype, np.number)  # Numeric
        assert np.issubdtype(df['pl_rade'].dtype, np.number)  # Numeric
        assert np.issubdtype(df['gaia_id'].dtype, np.integer)  # Integer

    def test_data_range_validation(self, sample_dataframe):
        """Test data range validation."""
        df = sample_dataframe.copy()

        # Stellar temperature should be reasonable
        assert all(df['st_teff'] > 0)
        assert all(df['st_teff'] < 50000)  # Very hot stars

        # Planet radius should be positive
        assert all(df['pl_rade'] > 0)

        # Gaia IDs should be large integers
        assert all(df['gaia_id'] > 1e15)

    def test_duplicate_detection(self, sample_dataframe):
        """Test duplicate detection."""
        df = sample_dataframe.copy()

        # Original data should have no duplicates
        assert not df.duplicated().any()

        # Add a duplicate row
        duplicate_row = df.iloc[0:1].copy()
        df_with_dup = pd.concat([df, duplicate_row], ignore_index=True)

        # Should now detect duplicates
        assert df_with_dup.duplicated().any()
        assert df_with_dup.duplicated().sum() == 1

    def test_cross_validation_checks(self, sample_dataframe):
        """Test cross-validation between related fields."""
        df = sample_dataframe.copy()

        # Add stellar luminosity in linear units for HZ calculation
        df['st_lum_linear'] = 10 ** df['st_lum']

        # Add HZ edges for validation
        from exolife.data.utils import add_hz_edges_to_df
        df_with_hz = add_hz_edges_to_df(df, lum_col='st_lum_linear')

        # HZ inner should be less than outer
        assert all(df_with_hz['hz_inner'] <= df_with_hz['hz_outer'])

        # Both should be positive
        assert all(df_with_hz['hz_inner'] > 0)
        assert all(df_with_hz['hz_outer'] > 0)


class TestDataMergeWorkflow:
    """Test cases for data merge workflow."""

    def test_simple_merge_workflow(self, sample_dataframe):
        """Test simple merge workflow."""
        # Create two related datasets
        df1 = sample_dataframe[['pl_name', 'st_teff', 'st_lum']].copy()
        df2 = sample_dataframe[['pl_name', 'pl_rade', 'pl_masse']].copy()

        # Merge on planet name
        merged = pd.merge(df1, df2, on='pl_name', how='inner')

        # Should have all columns
        expected_cols = ['pl_name', 'st_teff', 'st_lum', 'pl_rade', 'pl_masse']
        assert all(col in merged.columns for col in expected_cols)

        # Should have same number of rows (perfect match)
        assert len(merged) == len(sample_dataframe)

    def test_merge_with_missing_data(self, sample_dataframe):
        """Test merge workflow with missing data."""
        # Create datasets with partial overlap
        df1 = sample_dataframe[['pl_name', 'st_teff']].copy()
        df2 = sample_dataframe[['pl_name', 'pl_rade']
                               ].iloc[1:].copy()  # Missing first row

        # Inner join should have fewer rows
        inner_merged = pd.merge(df1, df2, on='pl_name', how='inner')
        assert len(inner_merged) == len(sample_dataframe) - 1

        # Left join should preserve all df1 rows
        left_merged = pd.merge(df1, df2, on='pl_name', how='left')
        assert len(left_merged) == len(sample_dataframe)
        assert left_merged['pl_rade'].isna().sum() == 1

    def test_merge_with_fuzzy_matching(self, sample_dataframe):
        """Test merge with fuzzy string matching."""
        # Create datasets with slightly different planet names
        df1 = sample_dataframe[['pl_name', 'st_teff']].copy()
        df2 = sample_dataframe[['pl_name', 'pl_rade']].copy()

        # Modify one name slightly
        df2.loc[0, 'pl_name'] = df2.loc[0, 'pl_name'].replace(
            ' b', 'b')  # Remove space

        # Direct merge should fail to match this row
        direct_merged = pd.merge(df1, df2, on='pl_name', how='inner')
        assert len(direct_merged) == len(sample_dataframe) - 1

        # Simple fuzzy matching approach
        from exolife.data.utils import norm_name
        df1_norm = df1.copy()
        df2_norm = df2.copy()
        df1_norm['pl_name_norm'] = norm_name(df1_norm['pl_name'])
        df2_norm['pl_name_norm'] = norm_name(df2_norm['pl_name'])

        fuzzy_merged = pd.merge(
            df1_norm, df2_norm, on='pl_name_norm', how='inner')
        assert len(fuzzy_merged) == len(sample_dataframe)  # Should match all


class TestPreprocessingWorkflow:
    """Test cases for data preprocessing workflow."""

    def test_column_filtering_workflow(self, sample_dataframe):
        """Test column filtering preprocessing."""
        # Define columns to keep
        keep_columns = ['pl_name', 'st_teff', 'pl_rade']

        # Filter columns
        filtered_df = sample_dataframe[keep_columns].copy()

        assert list(filtered_df.columns) == keep_columns
        assert len(filtered_df) == len(sample_dataframe)

    def test_data_cleaning_workflow(self, sample_dataframe):
        """Test data cleaning preprocessing."""
        df = sample_dataframe.copy()

        # Add some problematic data
        df.loc[0, 'st_teff'] = -1000  # Invalid temperature
        df.loc[1, 'pl_rade'] = 0      # Invalid radius

        # Clean data
        cleaned_df = df.copy()

        # Remove invalid temperatures
        cleaned_df = cleaned_df[cleaned_df['st_teff'] > 0]

        # Remove invalid radii
        cleaned_df = cleaned_df[cleaned_df['pl_rade'] > 0]

        # Should have fewer rows after cleaning
        assert len(cleaned_df) < len(df)
        assert all(cleaned_df['st_teff'] > 0)
        assert all(cleaned_df['pl_rade'] > 0)

    def test_feature_engineering_workflow(self, sample_dataframe):
        """Test feature engineering preprocessing."""
        df = sample_dataframe.copy()

        # Add derived features
        df['pl_density'] = df['pl_masse'] / \
            (df['pl_rade'] ** 3)  # Rough density
        df['st_lum_linear'] = 10 ** df['st_lum']  # Convert log to linear

        # Add HZ features
        from exolife.data.utils import add_hz_edges_to_df
        df = add_hz_edges_to_df(df, lum_col='st_lum_linear')

        # Check new features
        assert 'pl_density' in df.columns
        assert 'st_lum_linear' in df.columns
        assert 'hz_inner' in df.columns
        assert 'hz_outer' in df.columns

        # Validate derived features
        assert all(df['pl_density'] > 0)
        assert all(df['st_lum_linear'] > 0)
        assert all(df['hz_inner'] > 0)


class TestMergeAndPreprocessIntegration:
    """Integration tests for merge and preprocess components."""

    def test_complete_data_pipeline(self, sample_dataframe):
        """Test complete data processing pipeline."""
        # Step 1: Simulate multiple data sources
        nasa_data = sample_dataframe[['pl_name', 'st_teff', 'st_lum']].copy()
        phl_data = sample_dataframe[['pl_name', 'pl_rade', 'pl_masse']].copy()
        gaia_data = sample_dataframe[['pl_name', 'gaia_id']].copy()

        # Step 2: Merge data sources
        merger = MockMerger()
        sources_data = {
            'nasa': nasa_data,
            'phl': phl_data,
            'gaia': gaia_data
        }
        merged_data = merger.merge(sources_data)

        # Step 3: Preprocess merged data
        preprocessor = MockPreprocessor()
        processed_data = preprocessor.preprocess(merged_data)

        # Verify complete pipeline
        assert len(processed_data) == len(
            sample_dataframe) * 3  # Three sources
        assert 'source' in processed_data.columns  # From merger
        assert 'processed' in processed_data.columns  # From preprocessor

        # Each source should be represented
        sources = processed_data['source'].unique()
        assert set(sources) == {'nasa', 'phl', 'gaia'}

    def test_pipeline_error_handling(self):
        """Test pipeline error handling."""
        merger = MockMerger()
        preprocessor = MockPreprocessor()

        # Test with empty data
        empty_merged = merger.merge({})
        assert isinstance(empty_merged, pd.DataFrame)
        assert len(empty_merged) == 0

        # Preprocessing empty data should work
        empty_processed = preprocessor.preprocess(empty_merged)
        assert isinstance(empty_processed, pd.DataFrame)
        assert len(empty_processed) == 0

    def test_pipeline_data_consistency(self, sample_dataframe):
        """Test data consistency through pipeline."""
        # Track original planet names
        original_names = set(sample_dataframe['pl_name'])

        # Run through pipeline
        merger = MockMerger()
        preprocessor = MockPreprocessor()

        sources_data = {'test': sample_dataframe}
        merged = merger.merge(sources_data)
        processed = preprocessor.preprocess(merged)

        # Planet names should be preserved
        final_names = set(processed['pl_name'])
        assert final_names == original_names

        # Should have same number of unique planets
        assert len(final_names) == len(original_names)
