"""
Automated Data Preprocessing Service
Intelligent data cleaning and preprocessing based on schema analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from app.services.smart_data_profiler import DataSchema, ColumnProfile, ColumnType, ColumnIntent
from app.services.column_standardizer import StandardizationResult

class AutomatedPreprocessor:
    """
    Automated data preprocessing based on intelligent schema analysis
    """
    
    def __init__(self):
        self.preprocessing_log = []
        self.imputation_strategies = {
            'numerical_measure': 'zero',  # Sales, revenue -> 0
            'numerical_dimension': 'median',  # Age, score -> median
            'categorical': 'mode',  # Category -> most frequent
            'temporal': 'forward_fill',  # Date -> forward fill
            'boolean': 'mode',  # Boolean -> most frequent
            'identifier': 'drop',  # ID -> drop rows with missing IDs
            'text': 'unknown'  # Text -> 'Unknown'
        }
    
    def preprocess_dataframe(self, df: pd.DataFrame, schema: DataSchema, 
                           standardization_results: Dict[str, StandardizationResult]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Comprehensive data preprocessing based on schema and standardization
        """
        preprocessing_info = {
            'original_shape': df.shape,
            'preprocessing_steps': [],
            'data_quality_improvements': {},
            'warnings': [],
            'recommendations': []
        }
        
        # Step 1: Apply column standardization
        df_processed, rename_map = self._apply_column_standardization(df, standardization_results)
        preprocessing_info['preprocessing_steps'].append('Column Standardization')
        preprocessing_info['column_renames'] = rename_map
        
        # Step 2: Handle data type conversions
        df_processed = self._convert_data_types(df_processed, schema)
        preprocessing_info['preprocessing_steps'].append('Data Type Conversion')
        
        # Step 3: Handle missing values
        df_processed, missing_info = self._handle_missing_values(df_processed, schema)
        preprocessing_info['preprocessing_steps'].append('Missing Value Treatment')
        preprocessing_info['missing_value_treatment'] = missing_info
        
        # Step 4: Handle outliers
        df_processed, outlier_info = self._handle_outliers(df_processed, schema)
        preprocessing_info['preprocessing_steps'].append('Outlier Treatment')
        preprocessing_info['outlier_treatment'] = outlier_info
        
        # Step 5: Data validation and cleaning
        df_processed, validation_info = self._validate_and_clean_data(df_processed, schema)
        preprocessing_info['preprocessing_steps'].append('Data Validation')
        preprocessing_info['validation_results'] = validation_info
        
        # Step 6: Feature engineering (basic)
        df_processed, feature_info = self._basic_feature_engineering(df_processed, schema)
        preprocessing_info['preprocessing_steps'].append('Feature Engineering')
        preprocessing_info['feature_engineering'] = feature_info
        
        # Calculate final statistics
        preprocessing_info['final_shape'] = df_processed.shape
        preprocessing_info['data_quality_score'] = self._calculate_final_quality_score(df_processed, schema)
        
        return df_processed, preprocessing_info
    
    def _apply_column_standardization(self, df: pd.DataFrame, 
                                    standardization_results: Dict[str, StandardizationResult]) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """
        Apply column standardization to DataFrame
        """
        rename_map = {}
        
        for original_name, result in standardization_results.items():
            if original_name in df.columns:
                # Ensure unique names
                standardized_name = result.standardized_name
                counter = 1
                while standardized_name in rename_map.values() or standardized_name in df.columns:
                    standardized_name = f"{result.standardized_name} {counter}"
                    counter += 1
                
                rename_map[original_name] = standardized_name
        
        df_renamed = df.rename(columns=rename_map)
        self.preprocessing_log.append(f"Renamed {len(rename_map)} columns")
        
        return df_renamed, rename_map
    
    def _convert_data_types(self, df: pd.DataFrame, schema: DataSchema) -> pd.DataFrame:
        """
        Convert data types based on schema analysis
        """
        df_converted = df.copy()
        conversion_log = []
        
        for profile in schema.columns:
            col_name = profile.name
            if col_name not in df_converted.columns:
                continue
            
            try:
                if profile.detected_type == ColumnType.TEMPORAL:
                    # Convert to datetime
                    df_converted[col_name] = pd.to_datetime(df_converted[col_name], errors='coerce')
                    conversion_log.append(f"Converted {col_name} to datetime")
                
                elif profile.detected_type == ColumnType.NUMERICAL:
                    # Convert to numeric
                    if df_converted[col_name].dtype == 'object':
                        # Handle currency symbols, commas, etc.
                        df_converted[col_name] = df_converted[col_name].astype(str).str.replace(r'[,$%]', '', regex=True)
                        df_converted[col_name] = pd.to_numeric(df_converted[col_name], errors='coerce')
                    conversion_log.append(f"Converted {col_name} to numeric")
                
                elif profile.detected_type == ColumnType.BOOLEAN:
                    # Convert to boolean
                    df_converted[col_name] = self._convert_to_boolean(df_converted[col_name])
                    conversion_log.append(f"Converted {col_name} to boolean")
                
                elif profile.detected_type == ColumnType.CATEGORICAL:
                    # Convert to category
                    df_converted[col_name] = df_converted[col_name].astype('category')
                    conversion_log.append(f"Converted {col_name} to category")
                
            except Exception as e:
                self.preprocessing_log.append(f"Warning: Could not convert {col_name}: {str(e)}")
        
        self.preprocessing_log.extend(conversion_log)
        return df_converted
    
    def _convert_to_boolean(self, series: pd.Series) -> pd.Series:
        """
        Convert series to boolean with intelligent mapping
        """
        # Common boolean mappings
        boolean_mappings = {
            'true': True, 'false': False,
            'yes': True, 'no': False,
            'y': True, 'n': False,
            '1': True, '0': False,
            't': True, 'f': False,
            'on': True, 'off': False,
            'active': True, 'inactive': False,
            'enabled': True, 'disabled': False
        }
        
        # Convert to string and lowercase
        str_series = series.astype(str).str.lower().str.strip()
        
        # Apply mapping
        boolean_series = str_series.map(boolean_mappings)
        
        # Handle unmapped values
        unmapped_mask = boolean_series.isnull() & str_series.notnull()
        if unmapped_mask.any():
            # For unmapped values, try to infer from context
            unique_values = str_series[unmapped_mask].unique()
            for value in unique_values:
                if any(keyword in value for keyword in ['true', 'yes', 'active', 'enabled', '1']):
                    boolean_series.loc[str_series == value] = True
                elif any(keyword in value for keyword in ['false', 'no', 'inactive', 'disabled', '0']):
                    boolean_series.loc[str_series == value] = False
        
        return boolean_series
    
    def _handle_missing_values(self, df: pd.DataFrame, schema: DataSchema) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Handle missing values based on column type and intent
        """
        df_cleaned = df.copy()
        missing_info = {
            'columns_processed': [],
            'imputation_strategies': {},
            'missing_before': {},
            'missing_after': {}
        }
        
        for profile in schema.columns:
            col_name = profile.name
            if col_name not in df_cleaned.columns:
                continue
            
            missing_before = df_cleaned[col_name].isnull().sum()
            missing_info['missing_before'][col_name] = int(missing_before)
            
            if missing_before > 0:
                # Determine imputation strategy
                strategy = self._get_imputation_strategy(profile)
                missing_info['imputation_strategies'][col_name] = strategy
                
                # Apply imputation
                df_cleaned[col_name] = self._apply_imputation(df_cleaned[col_name], strategy, profile)
                
                missing_after = df_cleaned[col_name].isnull().sum()
                missing_info['missing_after'][col_name] = int(missing_after)
                missing_info['columns_processed'].append(col_name)
                
                self.preprocessing_log.append(
                    f"Imputed {missing_before} missing values in {col_name} using {strategy} strategy"
                )
        
        return df_cleaned, missing_info
    
    def _get_imputation_strategy(self, profile: ColumnProfile) -> str:
        """
        Determine appropriate imputation strategy for a column
        """
        if profile.intent == ColumnIntent.IDENTIFIER:
            return 'drop'
        
        if profile.detected_type == ColumnType.NUMERICAL:
            if profile.intent == ColumnIntent.MEASURE:
                return 'zero'  # Sales, revenue -> 0
            else:
                return 'median'  # Age, score -> median
        
        elif profile.detected_type == ColumnType.TEMPORAL:
            return 'forward_fill'
        
        elif profile.detected_type == ColumnType.BOOLEAN:
            return 'mode'
        
        elif profile.detected_type == ColumnType.CATEGORICAL:
            return 'mode'
        
        else:
            return 'unknown'
    
    def _apply_imputation(self, series: pd.Series, strategy: str, profile: ColumnProfile) -> pd.Series:
        """
        Apply imputation strategy to a series
        """
        if strategy == 'drop':
            # Drop rows with missing values (for identifiers)
            return series.dropna()
        
        elif strategy == 'zero':
            return series.fillna(0)
        
        elif strategy == 'median':
            if profile.detected_type == ColumnType.NUMERICAL:
                return series.fillna(series.median())
            else:
                return series.fillna(series.mode()[0] if not series.mode().empty else 'Unknown')
        
        elif strategy == 'mode':
            mode_value = series.mode()[0] if not series.mode().empty else 'Unknown'
            return series.fillna(mode_value)
        
        elif strategy == 'forward_fill':
            return series.fillna(method='ffill').fillna(method='bfill')
        
        elif strategy == 'unknown':
            return series.fillna('Unknown')
        
        else:
            return series.fillna('Unknown')
    
    def _handle_outliers(self, df: pd.DataFrame, schema: DataSchema) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Handle outliers in numerical columns
        """
        df_processed = df.copy()
        outlier_info = {
            'columns_processed': [],
            'outliers_detected': {},
            'outliers_treated': {},
            'treatment_method': {}
        }
        
        for profile in schema.columns:
            if (profile.detected_type == ColumnType.NUMERICAL and 
                profile.intent == ColumnIntent.MEASURE and 
                profile.name in df_processed.columns):
                
                col_name = profile.name
                outliers = self._detect_outliers(df_processed[col_name])
                outlier_info['outliers_detected'][col_name] = len(outliers)
                
                if len(outliers) > 0:
                    # Cap outliers instead of removing them
                    Q1 = df_processed[col_name].quantile(0.25)
                    Q3 = df_processed[col_name].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    df_processed[col_name] = df_processed[col_name].clip(lower_bound, upper_bound)
                    
                    outliers_after = self._detect_outliers(df_processed[col_name])
                    outlier_info['outliers_treated'][col_name] = len(outliers) - len(outliers_after)
                    outlier_info['treatment_method'][col_name] = 'capping'
                    outlier_info['columns_processed'].append(col_name)
                    
                    self.preprocessing_log.append(
                        f"Capped {len(outliers)} outliers in {col_name}"
                    )
        
        return df_processed, outlier_info
    
    def _detect_outliers(self, series: pd.Series) -> List[int]:
        """
        Detect outliers using IQR method
        """
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = series[(series < lower_bound) | (series > upper_bound)]
        return outliers.index.tolist()
    
    def _validate_and_clean_data(self, df: pd.DataFrame, schema: DataSchema) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Validate and clean data based on business rules
        """
        df_cleaned = df.copy()
        validation_info = {
            'validation_rules_applied': [],
            'data_quality_issues': [],
            'cleaning_actions': []
        }
        
        for profile in schema.columns:
            col_name = profile.name
            if col_name not in df_cleaned.columns:
                continue
            
            # Validate based on column type and intent
            if profile.detected_type == ColumnType.NUMERICAL and profile.intent == ColumnIntent.MEASURE:
                # Check for negative values in measures that shouldn't be negative
                if any(keyword in col_name.lower() for keyword in ['sales', 'revenue', 'amount', 'price', 'count']):
                    negative_count = (df_cleaned[col_name] < 0).sum()
                    if negative_count > 0:
                        df_cleaned[col_name] = df_cleaned[col_name].clip(lower=0)
                        validation_info['cleaning_actions'].append(
                            f"Fixed {negative_count} negative values in {col_name}"
                        )
            
            elif profile.detected_type == ColumnType.TEMPORAL:
                # Check for future dates in historical data
                if 'date' in col_name.lower():
                    future_dates = (df_cleaned[col_name] > datetime.now()).sum()
                    if future_dates > 0:
                        validation_info['data_quality_issues'].append(
                            f"Found {future_dates} future dates in {col_name}"
                        )
            
            elif profile.detected_type == ColumnType.CATEGORICAL:
                # Check for inconsistent categories
                if df_cleaned[col_name].dtype == 'object':
                    # Standardize case
                    df_cleaned[col_name] = df_cleaned[col_name].astype(str).str.strip()
        
        validation_info['validation_rules_applied'] = [
            'Negative value correction for measures',
            'Date validation',
            'Text standardization'
        ]
        
        return df_cleaned, validation_info
    
    def _basic_feature_engineering(self, df: pd.DataFrame, schema: DataSchema) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Basic feature engineering based on schema
        """
        df_engineered = df.copy()
        feature_info = {
            'features_created': [],
            'feature_types': {}
        }
        
        # Extract date components from temporal columns
        for profile in schema.columns:
            if (profile.detected_type == ColumnType.TEMPORAL and 
                profile.name in df_engineered.columns):
                
                col_name = profile.name
                try:
                    # Extract year, month, day, day_of_week
                    df_engineered[f'{col_name}_Year'] = df_engineered[col_name].dt.year
                    df_engineered[f'{col_name}_Month'] = df_engineered[col_name].dt.month
                    df_engineered[f'{col_name}_Day'] = df_engineered[col_name].dt.day
                    df_engineered[f'{col_name}_DayOfWeek'] = df_engineered[col_name].dt.dayofweek
                    df_engineered[f'{col_name}_Quarter'] = df_engineered[col_name].dt.quarter
                    
                    feature_info['features_created'].extend([
                        f'{col_name}_Year', f'{col_name}_Month', f'{col_name}_Day',
                        f'{col_name}_DayOfWeek', f'{col_name}_Quarter'
                    ])
                    feature_info['feature_types'][col_name] = 'temporal_extraction'
                    
                except Exception as e:
                    self.preprocessing_log.append(f"Could not extract date features from {col_name}: {str(e)}")
        
        # Create aggregation features for measures
        measures = [profile.name for profile in schema.columns 
                   if profile.intent == ColumnIntent.MEASURE and profile.name in df_engineered.columns]
        
        if len(measures) > 1:
            # Create total and average features
            df_engineered['Total_Measures'] = df_engineered[measures].sum(axis=1)
            df_engineered['Average_Measures'] = df_engineered[measures].mean(axis=1)
            
            feature_info['features_created'].extend(['Total_Measures', 'Average_Measures'])
            feature_info['feature_types']['measures'] = 'aggregation'
        
        return df_engineered, feature_info
    
    def _calculate_final_quality_score(self, df: pd.DataFrame, schema: DataSchema) -> float:
        """
        Calculate final data quality score after preprocessing
        """
        total_cells = df.size
        missing_cells = df.isnull().sum().sum()
        completeness = 1 - (missing_cells / total_cells) if total_cells > 0 else 0
        
        # Check for data type consistency
        type_consistency = 1.0
        for col in df.columns:
            if df[col].dtype == 'object':
                # Check if object column should be numeric
                try:
                    pd.to_numeric(df[col], errors='raise')
                    type_consistency *= 0.9  # Slight penalty for object columns that could be numeric
                except:
                    pass
        
        # Check for duplicate rows
        duplicate_ratio = df.duplicated().sum() / len(df)
        uniqueness = 1 - duplicate_ratio
        
        # Overall quality score
        quality_score = (completeness * 0.4 + type_consistency * 0.3 + uniqueness * 0.3)
        
        return round(quality_score, 3)
    
    def get_preprocessing_summary(self, preprocessing_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate preprocessing summary report
        """
        return {
            'preprocessing_completed': True,
            'original_shape': preprocessing_info['original_shape'],
            'final_shape': preprocessing_info['final_shape'],
            'data_quality_improvement': preprocessing_info['data_quality_score'],
            'steps_completed': preprocessing_info['preprocessing_steps'],
            'columns_renamed': len(preprocessing_info.get('column_renames', {})),
            'missing_values_treated': sum(
                preprocessing_info.get('missing_value_treatment', {}).get('missing_before', {}).values()
            ),
            'outliers_treated': sum(
                preprocessing_info.get('outlier_treatment', {}).get('outliers_treated', {}).values()
            ),
            'features_created': len(
                preprocessing_info.get('feature_engineering', {}).get('features_created', [])
            ),
            'warnings': preprocessing_info.get('warnings', []),
            'recommendations': preprocessing_info.get('recommendations', [])
        }
