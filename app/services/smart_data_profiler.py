"""
Smart Data Profiling Service
Advanced data profiling with auto-detection, classification, and contextualization
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import re
from dataclasses import dataclass, field
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

class ColumnType(Enum):
    """Enhanced column type classification"""
    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"
    TEMPORAL = "temporal"
    TEXT = "text"
    BOOLEAN = "boolean"
    IDENTIFIER = "identifier"
    UNKNOWN = "unknown"

class ColumnIntent(Enum):
    """Column intent classification"""
    MEASURE = "measure"          # Numbers to be aggregated (Sales, Revenue, Count)
    DIMENSION = "dimension"      # Categories to group by (Region, Product, Date)
    IDENTIFIER = "identifier"    # Unique keys (Customer ID, Order ID)
    METADATA = "metadata"        # Non-analytical data (Notes, Comments)

@dataclass
class ColumnProfile:
    """Comprehensive column profile"""
    name: str
    original_name: str
    detected_type: ColumnType
    intent: ColumnIntent
    cardinality: int
    null_count: int
    null_percentage: float
    unique_count: int
    sample_values: List[Any] = field(default_factory=list)
    statistics: Dict[str, Any] = field(default_factory=dict)
    patterns: List[str] = field(default_factory=list)
    confidence_score: float = 0.0
    suggested_name: str = ""
    is_pii: bool = False
    pii_type: Optional[str] = None

@dataclass
class DataSchema:
    """Complete data schema with contextual information"""
    columns: List[ColumnProfile]
    total_rows: int
    total_columns: int
    data_quality_score: float
    suggested_measures: List[str] = field(default_factory=list)
    suggested_dimensions: List[str] = field(default_factory=list)
    temporal_columns: List[str] = field(default_factory=list)
    pii_columns: List[str] = field(default_factory=list)
    industry_context: str = ""
    business_domain: str = ""

class SmartDataProfiler:
    """
    Advanced data profiler with intelligent type detection and classification
    """
    
    def __init__(self):
        self.temporal_patterns = [
            r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
            r'\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY
            r'\d{2}-\d{2}-\d{4}',  # MM-DD-YYYY
            r'\d{4}/\d{2}/\d{2}',  # YYYY/MM/DD
            r'\d{1,2}/\d{1,2}/\d{2,4}',  # Flexible date
            r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}',  # DateTime
            r'\d{2}:\d{2}:\d{2}',  # Time
        ]
        
        self.pii_patterns = {
            'email': r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
            'phone': r'(\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}',
            'ssn': r'\d{3}-\d{2}-\d{4}',
            'credit_card': r'\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}',
            'address': r'\d+\s+[a-zA-Z\s]+(?:street|st|avenue|ave|road|rd|drive|dr|lane|ln|boulevard|blvd)',
        }
        
        self.measure_keywords = [
            'sales', 'revenue', 'amount', 'price', 'cost', 'profit', 'margin',
            'quantity', 'count', 'total', 'sum', 'avg', 'average', 'rate',
            'percentage', 'ratio', 'score', 'value', 'worth', 'income',
            'expense', 'budget', 'fee', 'charge', 'payment', 'balance'
        ]
        
        self.dimension_keywords = [
            'region', 'country', 'state', 'city', 'location', 'area', 'zone',
            'product', 'category', 'type', 'class', 'group', 'segment',
            'customer', 'client', 'user', 'person', 'employee', 'staff',
            'department', 'division', 'team', 'unit', 'branch', 'store',
            'date', 'time', 'year', 'month', 'day', 'quarter', 'period',
            'status', 'stage', 'phase', 'level', 'grade', 'rank', 'tier'
        ]
        
        self.id_keywords = [
            'id', 'key', 'code', 'number', 'num', 'ref', 'reference',
            'uuid', 'guid', 'token', 'hash', 'serial', 'sequence'
        ]

    def profile_dataframe(self, df: pd.DataFrame, industry_context: str = "general") -> DataSchema:
        """
        Create comprehensive data profile with smart classification
        """
        profiles = []
        
        for col in df.columns:
            profile = self._profile_column(df, col, industry_context)
            profiles.append(profile)
        
        # Create schema
        schema = DataSchema(
            columns=profiles,
            total_rows=len(df),
            total_columns=len(df.columns),
            data_quality_score=self._calculate_data_quality_score(df, profiles),
            industry_context=industry_context
        )
        
        # Analyze schema for business context
        self._analyze_business_context(schema)
        
        return schema
    
    def _profile_column(self, df: pd.DataFrame, col_name: str, industry_context: str) -> ColumnProfile:
        """
        Create detailed profile for a single column
        """
        series = df[col_name]
        
        # Basic statistics
        null_count = series.isnull().sum()
        null_percentage = (null_count / len(series)) * 100
        unique_count = series.nunique()
        cardinality = unique_count / len(series) if len(series) > 0 else 0
        
        # Detect column type
        detected_type, confidence = self._detect_column_type(series, col_name)
        
        # Determine intent
        intent = self._determine_column_intent(series, col_name, detected_type, industry_context)
        
        # Get sample values
        sample_values = self._get_sample_values(series)
        
        # Calculate statistics
        statistics = self._calculate_statistics(series, detected_type)
        
        # Detect patterns
        patterns = self._detect_patterns(series, col_name)
        
        # Check for PII
        is_pii, pii_type = self._detect_pii(series, col_name)
        
        # Suggest clean name
        suggested_name = self._suggest_clean_name(col_name, intent, detected_type)
        
        return ColumnProfile(
            name=col_name,
            original_name=col_name,
            detected_type=detected_type,
            intent=intent,
            cardinality=cardinality,
            null_count=int(null_count),
            null_percentage=null_percentage,
            unique_count=unique_count,
            sample_values=sample_values,
            statistics=statistics,
            patterns=patterns,
            confidence_score=confidence,
            suggested_name=suggested_name,
            is_pii=is_pii,
            pii_type=pii_type
        )
    
    def _detect_column_type(self, series: pd.Series, col_name: str) -> Tuple[ColumnType, float]:
        """
        Intelligently detect column type with confidence scoring
        """
        # Handle null series
        if series.isnull().all():
            return ColumnType.UNKNOWN, 0.0
        
        non_null_series = series.dropna()
        if len(non_null_series) == 0:
            return ColumnType.UNKNOWN, 0.0
        
        # Check for temporal data
        if self._is_temporal(series, col_name):
            return ColumnType.TEMPORAL, 0.9
        
        # Check for boolean data
        if self._is_boolean(series):
            return ColumnType.BOOLEAN, 0.95
        
        # Check for numerical data
        if self._is_numerical(series):
            return ColumnType.NUMERICAL, 0.9
        
        # Check for categorical data
        if self._is_categorical(series, col_name):
            return ColumnType.CATEGORICAL, 0.8
        
        # Check for identifier
        if self._is_identifier(series, col_name):
            return ColumnType.IDENTIFIER, 0.85
        
        # Default to text
        return ColumnType.TEXT, 0.6
    
    def _is_temporal(self, series: pd.Series, col_name: str) -> bool:
        """Check if series contains temporal data"""
        # Check column name for temporal keywords
        col_lower = col_name.lower()
        if any(keyword in col_lower for keyword in ['date', 'time', 'created', 'updated', 'timestamp']):
            return True
        
        # Check data patterns
        sample_values = series.dropna().head(100)
        if len(sample_values) == 0:
            return False
        
        # Try to parse as datetime
        try:
            pd.to_datetime(sample_values, errors='raise')
            return True
        except:
            pass
        
        # Check for temporal patterns in string data
        if series.dtype == 'object':
            pattern_matches = 0
            for value in sample_values:
                if any(re.search(pattern, str(value)) for pattern in self.temporal_patterns):
                    pattern_matches += 1
            
            return pattern_matches / len(sample_values) > 0.5
        
        return False
    
    def _is_boolean(self, series: pd.Series) -> bool:
        """Check if series contains boolean data"""
        unique_values = set(series.dropna().astype(str).str.lower())
        
        # Common boolean patterns
        boolean_patterns = [
            {'true', 'false'},
            {'yes', 'no'},
            {'y', 'n'},
            {'1', '0'},
            {'t', 'f'},
            {'on', 'off'},
            {'active', 'inactive'},
            {'enabled', 'disabled'}
        ]
        
        for pattern in boolean_patterns:
            if unique_values.issubset(pattern):
                return True
        
        return False
    
    def _is_numerical(self, series: pd.Series) -> bool:
        """Check if series contains numerical data"""
        # Already numeric
        if pd.api.types.is_numeric_dtype(series):
            return True
        
        # Try to convert to numeric
        try:
            pd.to_numeric(series, errors='raise')
            return True
        except:
            pass
        
        # Check if object series contains mostly numbers
        if series.dtype == 'object':
            sample_values = series.dropna().head(1000)
            if len(sample_values) == 0:
                return False
            
            numeric_count = 0
            for value in sample_values:
                try:
                    float(str(value).replace(',', '').replace('$', '').replace('%', ''))
                    numeric_count += 1
                except:
                    pass
            
            return numeric_count / len(sample_values) > 0.8
        
        return False
    
    def _is_categorical(self, series: pd.Series, col_name: str) -> bool:
        """Check if series contains categorical data"""
        unique_count = series.nunique()
        total_count = len(series)
        
        # High cardinality suggests not categorical
        if unique_count / total_count > 0.5:
            return False
        
        # Check for reasonable number of categories
        if unique_count > 1000:
            return False
        
        # Check if values look like categories
        if series.dtype == 'object':
            sample_values = series.dropna().head(100)
            # Categories should be relatively short strings
            avg_length = np.mean([len(str(val)) for val in sample_values])
            if avg_length > 50:  # Very long strings are likely text, not categories
                return False
        
        return True
    
    def _is_identifier(self, series: pd.Series, col_name: str) -> bool:
        """Check if series contains identifier data"""
        col_lower = col_name.lower()
        
        # Check column name for ID keywords
        if any(keyword in col_lower for keyword in self.id_keywords):
            return True
        
        # Check if all values are unique (high cardinality)
        if series.nunique() / len(series) > 0.95:
            return True
        
        # Check for ID-like patterns
        if series.dtype == 'object':
            sample_values = series.dropna().head(100)
            id_patterns = [
                r'^[A-Z0-9]{6,}$',  # Alphanumeric codes
                r'^\d{6,}$',        # Long numbers
                r'^[A-Z]{2,}\d{3,}$',  # Letter-number combinations
            ]
            
            pattern_matches = 0
            for value in sample_values:
                if any(re.search(pattern, str(value)) for pattern in id_patterns):
                    pattern_matches += 1
            
            return pattern_matches / len(sample_values) > 0.7
        
        return False
    
    def _determine_column_intent(self, series: pd.Series, col_name: str, 
                                col_type: ColumnType, industry_context: str) -> ColumnIntent:
        """
        Determine the business intent of a column
        """
        col_lower = col_name.lower()
        
        # Check for identifier keywords
        if any(keyword in col_lower for keyword in self.id_keywords):
            return ColumnIntent.IDENTIFIER
        
        # Check for measure keywords
        if any(keyword in col_lower for keyword in self.measure_keywords):
            return ColumnIntent.MEASURE
        
        # Check for dimension keywords
        if any(keyword in col_lower for keyword in self.dimension_keywords):
            return ColumnIntent.DIMENSION
        
        # Type-based classification
        if col_type == ColumnType.NUMERICAL:
            # High cardinality numerical = likely measure
            if series.nunique() / len(series) > 0.1:
                return ColumnIntent.MEASURE
            else:
                return ColumnIntent.DIMENSION
        elif col_type == ColumnType.TEMPORAL:
            return ColumnIntent.DIMENSION
        elif col_type == ColumnType.CATEGORICAL:
            return ColumnIntent.DIMENSION
        elif col_type == ColumnType.IDENTIFIER:
            return ColumnIntent.IDENTIFIER
        else:
            return ColumnIntent.METADATA
    
    def _get_sample_values(self, series: pd.Series, n: int = 5) -> List[Any]:
        """Get representative sample values"""
        non_null_series = series.dropna()
        if len(non_null_series) == 0:
            return []
        
        # Get diverse samples
        if len(non_null_series) <= n:
            return non_null_series.tolist()
        
        # Get first, last, and some middle values
        indices = [0, len(non_null_series)//4, len(non_null_series)//2, 
                  3*len(non_null_series)//4, len(non_null_series)-1]
        return [non_null_series.iloc[i] for i in indices if i < len(non_null_series)]
    
    def _calculate_statistics(self, series: pd.Series, col_type: ColumnType) -> Dict[str, Any]:
        """Calculate type-specific statistics"""
        stats = {}
        
        if col_type == ColumnType.NUMERICAL:
            try:
                numeric_series = pd.to_numeric(series, errors='coerce')
                stats = {
                    'mean': float(numeric_series.mean()) if not numeric_series.isnull().all() else None,
                    'median': float(numeric_series.median()) if not numeric_series.isnull().all() else None,
                    'std': float(numeric_series.std()) if not numeric_series.isnull().all() else None,
                    'min': float(numeric_series.min()) if not numeric_series.isnull().all() else None,
                    'max': float(numeric_series.max()) if not numeric_series.isnull().all() else None,
                    'q25': float(numeric_series.quantile(0.25)) if not numeric_series.isnull().all() else None,
                    'q75': float(numeric_series.quantile(0.75)) if not numeric_series.isnull().all() else None,
                }
            except:
                pass
        
        elif col_type == ColumnType.CATEGORICAL:
            value_counts = series.value_counts()
            stats = {
                'top_values': value_counts.head(5).to_dict(),
                'value_distribution': (value_counts / len(series)).head(10).to_dict()
            }
        
        elif col_type == ColumnType.TEMPORAL:
            try:
                temporal_series = pd.to_datetime(series, errors='coerce')
                if not temporal_series.isnull().all():
                    stats = {
                        'earliest': temporal_series.min().isoformat(),
                        'latest': temporal_series.max().isoformat(),
                        'span_days': (temporal_series.max() - temporal_series.min()).days
                    }
            except:
                pass
        
        return stats
    
    def _detect_patterns(self, series: pd.Series, col_name: str) -> List[str]:
        """Detect patterns in the data"""
        patterns = []
        
        if series.dtype == 'object':
            sample_values = series.dropna().head(100)
            
            # Check for common patterns
            if all(str(val).startswith('http') for val in sample_values if val):
                patterns.append('URL')
            
            if all(re.search(self.pii_patterns['email'], str(val)) for val in sample_values if val):
                patterns.append('Email')
            
            if all(re.search(self.pii_patterns['phone'], str(val)) for val in sample_values if val):
                patterns.append('Phone')
        
        return patterns
    
    def _detect_pii(self, series: pd.Series, col_name: str) -> Tuple[bool, Optional[str]]:
        """Detect personally identifiable information"""
        col_lower = col_name.lower()
        
        # Check column name for PII keywords
        pii_keywords = {
            'email': ['email', 'e-mail', 'mail'],
            'phone': ['phone', 'mobile', 'cell', 'telephone'],
            'ssn': ['ssn', 'social', 'security'],
            'address': ['address', 'street', 'location', 'zip', 'postal'],
            'name': ['name', 'first', 'last', 'full'],
            'id': ['id', 'identifier', 'key']
        }
        
        for pii_type, keywords in pii_keywords.items():
            if any(keyword in col_lower for keyword in keywords):
                return True, pii_type
        
        # Check data patterns
        if series.dtype == 'object':
            sample_values = series.dropna().head(50)
            
            for pii_type, pattern in self.pii_patterns.items():
                matches = sum(1 for val in sample_values if re.search(pattern, str(val)))
                if matches / len(sample_values) > 0.3:
                    return True, pii_type
        
        return False, None
    
    def _suggest_clean_name(self, col_name: str, intent: ColumnIntent, col_type: ColumnType) -> str:
        """Suggest a clean, standardized column name"""
        # This will be enhanced by the LLM service
        # For now, basic cleaning
        clean_name = col_name.lower()
        clean_name = re.sub(r'[^a-zA-Z0-9\s]', ' ', clean_name)
        clean_name = re.sub(r'\s+', ' ', clean_name).strip()
        clean_name = clean_name.title()
        
        return clean_name
    
    def _calculate_data_quality_score(self, df: pd.DataFrame, profiles: List[ColumnProfile]) -> float:
        """Calculate overall data quality score"""
        if len(profiles) == 0:
            return 0.0
        
        scores = []
        for profile in profiles:
            # Completeness score
            completeness = 1 - (profile.null_percentage / 100)
            
            # Consistency score (based on type detection confidence)
            consistency = profile.confidence_score
            
            # Validity score (based on patterns and PII detection)
            validity = 1.0 if not profile.is_pii else 0.8
            
            # Overall column score
            column_score = (completeness * 0.4 + consistency * 0.4 + validity * 0.2)
            scores.append(column_score)
        
        return float(np.mean(scores))
    
    def _analyze_business_context(self, schema: DataSchema):
        """Analyze schema for business context and suggestions"""
        # Identify measures and dimensions
        measures = [col.name for col in schema.columns if col.intent == ColumnIntent.MEASURE]
        dimensions = [col.name for col in schema.columns if col.intent == ColumnIntent.DIMENSION]
        temporal_cols = [col.name for col in schema.columns if col.detected_type == ColumnType.TEMPORAL]
        pii_cols = [col.name for col in schema.columns if col.is_pii]
        
        schema.suggested_measures = measures
        schema.suggested_dimensions = dimensions
        schema.temporal_columns = temporal_cols
        schema.pii_columns = pii_cols
        
        # Determine business domain
        if any('sales' in col.name.lower() for col in schema.columns):
            schema.business_domain = 'sales'
        elif any('customer' in col.name.lower() for col in schema.columns):
            schema.business_domain = 'customer'
        elif any('product' in col.name.lower() for col in schema.columns):
            schema.business_domain = 'product'
        else:
            schema.business_domain = 'general'

    def get_schema_summary(self, schema: DataSchema) -> Dict[str, Any]:
        """Get a summary of the data schema"""
        return {
            'total_rows': schema.total_rows,
            'total_columns': schema.total_columns,
            'data_quality_score': schema.data_quality_score,
            'business_domain': schema.business_domain,
            'measures': schema.suggested_measures,
            'dimensions': schema.suggested_dimensions,
            'temporal_columns': schema.temporal_columns,
            'pii_columns': schema.pii_columns,
            'column_profiles': [
                {
                    'name': col.name,
                    'type': col.detected_type.value,
                    'intent': col.intent.value,
                    'cardinality': col.cardinality,
                    'null_percentage': col.null_percentage,
                    'suggested_name': col.suggested_name,
                    'is_pii': col.is_pii,
                    'confidence': col.confidence_score
                }
                for col in schema.columns
            ]
        }
