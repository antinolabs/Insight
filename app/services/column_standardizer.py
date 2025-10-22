"""
LLM-Powered Column Standardization Service
Intelligent column name cleaning and standardization using LLM
"""

import json
import re
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import requests
import os
from app.services.smart_data_profiler import ColumnProfile, ColumnIntent, ColumnType

@dataclass
class StandardizationResult:
    """Result of column standardization"""
    original_name: str
    standardized_name: str
    confidence: float
    reasoning: str
    category: str
    suggested_format: str

class LLMColumnStandardizer:
    """
    LLM-powered service for intelligent column name standardization
    """
    
    def __init__(self):
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
        self.llm_provider = os.getenv('LLM_PROVIDER', 'openai')
        self.model = os.getenv('LLM_MODEL', 'gpt-3.5-turbo')
        
        # Standard naming conventions
        self.naming_conventions = {
            'measures': {
                'currency': ['Sales', 'Revenue', 'Amount', 'Price', 'Cost', 'Profit', 'Income'],
                'count': ['Count', 'Quantity', 'Number', 'Total', 'Sum'],
                'rate': ['Rate', 'Percentage', 'Ratio', 'Score', 'Index'],
                'time': ['Duration', 'Hours', 'Minutes', 'Days']
            },
            'dimensions': {
                'geographic': ['Region', 'Country', 'State', 'City', 'Location', 'Area'],
                'temporal': ['Date', 'Time', 'Year', 'Month', 'Day', 'Quarter', 'Period'],
                'categorical': ['Category', 'Type', 'Class', 'Group', 'Segment', 'Status'],
                'organizational': ['Department', 'Team', 'Division', 'Branch', 'Store', 'Unit']
            },
            'identifiers': {
                'primary': ['ID', 'Key', 'Code', 'Number', 'Reference'],
                'composite': ['Customer ID', 'Product Code', 'Order Number', 'Transaction ID']
            }
        }
    
    def standardize_columns(self, column_profiles: List[ColumnProfile], 
                          industry_context: str = "general") -> Dict[str, StandardizationResult]:
        """
        Standardize column names using LLM intelligence
        """
        try:
            # Prepare context for LLM
            context = self._prepare_standardization_context(column_profiles, industry_context)
            
            # Call LLM for standardization
            llm_response = self._call_llm_for_standardization(context)
            
            # Parse and validate results
            standardization_map = self._parse_standardization_response(llm_response, column_profiles)
            
            # Create standardization results
            results = {}
            for profile in column_profiles:
                if profile.name in standardization_map:
                    result = standardization_map[profile.name]
                    results[profile.name] = StandardizationResult(
                        original_name=profile.name,
                        standardized_name=result['standardized_name'],
                        confidence=result['confidence'],
                        reasoning=result['reasoning'],
                        category=result['category'],
                        suggested_format=result['suggested_format']
                    )
                else:
                    # Fallback standardization
                    results[profile.name] = self._fallback_standardization(profile)
            
            return results
            
        except Exception as e:
            print(f"Error in column standardization: {e}")
            # Return fallback results
            return {profile.name: self._fallback_standardization(profile) for profile in column_profiles}
    
    def _prepare_standardization_context(self, column_profiles: List[ColumnProfile], 
                                       industry_context: str) -> str:
        """
        Prepare context for LLM standardization
        """
        context = f"""
You are a data analyst expert specializing in column name standardization. 
I need you to standardize column names for a {industry_context} industry dataset.

For each column, I'll provide:
- Original column name
- Detected data type (numerical, categorical, temporal, text, boolean, identifier)
- Business intent (measure, dimension, identifier, metadata)
- Sample values
- Cardinality information

Please return a JSON object with standardized names following these rules:

1. Use clear, human-readable names (e.g., 'Sales' not 'sls_v2')
2. Use proper capitalization (Title Case for most, ALL CAPS for IDs)
3. Remove abbreviations and technical jargon
4. Be consistent with naming conventions
5. Consider the business context and data type
6. For measures: Use descriptive names like 'Sales Amount', 'Customer Count'
7. For dimensions: Use clear category names like 'Product Category', 'Sales Region'
8. For identifiers: Use clear ID names like 'Customer ID', 'Order Number'
9. For temporal: Use clear date/time names like 'Order Date', 'Created Time'

Column Information:
"""
        
        for profile in column_profiles:
            context += f"""
- Column: "{profile.name}"
  Type: {profile.detected_type.value}
  Intent: {profile.intent.value}
  Cardinality: {profile.cardinality:.2f}
  Sample Values: {profile.sample_values[:3]}
  Suggested Name: "{profile.suggested_name}"
"""
        
        context += """

Please return a JSON object in this format:
{
  "column_name": {
    "standardized_name": "Clean Name",
    "confidence": 0.95,
    "reasoning": "Brief explanation of the standardization",
    "category": "measure|dimension|identifier|temporal",
    "suggested_format": "currency|count|percentage|text|date|id"
  }
}

Focus on creating professional, business-friendly column names that clearly communicate the data's purpose.
"""
        
        return context
    
    def _call_llm_for_standardization(self, context: str) -> str:
        """
        Call LLM for column standardization
        """
        if self.llm_provider == 'openai' and self.openai_api_key:
            return self._call_openai_standardization(context)
        elif self.llm_provider == 'anthropic' and self.anthropic_api_key:
            return self._call_anthropic_standardization(context)
        else:
            return self._fallback_standardization_response()
    
    def _call_openai_standardization(self, context: str) -> str:
        """
        Call OpenAI API for standardization
        """
        try:
            headers = {
                'Authorization': f'Bearer {self.openai_api_key}',
                'Content-Type': 'application/json'
            }
            
            data = {
                'model': self.model,
                'messages': [
                    {
                        'role': 'system', 
                        'content': 'You are a data analyst expert. Return only valid JSON for column standardization.'
                    },
                    {'role': 'user', 'content': context}
                ],
                'temperature': 0.1,  # Low temperature for consistent results
                'max_tokens': 2000
            }
            
            response = requests.post(
                'https://api.openai.com/v1/chat/completions',
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()['choices'][0]['message']['content']
            else:
                raise Exception(f"OpenAI API error: {response.status_code}")
                
        except Exception as e:
            print(f"OpenAI API error: {e}")
            return self._fallback_standardization_response()
    
    def _call_anthropic_standardization(self, context: str) -> str:
        """
        Call Anthropic Claude API for standardization
        """
        try:
            headers = {
                'x-api-key': self.anthropic_api_key,
                'Content-Type': 'application/json',
                'anthropic-version': '2023-06-01'
            }
            
            data = {
                'model': 'claude-3-sonnet-20240229',
                'max_tokens': 2000,
                'messages': [
                    {'role': 'user', 'content': context}
                ]
            }
            
            response = requests.post(
                'https://api.anthropic.com/v1/messages',
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()['content'][0]['text']
            else:
                raise Exception(f"Anthropic API error: {response.status_code}")
                
        except Exception as e:
            print(f"Anthropic API error: {e}")
            return self._fallback_standardization_response()
    
    def _parse_standardization_response(self, response: str, 
                                      column_profiles: List[ColumnProfile]) -> Dict[str, Dict[str, Any]]:
        """
        Parse LLM response and validate results
        """
        try:
            # Extract JSON from response
            if '{' in response and '}' in response:
                start = response.find('{')
                end = response.rfind('}') + 1
                json_str = response[start:end]
                standardization_map = json.loads(json_str)
                
                # Validate and clean results
                validated_map = {}
                for col_name, result in standardization_map.items():
                    if isinstance(result, dict) and 'standardized_name' in result:
                        validated_map[col_name] = {
                            'standardized_name': self._clean_standardized_name(result['standardized_name']),
                            'confidence': float(result.get('confidence', 0.8)),
                            'reasoning': result.get('reasoning', 'LLM standardization'),
                            'category': result.get('category', 'general'),
                            'suggested_format': result.get('suggested_format', 'text')
                        }
                
                return validated_map
            else:
                raise ValueError("No JSON found in response")
                
        except Exception as e:
            print(f"Error parsing standardization response: {e}")
            return {}
    
    def _clean_standardized_name(self, name: str) -> str:
        """
        Clean and validate standardized name
        """
        # Remove quotes if present
        name = name.strip('"\'')
        
        # Basic cleaning
        name = re.sub(r'[^\w\s-]', '', name)
        name = re.sub(r'\s+', ' ', name).strip()
        
        # Ensure it's not empty
        if not name:
            return "Column"
        
        return name
    
    def _fallback_standardization(self, profile: ColumnProfile) -> StandardizationResult:
        """
        Fallback standardization when LLM is not available
        """
        # Basic cleaning
        clean_name = profile.name.lower()
        clean_name = re.sub(r'[^a-zA-Z0-9\s]', ' ', clean_name)
        clean_name = re.sub(r'\s+', ' ', clean_name).strip()
        
        # Apply naming conventions based on intent and type
        if profile.intent == ColumnIntent.MEASURE:
            if profile.detected_type == ColumnType.NUMERICAL:
                if any(keyword in profile.name.lower() for keyword in ['sales', 'revenue', 'amount']):
                    clean_name = 'Sales Amount'
                elif any(keyword in profile.name.lower() for keyword in ['count', 'number', 'total']):
                    clean_name = 'Count'
                else:
                    clean_name = 'Value'
            else:
                clean_name = 'Measure'
        
        elif profile.intent == ColumnIntent.DIMENSION:
            if profile.detected_type == ColumnType.TEMPORAL:
                clean_name = 'Date'
            elif profile.detected_type == ColumnType.CATEGORICAL:
                if any(keyword in profile.name.lower() for keyword in ['region', 'area', 'location']):
                    clean_name = 'Region'
                elif any(keyword in profile.name.lower() for keyword in ['product', 'item']):
                    clean_name = 'Product'
                elif any(keyword in profile.name.lower() for keyword in ['category', 'type']):
                    clean_name = 'Category'
                else:
                    clean_name = 'Category'
            else:
                clean_name = 'Dimension'
        
        elif profile.intent == ColumnIntent.IDENTIFIER:
            clean_name = 'ID'
        
        else:
            clean_name = clean_name.title() if clean_name else 'Column'
        
        return StandardizationResult(
            original_name=profile.name,
            standardized_name=clean_name,
            confidence=0.6,
            reasoning="Fallback standardization",
            category=profile.intent.value,
            suggested_format=self._get_suggested_format(profile)
        )
    
    def _get_suggested_format(self, profile: ColumnProfile) -> str:
        """
        Get suggested format for the column
        """
        if profile.detected_type == ColumnType.NUMERICAL:
            if any(keyword in profile.name.lower() for keyword in ['sales', 'revenue', 'amount', 'price', 'cost']):
                return 'currency'
            elif any(keyword in profile.name.lower() for keyword in ['count', 'number', 'quantity']):
                return 'count'
            elif any(keyword in profile.name.lower() for keyword in ['rate', 'percentage', 'ratio']):
                return 'percentage'
            else:
                return 'number'
        
        elif profile.detected_type == ColumnType.TEMPORAL:
            return 'date'
        
        elif profile.intent == ColumnIntent.IDENTIFIER:
            return 'id'
        
        else:
            return 'text'
    
    def _fallback_standardization_response(self) -> str:
        """
        Fallback response when LLM is not available
        """
        return '{"error": "LLM not available, using fallback standardization"}'
    
    def apply_standardization(self, df, standardization_results: Dict[str, StandardizationResult]) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """
        Apply standardization results to DataFrame
        """
        rename_map = {}
        applied_results = {}
        
        for original_name, result in standardization_results.items():
            if original_name in df.columns:
                # Ensure unique names
                standardized_name = result.standardized_name
                counter = 1
                while standardized_name in rename_map.values() or standardized_name in df.columns:
                    standardized_name = f"{result.standardized_name} {counter}"
                    counter += 1
                
                rename_map[original_name] = standardized_name
                applied_results[original_name] = standardized_name
        
        # Apply renaming
        df_standardized = df.rename(columns=rename_map)
        
        return df_standardized, applied_results
    
    def get_standardization_report(self, standardization_results: Dict[str, StandardizationResult]) -> Dict[str, Any]:
        """
        Generate standardization report
        """
        total_columns = len(standardization_results)
        high_confidence = sum(1 for result in standardization_results.values() if result.confidence > 0.8)
        medium_confidence = sum(1 for result in standardization_results.values() if 0.6 <= result.confidence <= 0.8)
        low_confidence = sum(1 for result in standardization_results.values() if result.confidence < 0.6)
        
        categories = {}
        for result in standardization_results.values():
            category = result.category
            categories[category] = categories.get(category, 0) + 1
        
        return {
            'total_columns': total_columns,
            'confidence_distribution': {
                'high': high_confidence,
                'medium': medium_confidence,
                'low': low_confidence
            },
            'category_distribution': categories,
            'average_confidence': sum(r.confidence for r in standardization_results.values()) / total_columns if total_columns > 0 else 0,
            'standardization_details': [
                {
                    'original': result.original_name,
                    'standardized': result.standardized_name,
                    'confidence': result.confidence,
                    'category': result.category,
                    'reasoning': result.reasoning
                }
                for result in standardization_results.values()
            ]
        }
