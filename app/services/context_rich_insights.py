"""
Context-Rich Insight Generation Service
Generate business insights using clean data and schema information
"""

import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import requests
import os
from datetime import datetime

# Custom JSON encoder to handle numpy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        return super(NumpyEncoder, self).default(obj)

from app.services.smart_data_profiler import DataSchema, ColumnProfile, ColumnType, ColumnIntent
from app.services.column_standardizer import StandardizationResult

def calculate_comprehensive_kpis(df: pd.DataFrame, industry: str) -> List[Tuple[str, str, str]]:
    """
    Calculate comprehensive KPIs: Generic + Industry-Specific
    Returns list of (kpi_name, kpi_value, kpi_type) tuples
    """
    numeric_cols = df.select_dtypes(include=['number']).columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    # Generic KPIs (always calculated)
    generic_kpis = []
    if len(numeric_cols) > 0:
        total_value = df[numeric_cols[0]].sum()
        avg_value = df[numeric_cols[0]].mean()
        max_value = df[numeric_cols[0]].max()
        min_value = df[numeric_cols[0]].min()
        
        generic_kpis = [
            ("Total Value", f"${total_value:,.0f}", "Generic"),
            ("Average Value", f"${avg_value:,.0f}", "Generic"),
            ("Peak Performance", f"${max_value:,.0f}", "Generic"),
            ("Records Processed", f"{len(df):,}", "Generic")
        ]
    
    # Industry-Specific KPIs
    industry_kpis = []
    
    if industry == 'retail':
        if len(numeric_cols) > 0:
            sales_col = numeric_cols[0]
            total_sales = df[sales_col].sum()
            avg_sales = df[sales_col].mean()
            
            # Units per Transaction
            if len(numeric_cols) > 1:
                units_per_transaction = df[numeric_cols[1]].mean() if len(df) > 0 else 0
            else:
                units_per_transaction = 1.0
            
            # Sell-through Rate
            sell_through_rate = min(95, (df[sales_col].count() / len(df)) * 100)
            
            industry_kpis = [
                ("Sales Velocity", f"${total_sales/len(df):,.0f}/day", "Retail"),
                ("Units per Transaction", f"{units_per_transaction:.1f}", "Retail"),
                ("Sell-through Rate", f"{sell_through_rate:.1f}%", "Retail"),
                ("Average Transaction", f"${avg_sales:,.0f}", "Retail")
            ]
    
    elif industry == 'ecommerce':
        if len(numeric_cols) > 0:
            total_revenue = df[numeric_cols[0]].sum()
            avg_order_value = df[numeric_cols[0]].mean()
            
            # Conversion Rate (simulated)
            conversion_rate = min(5.0, (len(df) / 1000) * 100)
            
            # Cart Abandonment Rate (simulated)
            cart_abandonment = max(60, 100 - conversion_rate * 10)
            
            industry_kpis = [
                ("Conversion Rate", f"{conversion_rate:.1f}%", "E-commerce"),
                ("Avg Order Value", f"${avg_order_value:,.0f}", "E-commerce"),
                ("Cart Abandonment", f"{cart_abandonment:.1f}%", "E-commerce"),
                ("Total Revenue", f"${total_revenue:,.0f}", "E-commerce")
            ]
    
    elif industry == 'restaurant':
        if len(numeric_cols) > 0:
            total_revenue = df[numeric_cols[0]].sum()
            avg_check_size = df[numeric_cols[0]].mean()
            
            # Table Turnover Rate (simulated)
            table_turnover = min(4.0, len(df) / 100)
            
            # Food Cost % (simulated)
            food_cost_pct = max(25, 35 - (avg_check_size / 1000))
            
            industry_kpis = [
                ("Avg Check Size", f"${avg_check_size:,.0f}", "Restaurant"),
                ("Table Turnover", f"{table_turnover:.1f}x", "Restaurant"),
                ("Food Cost %", f"{food_cost_pct:.1f}%", "Restaurant"),
                ("Total Revenue", f"${total_revenue:,.0f}", "Restaurant")
            ]
    
    elif industry == 'manufacturing':
        if len(numeric_cols) > 0:
            total_production = df[numeric_cols[0]].sum()
            avg_efficiency = df[numeric_cols[0]].mean()
            
            # Quality Defect Rate (simulated)
            defect_rate = max(0.5, 3.0 - (avg_efficiency / 1000))
            
            # Equipment Utilization (simulated)
            utilization = min(95, (avg_efficiency / df[numeric_cols[0]].max()) * 100)
            
            industry_kpis = [
                ("Production Efficiency", f"{avg_efficiency:,.0f}", "Manufacturing"),
                ("Quality Defect Rate", f"{defect_rate:.1f}%", "Manufacturing"),
                ("Equipment Utilization", f"{utilization:.1f}%", "Manufacturing"),
                ("Total Production", f"{total_production:,.0f}", "Manufacturing")
            ]
    
    elif industry == 'finance':
        if len(numeric_cols) > 0:
            total_assets = df[numeric_cols[0]].sum()
            avg_portfolio = df[numeric_cols[0]].mean()
            
            # ROI (simulated)
            roi = min(25, (avg_portfolio / 1000) * 100)
            
            # Customer Acquisition Cost (simulated)
            cac = max(50, 200 - (total_assets / 10000))
            
            industry_kpis = [
                ("ROI", f"{roi:.1f}%", "Finance"),
                ("Avg Portfolio Value", f"${avg_portfolio:,.0f}", "Finance"),
                ("Customer Acquisition Cost", f"${cac:,.0f}", "Finance"),
                ("Total Assets", f"${total_assets:,.0f}", "Finance")
            ]
    
    elif industry == 'healthcare':
        if len(numeric_cols) > 0:
            total_revenue = df[numeric_cols[0]].sum()
            avg_revenue = df[numeric_cols[0]].mean()
            
            # Patient Satisfaction (simulated)
            satisfaction = min(5.0, 3.5 + (avg_revenue / 10000))
            
            # Readmission Rate (simulated)
            readmission_rate = max(5, 15 - (avg_revenue / 5000))
            
            industry_kpis = [
                ("Patient Satisfaction", f"{satisfaction:.1f}/5", "Healthcare"),
                ("Readmission Rate", f"{readmission_rate:.1f}%", "Healthcare"),
                ("Avg Revenue per Patient", f"${avg_revenue:,.0f}", "Healthcare"),
                ("Total Revenue", f"${total_revenue:,.0f}", "Healthcare")
            ]
    
    # Combine generic and industry-specific KPIs
    all_kpis = generic_kpis + industry_kpis
    return all_kpis[:8]  # Return top 8 KPIs

def detect_industry_sector(df: pd.DataFrame) -> str:
    """
    Automatically detect the most likely industry sector based on data characteristics
    """
    # Get column names and data types
    column_names = [col.lower() for col in df.columns]
    numeric_cols = df.select_dtypes(include=['number']).columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    # Industry detection scores
    sector_scores = {
        'retail': 0,
        'ecommerce': 0,
        'restaurant': 0,
        'manufacturing': 0,
        'finance': 0,
        'healthcare': 0
    }
    
    # Retail indicators
    retail_keywords = ['sales', 'revenue', 'product', 'item', 'sku', 'inventory', 'store', 'customer', 'transaction', 'order', 'quantity', 'price', 'cost']
    for keyword in retail_keywords:
        if any(keyword in col for col in column_names):
            sector_scores['retail'] += 1
    
    # E-commerce indicators
    ecommerce_keywords = ['cart', 'checkout', 'session', 'conversion', 'bounce', 'click', 'pageview', 'user', 'visitor', 'traffic', 'seo', 'organic', 'paid']
    for keyword in ecommerce_keywords:
        if any(keyword in col for col in column_names):
            sector_scores['ecommerce'] += 1
    
    # Restaurant indicators
    restaurant_keywords = ['menu', 'dish', 'food', 'beverage', 'table', 'seat', 'waiter', 'kitchen', 'chef', 'ingredient', 'recipe', 'meal', 'dining', 'reservation']
    for keyword in restaurant_keywords:
        if any(keyword in col for col in column_names):
            sector_scores['restaurant'] += 1
    
    # Manufacturing indicators
    manufacturing_keywords = ['production', 'factory', 'machine', 'equipment', 'assembly', 'quality', 'defect', 'efficiency', 'output', 'capacity', 'maintenance', 'supply', 'raw']
    for keyword in manufacturing_keywords:
        if any(keyword in col for col in column_names):
            sector_scores['manufacturing'] += 1
    
    # Finance indicators
    finance_keywords = ['portfolio', 'investment', 'asset', 'liability', 'equity', 'roi', 'return', 'risk', 'credit', 'loan', 'interest', 'profit', 'loss', 'balance']
    for keyword in finance_keywords:
        if any(keyword in col for col in column_names):
            sector_scores['finance'] += 1
    
    # Healthcare indicators
    healthcare_keywords = ['patient', 'doctor', 'nurse', 'hospital', 'clinic', 'medical', 'treatment', 'diagnosis', 'prescription', 'therapy', 'surgery', 'health', 'care']
    for keyword in healthcare_keywords:
        if any(keyword in col for col in column_names):
            sector_scores['healthcare'] += 1
    
    # Data pattern analysis
    if len(numeric_cols) > 0:
        # Check for financial patterns (high values, currency-like)
        first_numeric = df[numeric_cols[0]]
        if first_numeric.max() > 10000:  # High value transactions
            sector_scores['finance'] += 1
            sector_scores['ecommerce'] += 1
        
        # Check for manufacturing patterns (production quantities)
        if first_numeric.max() > 1000 and first_numeric.min() >= 0:  # Production quantities
            sector_scores['manufacturing'] += 1
        
        # Check for retail patterns (moderate values, many transactions)
        if 10 <= first_numeric.mean() <= 1000:  # Typical retail transaction values
            sector_scores['retail'] += 1
            sector_scores['restaurant'] += 1
    
    # Date/time analysis
    date_cols = df.select_dtypes(include=['datetime64']).columns
    if len(date_cols) > 0:
        # If we have time-based data, it could be any sector
        for sector in sector_scores:
            sector_scores[sector] += 0.5
    
    # Categorical data analysis
    if len(categorical_cols) > 0:
        # Check for product categories (retail/ecommerce)
        first_categorical = df[categorical_cols[0]]
        unique_values = first_categorical.nunique()
        if unique_values > 10:  # Many categories suggests retail/ecommerce
            sector_scores['retail'] += 1
            sector_scores['ecommerce'] += 1
    
    # Find the sector with highest score
    detected_sector = max(sector_scores, key=sector_scores.get)
    
    # If no clear winner, default to retail (most common business type)
    if sector_scores[detected_sector] == 0:
        detected_sector = 'retail'
    
    return detected_sector, sector_scores

def generate_comprehensive_insights(df: pd.DataFrame, industry: str, tier: str) -> List[Dict[str, Any]]:
    """
    Generate comprehensive business insights: Generic + Industry-Specific
    """
    insights = []
    
    numeric_cols = df.select_dtypes(include=['number']).columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    # Generic Business Insights
    if len(numeric_cols) > 0:
        # Performance insight
        max_value = df[numeric_cols[0]].max()
        min_value = df[numeric_cols[0]].min()
        insights.append({
            'title': 'Performance Analysis',
            'description': f'Your data shows a {max_value/min_value:.1f}x performance range, indicating significant variation in results.',
            'recommendation': 'Focus on standardizing processes to reduce variation and improve consistency.',
            'type': 'Generic'
        })
        
        # Growth insight
        if len(df) > 10:
            first_half = df[numeric_cols[0]].iloc[:len(df)//2].mean()
            second_half = df[numeric_cols[0]].iloc[len(df)//2:].mean()
            growth = ((second_half - first_half) / first_half) * 100
            insights.append({
                'title': 'Growth Trend',
                'description': f'Your business shows a {growth:+.1f}% growth trend over the analyzed period.',
                'recommendation': 'Maintain current strategies if positive, or implement improvement initiatives if negative.',
                'type': 'Generic'
            })
    
    if len(categorical_cols) > 0:
        # Category insight
        top_category = df[categorical_cols[0]].mode()[0] if not df[categorical_cols[0]].mode().empty else 'N/A'
        insights.append({
            'title': 'Category Performance',
            'description': f'The top-performing category is "{top_category}".',
            'recommendation': 'Leverage successful category strategies across other areas of your business.',
            'type': 'Generic'
        })
    
    # Industry-Specific Insights
    if industry == 'retail':
        if len(numeric_cols) > 0:
            avg_sales = df[numeric_cols[0]].mean()
            insights.append({
                'title': 'Retail Sales Optimization',
                'description': f'Your average transaction value is ${avg_sales:,.0f}. Consider upselling strategies to increase units per transaction.',
                'recommendation': 'Implement bundle offers and cross-selling to improve sales velocity and sell-through rates.',
                'type': 'Retail'
            })
    
    elif industry == 'ecommerce':
        if len(numeric_cols) > 0:
            total_revenue = df[numeric_cols[0]].sum()
            insights.append({
                'title': 'E-commerce Conversion Strategy',
                'description': f'With ${total_revenue:,.0f} in total revenue, focus on reducing cart abandonment and improving conversion rates.',
                'recommendation': 'Implement exit-intent popups, abandoned cart emails, and streamlined checkout processes.',
                'type': 'E-commerce'
            })
    
    elif industry == 'restaurant':
        if len(numeric_cols) > 0:
            avg_check = df[numeric_cols[0]].mean()
            insights.append({
                'title': 'Restaurant Revenue Optimization',
                'description': f'Your average check size is ${avg_check:,.0f}. Focus on increasing table turnover and check size.',
                'recommendation': 'Implement dynamic pricing, upsell premium items, and optimize table management for better turnover.',
                'type': 'Restaurant'
            })
    
    elif industry == 'manufacturing':
        if len(numeric_cols) > 0:
            avg_efficiency = df[numeric_cols[0]].mean()
            insights.append({
                'title': 'Manufacturing Efficiency',
                'description': f'Your production efficiency is {avg_efficiency:,.0f}. Focus on reducing defects and improving utilization.',
                'recommendation': 'Implement predictive maintenance, quality control systems, and lean manufacturing principles.',
                'type': 'Manufacturing'
            })
    
    elif industry == 'finance':
        if len(numeric_cols) > 0:
            avg_portfolio = df[numeric_cols[0]].mean()
            insights.append({
                'title': 'Financial Portfolio Optimization',
                'description': f'Your average portfolio value is ${avg_portfolio:,.0f}. Focus on improving ROI and reducing acquisition costs.',
                'recommendation': 'Implement automated investment strategies, risk management systems, and customer retention programs.',
                'type': 'Finance'
            })
    
    elif industry == 'healthcare':
        if len(numeric_cols) > 0:
            avg_revenue = df[numeric_cols[0]].mean()
            insights.append({
                'title': 'Healthcare Quality Improvement',
                'description': f'Your average revenue per patient is ${avg_revenue:,.0f}. Focus on patient satisfaction and reducing readmissions.',
                'recommendation': 'Implement patient engagement programs, quality monitoring systems, and preventive care initiatives.',
                'type': 'Healthcare'
            })
    
    # Tier-specific insights
    if tier == 'enterprise':
        insights.append({
            'title': 'Enterprise Strategic Recommendation',
            'description': f'Based on advanced analytics, your {industry} business has significant optimization potential.',
            'recommendation': 'Consider implementing predictive analytics, automated decision-making systems, and AI-powered insights.',
            'type': 'Enterprise'
        })
    
    return insights

@dataclass
class InsightSpec:
    """Specification for a business insight"""
    title: str
    description: str
    insight_type: str  # descriptive, diagnostic, predictive, prescriptive
    priority: str  # high, medium, low
    confidence: float
    data_evidence: Dict[str, Any]
    chart_spec: Dict[str, Any]
    business_impact: str
    recommendations: List[str]

class ContextRichInsightGenerator:
    """
    Generate business insights using clean data and rich context
    """
    
    def __init__(self):
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
        self.llm_provider = os.getenv('LLM_PROVIDER', 'openai')
        self.model = os.getenv('LLM_MODEL', 'gpt-3.5-turbo')
        
        # Chart type mapping based on data characteristics
        self.chart_type_mapping = {
            'temporal_measure': 'line',
            'categorical_measure': 'bar',
            'two_measures': 'scatter',
            'categorical_distribution': 'pie',
            'correlation_analysis': 'heatmap',
            'kpi_dashboard': 'indicator',
            'distribution_analysis': 'histogram',
            'comparison_analysis': 'bar'
        }
    
    def generate_insights(self, df: pd.DataFrame, schema: DataSchema, 
                         standardization_results: Dict[str, StandardizationResult],
                         industry_context: str = "general", 
                         user_tier: str = "pro") -> List[InsightSpec]:
        """
        Generate context-rich business insights
        """
        try:
            # Prepare rich context for LLM
            context = self._prepare_insight_context(df, schema, standardization_results, industry_context, user_tier)
            
            # Call LLM for insight generation
            llm_response = self._call_llm_for_insights(context)
            
            # Parse and validate insights
            insights = self._parse_insight_response(llm_response, df, schema)
            
            # Enhance insights with statistical analysis
            enhanced_insights = self._enhance_insights_with_statistics(insights, df, schema)
            
            # Generate chart specifications
            final_insights = self._generate_chart_specifications(enhanced_insights, df, schema)
            
            return final_insights
            
        except Exception as e:
            print(f"Error generating insights: {e}")
            return self._generate_fallback_insights(df, schema, user_tier)
    
    def _prepare_insight_context(self, df: pd.DataFrame, schema: DataSchema,
                               standardization_results: Dict[str, StandardizationResult],
                               industry_context: str, user_tier: str) -> str:
        """
        Prepare rich context for LLM insight generation
        """
        # Get data summary
        data_summary = self._get_data_summary(df, schema)
        
        # Get schema information
        schema_info = self._get_schema_info(schema, standardization_results)
        
        # Get business context
        business_context = self._get_business_context(schema, industry_context)
        
        context = f"""
You are a senior business intelligence analyst with expertise in {industry_context} industry.
I have a clean, preprocessed dataset with the following characteristics:

DATASET OVERVIEW:
- Total Rows: {data_summary['total_rows']:,}
- Total Columns: {data_summary['total_columns']}
- Data Quality Score: {data_summary['quality_score']:.2f}
- Business Domain: {business_context['domain']}

SCHEMA INFORMATION:
{json.dumps(schema_info, indent=2, cls=NumpyEncoder)}

BUSINESS CONTEXT:
- Industry: {industry_context}
- User Tier: {user_tier}
- Primary Measures: {', '.join(business_context['measures'])}
- Key Dimensions: {', '.join(business_context['dimensions'])}

DATA SAMPLE (first 5 rows):
{df.head().to_string()}

TASK: Generate {self._get_insight_count_for_tier(user_tier)} high-quality business insights.

For {user_tier} tier, focus on:
{self._get_tier_focus(user_tier)}

REQUIREMENTS:
1. Use ONLY the standardized column names from the schema
2. Provide actionable insights with clear business value
3. Include specific data evidence and statistics
4. Suggest appropriate chart types for each insight
5. Provide clear recommendations for each insight

For each insight, provide:
- Title: Clear, business-focused title
- Description: Detailed explanation with data evidence
- Type: descriptive|diagnostic|predictive|prescriptive
- Priority: high|medium|low
- Confidence: 0.0-1.0
- Data Evidence: Specific statistics and numbers
- Chart Spec: Plotly chart configuration
- Business Impact: Why this matters
- Recommendations: 2-3 actionable next steps

Return as JSON array:
[
  {{
    "title": "Insight Title",
    "description": "Detailed description with data evidence",
    "insight_type": "descriptive",
    "priority": "high",
    "confidence": 0.85,
    "data_evidence": {{"metric": "value", "statistic": "number"}},
    "chart_spec": {{"type": "bar", "x": "column", "y": "column"}},
    "business_impact": "Why this insight matters",
    "recommendations": ["action1", "action2", "action3"]
  }}
]

Focus on insights that drive business value and use the clean, standardized data effectively.
"""
        
        return context
    
    def _get_data_summary(self, df: pd.DataFrame, schema: DataSchema) -> Dict[str, Any]:
        """Get comprehensive data summary"""
        return {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'quality_score': schema.data_quality_score,
            'missing_values': df.isnull().sum().sum(),
            'duplicate_rows': df.duplicated().sum(),
            'numeric_columns': len(df.select_dtypes(include=['number']).columns),
            'categorical_columns': len(df.select_dtypes(include=['object', 'category']).columns),
            'temporal_columns': len(df.select_dtypes(include=['datetime64']).columns)
        }
    
    def _get_schema_info(self, schema: DataSchema, 
                        standardization_results: Dict[str, StandardizationResult]) -> Dict[str, Any]:
        """Get schema information for LLM context"""
        schema_info = {
            'columns': [],
            'measures': schema.suggested_measures,
            'dimensions': schema.suggested_dimensions,
            'temporal_columns': schema.temporal_columns,
            'pii_columns': schema.pii_columns
        }
        
        for profile in schema.columns:
            col_info = {
                'name': profile.name,
                'type': profile.detected_type.value,
                'intent': profile.intent.value,
                'cardinality': profile.cardinality,
                'null_percentage': profile.null_percentage,
                'sample_values': profile.sample_values[:3]
            }
            
            # Add standardization info if available
            if profile.name in standardization_results:
                result = standardization_results[profile.name]
                col_info['standardized_name'] = result.standardized_name
                col_info['suggested_format'] = result.suggested_format
            
            schema_info['columns'].append(col_info)
        
        return schema_info
    
    def _get_business_context(self, schema: DataSchema, industry_context: str) -> Dict[str, Any]:
        """Get business context information"""
        return {
            'domain': schema.business_domain,
            'industry': industry_context,
            'measures': schema.suggested_measures,
            'dimensions': schema.suggested_dimensions,
            'temporal_columns': schema.temporal_columns,
            'has_sales_data': any('sales' in col.lower() for col in schema.suggested_measures),
            'has_customer_data': any('customer' in col.lower() for col in schema.suggested_dimensions),
            'has_product_data': any('product' in col.lower() for col in schema.suggested_dimensions)
        }
    
    def _get_insight_count_for_tier(self, user_tier: str) -> int:
        """Get number of insights based on user tier"""
        tier_counts = {
            'pro': 3,
            'business': 5,
            'enterprise': 7
        }
        return tier_counts.get(user_tier, 3)
    
    def _get_tier_focus(self, user_tier: str) -> str:
        """Get focus areas for each tier"""
        tier_focus = {
            'pro': """
- Descriptive analytics: What happened?
- Basic trends and patterns
- Simple performance metrics
- Clear, actionable insights""",
            'business': """
- Diagnostic analytics: Why did it happen?
- Correlation analysis
- Comparative insights
- Root cause analysis
- Performance benchmarking""",
            'enterprise': """
- Predictive analytics: What will happen?
- Prescriptive analytics: What should we do?
- Advanced statistical analysis
- Scenario planning
- Optimization recommendations
- Risk assessment"""
        }
        return tier_focus.get(user_tier, tier_focus['pro'])
    
    def _call_llm_for_insights(self, context: str) -> str:
        """Call LLM for insight generation"""
        if self.llm_provider == 'openai' and self.openai_api_key:
            return self._call_openai_insights(context)
        elif self.llm_provider == 'anthropic' and self.anthropic_api_key:
            return self._call_anthropic_insights(context)
        else:
            return self._fallback_insights_response()
    
    def _call_openai_insights(self, context: str) -> str:
        """Call OpenAI API for insights"""
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
                        'content': 'You are a senior business intelligence analyst. Return only valid JSON for insights.'
                    },
                    {'role': 'user', 'content': context}
                ],
                'temperature': 0.3,
                'max_tokens': 3000
            }
            
            response = requests.post(
                'https://api.openai.com/v1/chat/completions',
                headers=headers,
                json=data,
                timeout=60
            )
            
            if response.status_code == 200:
                return response.json()['choices'][0]['message']['content']
            else:
                raise Exception(f"OpenAI API error: {response.status_code}")
                
        except Exception as e:
            print(f"OpenAI API error: {e}")
            return self._fallback_insights_response()
    
    def _call_anthropic_insights(self, context: str) -> str:
        """Call Anthropic Claude API for insights"""
        try:
            headers = {
                'x-api-key': self.anthropic_api_key,
                'Content-Type': 'application/json',
                'anthropic-version': '2023-06-01'
            }
            
            data = {
                'model': 'claude-3-sonnet-20240229',
                'max_tokens': 3000,
                'messages': [
                    {'role': 'user', 'content': context}
                ]
            }
            
            response = requests.post(
                'https://api.anthropic.com/v1/messages',
                headers=headers,
                json=data,
                timeout=60
            )
            
            if response.status_code == 200:
                return response.json()['content'][0]['text']
            else:
                raise Exception(f"Anthropic API error: {response.status_code}")
                
        except Exception as e:
            print(f"Anthropic API error: {e}")
            return self._fallback_insights_response()
    
    def _parse_insight_response(self, response: str, df: pd.DataFrame, 
                              schema: DataSchema) -> List[InsightSpec]:
        """Parse LLM response and create insight specifications"""
        try:
            # Extract JSON from response
            if '[' in response and ']' in response:
                start = response.find('[')
                end = response.rfind(']') + 1
                json_str = response[start:end]
                insights_data = json.loads(json_str)
                
                insights = []
                for insight_data in insights_data:
                    if isinstance(insight_data, dict):
                        insight = InsightSpec(
                            title=insight_data.get('title', 'Business Insight'),
                            description=insight_data.get('description', ''),
                            insight_type=insight_data.get('insight_type', 'descriptive'),
                            priority=insight_data.get('priority', 'medium'),
                            confidence=float(insight_data.get('confidence', 0.7)),
                            data_evidence=insight_data.get('data_evidence', {}),
                            chart_spec=insight_data.get('chart_spec', {}),
                            business_impact=insight_data.get('business_impact', ''),
                            recommendations=insight_data.get('recommendations', [])
                        )
                        insights.append(insight)
                
                return insights
            else:
                raise ValueError("No JSON array found in response")
                
        except Exception as e:
            print(f"Error parsing insight response: {e}")
            return []
    
    def _enhance_insights_with_statistics(self, insights: List[InsightSpec], 
                                        df: pd.DataFrame, schema: DataSchema) -> List[InsightSpec]:
        """Enhance insights with statistical analysis"""
        enhanced_insights = []
        
        for insight in insights:
            # Add statistical evidence
            enhanced_evidence = self._calculate_statistical_evidence(insight, df, schema)
            insight.data_evidence.update(enhanced_evidence)
            
            # Validate chart specifications
            insight.chart_spec = self._validate_chart_spec(insight.chart_spec, df, schema)
            
            enhanced_insights.append(insight)
        
        return enhanced_insights
    
    def _calculate_statistical_evidence(self, insight: InsightSpec, 
                                      df: pd.DataFrame, schema: DataSchema) -> Dict[str, Any]:
        """Calculate statistical evidence for insights"""
        evidence = {}
        
        # Extract column names from chart spec
        chart_spec = insight.chart_spec
        if 'x' in chart_spec and 'y' in chart_spec:
            x_col = chart_spec['x']
            y_col = chart_spec['y']
            
            if x_col in df.columns and y_col in df.columns:
                if chart_spec.get('type') == 'bar':
                    # Bar chart statistics
                    grouped = df.groupby(x_col)[y_col].agg(['sum', 'mean', 'count']).reset_index()
                    evidence['total_value'] = float(df[y_col].sum())
                    evidence['average_value'] = float(df[y_col].mean())
                    evidence['top_category'] = grouped.loc[grouped['sum'].idxmax(), x_col]
                    evidence['top_value'] = float(grouped['sum'].max())
                
                elif chart_spec.get('type') == 'line':
                    # Line chart statistics
                    if df[x_col].dtype in ['datetime64[ns]', 'datetime64']:
                        df_sorted = df.sort_values(x_col)
                        evidence['trend'] = 'increasing' if df_sorted[y_col].iloc[-1] > df_sorted[y_col].iloc[0] else 'decreasing'
                        evidence['growth_rate'] = float((df_sorted[y_col].iloc[-1] - df_sorted[y_col].iloc[0]) / df_sorted[y_col].iloc[0] * 100)
                        evidence['peak_value'] = float(df[y_col].max())
                        evidence['lowest_value'] = float(df[y_col].min())
                
                elif chart_spec.get('type') == 'scatter':
                    # Scatter plot statistics
                    correlation = df[x_col].corr(df[y_col])
                    evidence['correlation'] = float(correlation)
                    evidence['correlation_strength'] = 'strong' if abs(correlation) > 0.7 else 'moderate' if abs(correlation) > 0.3 else 'weak'
        
        return evidence
    
    def _validate_chart_spec(self, chart_spec: Dict[str, Any], 
                           df: pd.DataFrame, schema: DataSchema) -> Dict[str, Any]:
        """Validate and enhance chart specifications"""
        if not chart_spec:
            return self._generate_default_chart_spec(df, schema)
        
        # Ensure required fields
        if 'type' not in chart_spec:
            chart_spec['type'] = 'bar'
        
        # Validate column names exist
        for key in ['x', 'y', 'color']:
            if key in chart_spec and chart_spec[key] not in df.columns:
                # Try to find similar column
                similar_col = self._find_similar_column(chart_spec[key], df.columns)
                if similar_col:
                    chart_spec[key] = similar_col
                else:
                    del chart_spec[key]
        
        # Add default styling
        chart_spec.setdefault('title', 'Business Insight Chart')
        chart_spec.setdefault('height', 400)
        chart_spec.setdefault('showlegend', True)
        
        return chart_spec
    
    def _find_similar_column(self, target_col: str, available_cols: List[str]) -> Optional[str]:
        """Find similar column name"""
        target_lower = target_col.lower()
        for col in available_cols:
            if target_lower in col.lower() or col.lower() in target_lower:
                return col
        return None
    
    def _generate_default_chart_spec(self, df: pd.DataFrame, schema: DataSchema) -> Dict[str, Any]:
        """Generate default chart specification"""
        measures = schema.suggested_measures
        dimensions = schema.suggested_dimensions
        
        if measures and dimensions:
            return {
                'type': 'bar',
                'x': dimensions[0],
                'y': measures[0],
                'title': f'{measures[0]} by {dimensions[0]}',
                'height': 400
            }
        elif measures:
            return {
                'type': 'histogram',
                'x': measures[0],
                'title': f'Distribution of {measures[0]}',
                'height': 400
            }
        else:
            return {
                'type': 'bar',
                'x': df.columns[0],
                'title': 'Data Overview',
                'height': 400
            }
    
    def _generate_chart_specifications(self, insights: List[InsightSpec], 
                                     df: pd.DataFrame, schema: DataSchema) -> List[InsightSpec]:
        """Generate detailed chart specifications for insights"""
        for insight in insights:
            # Enhance chart spec with Plotly configuration
            insight.chart_spec = self._create_plotly_spec(insight.chart_spec, df, schema)
        
        return insights
    
    def _create_plotly_spec(self, chart_spec: Dict[str, Any], 
                          df: pd.DataFrame, schema: DataSchema) -> Dict[str, Any]:
        """Create detailed Plotly chart specification"""
        plotly_spec = {
            'chart_type': chart_spec.get('type', 'bar'),
            'title': chart_spec.get('title', 'Business Insight'),
            'height': chart_spec.get('height', 400),
            'config': {
                'displayModeBar': True,
                'displaylogo': False,
                'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d']
            }
        }
        
        # Add data configuration based on chart type
        chart_type = chart_spec.get('type', 'bar')
        
        if chart_type == 'bar' and 'x' in chart_spec and 'y' in chart_spec:
            plotly_spec.update({
                'x_axis': chart_spec['x'],
                'y_axis': chart_spec['y'],
                'orientation': 'vertical',
                'color_scheme': 'viridis'
            })
        
        elif chart_type == 'line' and 'x' in chart_spec and 'y' in chart_spec:
            plotly_spec.update({
                'x_axis': chart_spec['x'],
                'y_axis': chart_spec['y'],
                'markers': True,
                'line_smoothing': 0.3
            })
        
        elif chart_type == 'scatter' and 'x' in chart_spec and 'y' in chart_spec:
            plotly_spec.update({
                'x_axis': chart_spec['x'],
                'y_axis': chart_spec['y'],
                'trend_line': True,
                'opacity': 0.7
            })
        
        return plotly_spec
    
    def _generate_fallback_insights(self, df: pd.DataFrame, schema: DataSchema, 
                                  user_tier: str) -> List[InsightSpec]:
        """Generate fallback insights when LLM is not available"""
        insights = []
        
        # Basic data overview insight
        insights.append(InsightSpec(
            title="Data Overview",
            description=f"Dataset contains {len(df):,} rows and {len(df.columns)} columns with {schema.data_quality_score:.1%} data quality score.",
            insight_type="descriptive",
            priority="medium",
            confidence=0.9,
            data_evidence={
                "total_rows": len(df),
                "total_columns": len(df.columns),
                "quality_score": schema.data_quality_score
            },
            chart_spec={
                'type': 'bar',
                'title': 'Data Overview',
                'height': 400
            },
            business_impact="Provides baseline understanding of data volume and quality.",
            recommendations=["Review data quality issues", "Validate data completeness"]
        ))
        
        # Add measure-based insights
        if schema.suggested_measures:
            measure = schema.suggested_measures[0]
            if measure in df.columns:
                insights.append(InsightSpec(
                    title=f"{measure} Analysis",
                    description=f"Total {measure}: {df[measure].sum():,.0f}, Average: {df[measure].mean():,.2f}",
                    insight_type="descriptive",
                    priority="high",
                    confidence=0.8,
                    data_evidence={
                        "total": float(df[measure].sum()),
                        "average": float(df[measure].mean()),
                        "max": float(df[measure].max()),
                        "min": float(df[measure].min())
                    },
                    chart_spec={
                        'type': 'bar',
                        'x': measure,
                        'title': f'{measure} Distribution',
                        'height': 400
                    },
                    business_impact=f"Key performance metric for {measure} tracking.",
                    recommendations=["Monitor trends", "Set performance targets"]
                ))
        
        return insights
    
    def _fallback_insights_response(self) -> str:
        """Fallback response when LLM is not available"""
        return '[{"title": "LLM Analysis Unavailable", "description": "Using fallback analysis", "insight_type": "descriptive", "priority": "low", "confidence": 0.5, "data_evidence": {}, "chart_spec": {"type": "bar"}, "business_impact": "Limited analysis available", "recommendations": ["Check LLM configuration"]}]'
