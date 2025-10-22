"""
Smart Data Contextualization Pipeline
Main integration service that orchestrates the complete pipeline
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import json
from app.utils.file_reader import robust_file_reader

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

from app.services.smart_data_profiler import SmartDataProfiler, DataSchema
from app.services.column_standardizer import LLMColumnStandardizer, StandardizationResult
from app.services.automated_preprocessor import AutomatedPreprocessor
from app.services.context_rich_insights import ContextRichInsightGenerator, InsightSpec, calculate_comprehensive_kpis, generate_comprehensive_insights, detect_industry_sector
from app.services.stunning_visualizer import StunningVisualizer

class SmartContextualizationPipeline:
    """
    Main pipeline that orchestrates the complete Smart Data Contextualization process
    """
    
    def __init__(self):
        self.profiler = SmartDataProfiler()
        self.standardizer = LLMColumnStandardizer()
        self.preprocessor = AutomatedPreprocessor()
        self.insight_generator = ContextRichInsightGenerator()
        self.visualizer = StunningVisualizer()
        
        self.pipeline_log = []
    
    def process_data(self, df: pd.DataFrame, industry_context: str = "general", 
                    user_tier: str = "pro", theme: str = "corporate") -> Dict[str, Any]:
        """
        Complete pipeline processing from raw data to stunning visualizations
        """
        try:
            pipeline_start = datetime.now()
            
            # Step 1: Smart Data Profiling
            self.pipeline_log.append("Starting Smart Data Profiling...")
            schema = self.profiler.profile_dataframe(df, industry_context)
            profiling_time = datetime.now()
            
            # Step 2: LLM-Powered Column Standardization
            self.pipeline_log.append("Starting Column Standardization...")
            standardization_results = self.standardizer.standardize_columns(
                schema.columns, industry_context
            )
            standardization_time = datetime.now()
            
            # Step 3: Automated Data Preprocessing
            self.pipeline_log.append("Starting Automated Preprocessing...")
            df_processed, preprocessing_info = self.preprocessor.preprocess_dataframe(
                df, schema, standardization_results
            )
            preprocessing_time = datetime.now()
            
            # Step 4: Context-Rich Insight Generation
            self.pipeline_log.append("Starting Insight Generation...")
            insights = self.insight_generator.generate_insights(
                df_processed, schema, standardization_results, industry_context, user_tier
            )
            insight_time = datetime.now()
            
            # Step 5: Stunning Visualization Generation
            self.pipeline_log.append("Starting Visualization Generation...")
            visualizations = []
            for insight in insights:
                viz_result = self.visualizer.render_insight_visualization(
                    insight, df_processed, schema, theme
                )
                visualizations.append({
                    'insight': insight,
                    'visualization': viz_result
                })
            visualization_time = datetime.now()
            
            # Calculate processing times
            processing_times = {
                'profiling': (profiling_time - pipeline_start).total_seconds(),
                'standardization': (standardization_time - profiling_time).total_seconds(),
                'preprocessing': (preprocessing_time - standardization_time).total_seconds(),
                'insight_generation': (insight_time - preprocessing_time).total_seconds(),
                'visualization': (visualization_time - insight_time).total_seconds(),
                'total': (visualization_time - pipeline_start).total_seconds()
            }
            
            # Generate comprehensive results
            results = {
                'status': 'success',
                'pipeline_version': '1.0.0',
                'processing_times': processing_times,
                'data_schema': schema,
                'standardization_results': standardization_results,
                'preprocessing_info': preprocessing_info,
                'insights': insights,
                'visualizations': visualizations,
                'summary': self._generate_pipeline_summary(
                    schema, standardization_results, preprocessing_info, insights
                ),
                'metadata': {
                    'processed_at': pipeline_start.isoformat(),
                    'industry_context': industry_context,
                    'user_tier': user_tier,
                    'theme': theme,
                    'original_shape': df.shape,
                    'processed_shape': df_processed.shape
                }
            }
            
            self.pipeline_log.append("Pipeline completed successfully!")
            return results
            
        except Exception as e:
            self.pipeline_log.append(f"Pipeline failed: {str(e)}")
            return {
                'status': 'error',
                'error': str(e),
                'pipeline_log': self.pipeline_log,
                'metadata': {
                    'processed_at': datetime.now().isoformat(),
                    'industry_context': industry_context,
                    'user_tier': user_tier,
                    'theme': theme
                }
            }
    
    def generate_comprehensive_analytics(self, df: pd.DataFrame, industry: str, tier: str) -> Dict[str, Any]:
        """
        Generate comprehensive analytics with both generic and industry-specific KPIs and insights
        """
        try:
            # Calculate comprehensive KPIs
            kpis = calculate_comprehensive_kpis(df, industry)
            
            # Generate comprehensive insights
            insights = generate_comprehensive_insights(df, industry, tier)
            
            # Separate generic and industry-specific
            generic_kpis = [kpi for kpi in kpis if kpi[2] == "Generic"]
            industry_kpis = [kpi for kpi in kpis if kpi[2] != "Generic"]
            
            generic_insights = [insight for insight in insights if insight.get('type') == 'Generic']
            industry_insights = [insight for insight in insights if insight.get('type') != 'Generic']
            
            return {
                'generic_kpis': generic_kpis,
                'industry_kpis': industry_kpis,
                'generic_insights': generic_insights,
                'industry_insights': industry_insights,
                'total_kpis': len(kpis),
                'total_insights': len(insights)
            }
            
        except Exception as e:
            self.pipeline_log.append(f"Error in comprehensive analytics: {str(e)}")
            return {
                'generic_kpis': [],
                'industry_kpis': [],
                'generic_insights': [],
                'industry_insights': [],
                'total_kpis': 0,
                'total_insights': 0,
                'error': str(e)
            }
    
    def auto_detect_industry(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Automatically detect the industry sector from data characteristics
        """
        try:
            detected_sector, sector_scores = detect_industry_sector(df)
            
            return {
                'detected_sector': detected_sector,
                'sector_scores': sector_scores,
                'confidence': max(sector_scores.values()) if sector_scores else 0,
                'all_sectors': list(sector_scores.keys()),
                'top_3_sectors': sorted(sector_scores.items(), key=lambda x: x[1], reverse=True)[:3]
            }
            
        except Exception as e:
            self.pipeline_log.append(f"Error in industry detection: {str(e)}")
            return {
                'detected_sector': 'retail',  # Default fallback
                'sector_scores': {},
                'confidence': 0,
                'all_sectors': ['retail', 'ecommerce', 'restaurant', 'manufacturing', 'finance', 'healthcare'],
                'top_3_sectors': [('retail', 0)],
                'error': str(e)
            }
    
    def _generate_pipeline_summary(self, schema: DataSchema, 
                                 standardization_results: Dict[str, StandardizationResult],
                                 preprocessing_info: Dict[str, Any],
                                 insights: List[InsightSpec]) -> Dict[str, Any]:
        """Generate comprehensive pipeline summary"""
        
        # Data quality improvements
        quality_improvement = preprocessing_info.get('data_quality_score', 0) - schema.data_quality_score
        
        # Standardization summary
        standardization_summary = self.standardizer.get_standardization_report(standardization_results)
        
        # Preprocessing summary
        preprocessing_summary = self.preprocessor.get_preprocessing_summary(preprocessing_info)
        
        # Insights summary
        insights_by_type = {}
        insights_by_priority = {}
        for insight in insights:
            insights_by_type[insight.insight_type] = insights_by_type.get(insight.insight_type, 0) + 1
            insights_by_priority[insight.priority] = insights_by_priority.get(insight.priority, 0) + 1
        
        return {
            'data_quality': {
                'initial_score': schema.data_quality_score,
                'final_score': preprocessing_info.get('data_quality_score', 0),
                'improvement': quality_improvement,
                'quality_rating': self._get_quality_rating(preprocessing_info.get('data_quality_score', 0))
            },
            'standardization': {
                'columns_processed': standardization_summary['total_columns'],
                'high_confidence': standardization_summary['confidence_distribution']['high'],
                'average_confidence': standardization_summary['average_confidence'],
                'categories_identified': standardization_summary['category_distribution']
            },
            'preprocessing': {
                'steps_completed': preprocessing_summary['steps_completed'],
                'columns_renamed': preprocessing_summary['columns_renamed'],
                'missing_values_treated': preprocessing_summary['missing_values_treated'],
                'outliers_treated': preprocessing_summary['outliers_treated'],
                'features_created': preprocessing_summary['features_created']
            },
            'insights': {
                'total_insights': len(insights),
                'insights_by_type': insights_by_type,
                'insights_by_priority': insights_by_priority,
                'average_confidence': np.mean([insight.confidence for insight in insights]) if insights else 0,
                'high_priority_insights': insights_by_priority.get('high', 0)
            },
            'business_value': {
                'measures_identified': len(schema.suggested_measures),
                'dimensions_identified': len(schema.suggested_dimensions),
                'temporal_columns': len(schema.temporal_columns),
                'pii_columns_detected': len(schema.pii_columns),
                'business_domain': schema.business_domain
            }
        }
    
    def _get_quality_rating(self, score: float) -> str:
        """Get quality rating based on score"""
        if score >= 0.95:
            return "Excellent"
        elif score >= 0.85:
            return "Good"
        elif score >= 0.70:
            return "Fair"
        else:
            return "Poor"
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status and configuration"""
        return {
            'pipeline_active': True,
            'components': {
                'profiler': 'SmartDataProfiler',
                'standardizer': 'LLMColumnStandardizer',
                'preprocessor': 'AutomatedPreprocessor',
                'insight_generator': 'ContextRichInsightGenerator',
                'visualizer': 'StunningVisualizer'
            },
            'supported_industries': [
                'retail', 'ecommerce', 'manufacturing', 'finance', 
                'healthcare', 'restaurant', 'general'
            ],
            'supported_tiers': ['pro', 'business', 'enterprise'],
            'supported_themes': ['corporate', 'modern', 'minimal'],
            'pipeline_log': self.pipeline_log[-10:]  # Last 10 log entries
        }
    
    def validate_input_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate input data before processing"""
        validation_results = {
            'is_valid': True,
            'warnings': [],
            'errors': [],
            'recommendations': []
        }
        
        # Check if DataFrame is empty
        if df.empty:
            validation_results['is_valid'] = False
            validation_results['errors'].append("DataFrame is empty")
            return validation_results
        
        # Check for minimum data requirements
        if len(df) < 10:
            validation_results['warnings'].append("Dataset has fewer than 10 rows - insights may be limited")
        
        if len(df.columns) < 2:
            validation_results['warnings'].append("Dataset has fewer than 2 columns - analysis options may be limited")
        
        # Check for excessive missing data
        missing_percentage = (df.isnull().sum().sum() / df.size) * 100
        if missing_percentage > 50:
            validation_results['warnings'].append(f"High missing data percentage: {missing_percentage:.1f}%")
        
        # Check for duplicate rows
        duplicate_percentage = (df.duplicated().sum() / len(df)) * 100
        if duplicate_percentage > 20:
            validation_results['warnings'].append(f"High duplicate percentage: {duplicate_percentage:.1f}%")
        
        # Recommendations
        if missing_percentage > 10:
            validation_results['recommendations'].append("Consider data cleaning to improve analysis quality")
        
        if len(df.columns) > 50:
            validation_results['recommendations'].append("Large number of columns detected - consider feature selection")
        
        if len(df) > 100000:
            validation_results['recommendations'].append("Large dataset detected - consider sampling for faster processing")
        
        return validation_results
    
    def get_sample_insights(self, industry_context: str = "general", 
                           user_tier: str = "pro") -> List[Dict[str, Any]]:
        """Get sample insights for demonstration purposes"""
        sample_insights = {
            'retail': {
                'pro': [
                    {
                        'title': 'Sales Performance Overview',
                        'description': 'Total sales across all categories with key performance indicators',
                        'insight_type': 'descriptive',
                        'priority': 'high',
                        'business_impact': 'Provides baseline understanding of sales performance'
                    },
                    {
                        'title': 'Top Performing Categories',
                        'description': 'Categories with highest sales volume and revenue contribution',
                        'insight_type': 'descriptive',
                        'priority': 'medium',
                        'business_impact': 'Identifies best-performing product categories'
                    }
                ],
                'business': [
                    {
                        'title': 'Sales Trend Analysis',
                        'description': 'Monthly sales trends with seasonal patterns and growth rates',
                        'insight_type': 'diagnostic',
                        'priority': 'high',
                        'business_impact': 'Reveals sales patterns and growth opportunities'
                    },
                    {
                        'title': 'Customer Segmentation Analysis',
                        'description': 'Customer groups based on purchasing behavior and value',
                        'insight_type': 'diagnostic',
                        'priority': 'medium',
                        'business_impact': 'Enables targeted marketing strategies'
                    }
                ],
                'enterprise': [
                    {
                        'title': 'Sales Forecasting Model',
                        'description': 'Predictive model for future sales with confidence intervals',
                        'insight_type': 'predictive',
                        'priority': 'high',
                        'business_impact': 'Enables proactive business planning and resource allocation'
                    },
                    {
                        'title': 'Optimization Recommendations',
                        'description': 'AI-driven recommendations for inventory and pricing optimization',
                        'insight_type': 'prescriptive',
                        'priority': 'high',
                        'business_impact': 'Provides actionable strategies for revenue optimization'
                    }
                ]
            }
        }
        
        return sample_insights.get(industry_context, {}).get(user_tier, [])
    
    def export_results(self, results: Dict[str, Any], format: str = "json") -> Dict[str, Any]:
        """Export pipeline results in various formats"""
        try:
            if format == "json":
                return {
                    'success': True,
                    'data': json.dumps(results, indent=2, cls=NumpyEncoder),
                    'format': 'json',
                    'size': len(json.dumps(results, cls=NumpyEncoder))
                }
            elif format == "summary":
                # Export only summary information
                summary_data = {
                    'pipeline_summary': results.get('summary', {}),
                    'metadata': results.get('metadata', {}),
                    'processing_times': results.get('processing_times', {})
                }
                return {
                    'success': True,
                    'data': json.dumps(summary_data, indent=2, cls=NumpyEncoder),
                    'format': 'summary',
                    'size': len(json.dumps(summary_data, cls=NumpyEncoder))
                }
            else:
                return {
                    'success': False,
                    'error': f"Unsupported export format: {format}",
                    'format': format
                }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'format': format
            }

# Global pipeline instance
smart_pipeline = SmartContextualizationPipeline()
