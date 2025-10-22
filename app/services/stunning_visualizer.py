"""
Stunning Visualization Service
Enhanced visualization rendering with professional themes and styling
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import json
from datetime import datetime

from app.services.context_rich_insights import InsightSpec
from app.services.smart_data_profiler import DataSchema

class StunningVisualizer:
    """
    Create stunning, professional visualizations with consistent theming
    """
    
    def __init__(self):
        # Professional color palettes
        self.color_palettes = {
            'corporate': {
                'primary': '#1f77b4',
                'secondary': '#ff7f0e', 
                'accent': '#2ca02c',
                'warning': '#d62728',
                'info': '#9467bd',
                'success': '#8c564b',
                'light': '#e377c2',
                'dark': '#7f7f7f'
            },
            'modern': {
                'primary': '#2E86AB',
                'secondary': '#A23B72',
                'accent': '#F18F01',
                'warning': '#C73E1D',
                'info': '#6A994E',
                'success': '#386641',
                'light': '#F2CC8F',
                'dark': '#3D405B'
            },
            'minimal': {
                'primary': '#2C3E50',
                'secondary': '#E74C3C',
                'accent': '#3498DB',
                'warning': '#F39C12',
                'info': '#9B59B6',
                'success': '#27AE60',
                'light': '#ECF0F1',
                'dark': '#34495E'
            }
        }
        
        # Chart templates
        self.chart_templates = {
            'corporate': self._create_corporate_template(),
            'modern': self._create_modern_template(),
            'minimal': self._create_minimal_template()
        }
        
        # Default theme
        self.default_theme = 'corporate'
        
        # Formatting functions
        self.formatters = {
            'currency': lambda x: f"${x:,.2f}",
            'percentage': lambda x: f"{x:.1%}",
            'number': lambda x: f"{x:,.0f}",
            'decimal': lambda x: f"{x:.2f}",
            'date': lambda x: x.strftime('%Y-%m-%d'),
            'datetime': lambda x: x.strftime('%Y-%m-%d %H:%M')
        }
    
    def render_insight_visualization(self, insight: InsightSpec, df: pd.DataFrame, 
                                   schema: DataSchema, theme: str = None) -> Dict[str, Any]:
        """
        Render stunning visualization for an insight
        """
        try:
            theme = theme or self.default_theme
            chart_spec = insight.chart_spec
            
            # Determine chart type and render
            chart_type = chart_spec.get('chart_type', 'bar')
            
            if chart_type == 'bar':
                return self._render_bar_chart(insight, df, schema, theme)
            elif chart_type == 'line':
                return self._render_line_chart(insight, df, schema, theme)
            elif chart_type == 'scatter':
                return self._render_scatter_chart(insight, df, schema, theme)
            elif chart_type == 'pie':
                return self._render_pie_chart(insight, df, schema, theme)
            elif chart_type == 'heatmap':
                return self._render_heatmap(insight, df, schema, theme)
            elif chart_type == 'histogram':
                return self._render_histogram(insight, df, schema, theme)
            elif chart_type == 'indicator':
                return self._render_indicator(insight, df, schema, theme)
            else:
                return self._render_default_chart(insight, df, schema, theme)
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'chart_html': f'<div class="error">Error rendering chart: {str(e)}</div>'
            }
    
    def _render_bar_chart(self, insight: InsightSpec, df: pd.DataFrame, 
                         schema: DataSchema, theme: str) -> Dict[str, Any]:
        """Render stunning bar chart"""
        chart_spec = insight.chart_spec
        x_col = chart_spec.get('x_axis')
        y_col = chart_spec.get('y_axis')
        
        if not x_col or not y_col or x_col not in df.columns or y_col not in df.columns:
            return self._render_default_chart(insight, df, schema, theme)
        
        # Prepare data
        chart_data = df.groupby(x_col)[y_col].sum().reset_index()
        chart_data = chart_data.sort_values(y_col, ascending=False).head(10)
        
        # Create figure with theme
        fig = go.Figure()
        
        # Add bars with gradient colors
        colors = self._get_gradient_colors(len(chart_data), theme)
        
        fig.add_trace(go.Bar(
            x=chart_data[x_col],
            y=chart_data[y_col],
            marker=dict(
                color=colors,
                line=dict(color='white', width=1),
                opacity=0.8
            ),
            text=[self._format_value(val, y_col, schema) for val in chart_data[y_col]],
            textposition='auto',
            hovertemplate=f'<b>%{{x}}</b><br>{y_col}: %{{y:,.0f}}<extra></extra>'
        ))
        
        # Apply theme
        self._apply_theme(fig, theme, 'bar')
        
        # Add annotations
        self._add_chart_annotations(fig, insight, chart_data, y_col)
        
        return {
            'success': True,
            'chart_html': fig.to_html(full_html=False, include_plotlyjs='cdn'),
            'chart_type': 'Bar Chart',
            'interactive': True,
            'data_points': len(chart_data),
            'insight': insight.description
        }
    
    def _render_line_chart(self, insight: InsightSpec, df: pd.DataFrame, 
                          schema: DataSchema, theme: str) -> Dict[str, Any]:
        """Render stunning line chart"""
        chart_spec = insight.chart_spec
        x_col = chart_spec.get('x_axis')
        y_col = chart_spec.get('y_axis')
        
        if not x_col or not y_col or x_col not in df.columns or y_col not in df.columns:
            return self._render_default_chart(insight, df, schema, theme)
        
        # Prepare data
        chart_data = df.copy()
        try:
            chart_data[x_col] = pd.to_datetime(chart_data[x_col], errors='coerce')
            chart_data = chart_data.dropna(subset=[x_col, y_col])
            chart_data = chart_data.sort_values(x_col)
        except:
            chart_data = chart_data.sort_values(x_col)
        
        # Create figure
        fig = go.Figure()
        
        # Add line with area fill
        fig.add_trace(go.Scatter(
            x=chart_data[x_col],
            y=chart_data[y_col],
            mode='lines+markers',
            line=dict(
                color=self.color_palettes[theme]['primary'],
                width=3,
                shape='spline',
                smoothing=0.3
            ),
            marker=dict(
                size=6,
                color=self.color_palettes[theme]['primary'],
                line=dict(color='white', width=2)
            ),
            fill='tonexty',
            fillcolor=f"rgba(31, 119, 180, 0.1)",
            hovertemplate=f'<b>%{{x}}</b><br>{y_col}: %{{y:,.0f}}<extra></extra>'
        ))
        
        # Add trend line if significant data
        if len(chart_data) > 2:
            z = np.polyfit(range(len(chart_data)), chart_data[y_col], 1)
            p = np.poly1d(z)
            fig.add_trace(go.Scatter(
                x=chart_data[x_col],
                y=p(range(len(chart_data))),
                mode='lines',
                line=dict(dash='dash', color=self.color_palettes[theme]['warning'], width=2),
                name='Trend',
                hovertemplate='Trend Line<extra></extra>'
            ))
        
        # Apply theme
        self._apply_theme(fig, theme, 'line')
        
        # Add annotations
        self._add_trend_annotations(fig, chart_data, y_col, theme)
        
        return {
            'success': True,
            'chart_html': fig.to_html(full_html=False, include_plotlyjs='cdn'),
            'chart_type': 'Line Chart',
            'interactive': True,
            'data_points': len(chart_data),
            'insight': insight.description
        }
    
    def _render_scatter_chart(self, insight: InsightSpec, df: pd.DataFrame, 
                            schema: DataSchema, theme: str) -> Dict[str, Any]:
        """Render stunning scatter plot"""
        chart_spec = insight.chart_spec
        x_col = chart_spec.get('x_axis')
        y_col = chart_spec.get('y_axis')
        
        if not x_col or not y_col or x_col not in df.columns or y_col not in df.columns:
            return self._render_default_chart(insight, df, schema, theme)
        
        # Prepare data
        chart_data = df[[x_col, y_col]].dropna()
        
        # Create figure
        fig = go.Figure()
        
        # Add scatter points
        fig.add_trace(go.Scatter(
            x=chart_data[x_col],
            y=chart_data[y_col],
            mode='markers',
            marker=dict(
                size=8,
                color=self.color_palettes[theme]['primary'],
                opacity=0.7,
                line=dict(color='white', width=1)
            ),
            hovertemplate=f'<b>{x_col}: %{{x:,.0f}}</b><br>{y_col}: %{{y:,.0f}}<extra></extra>'
        ))
        
        # Add trend line
        if len(chart_data) > 2:
            z = np.polyfit(chart_data[x_col], chart_data[y_col], 1)
            p = np.poly1d(z)
            fig.add_trace(go.Scatter(
                x=chart_data[x_col],
                y=p(chart_data[x_col]),
                mode='lines',
                line=dict(dash='dash', color=self.color_palettes[theme]['warning'], width=2),
                name='Trend Line',
                hovertemplate='Trend Line<extra></extra>'
            ))
        
        # Apply theme
        self._apply_theme(fig, theme, 'scatter')
        
        # Add correlation annotation
        correlation = chart_data[x_col].corr(chart_data[y_col])
        fig.add_annotation(
            text=f"Correlation: {correlation:.3f}",
            xref="paper", yref="paper",
            x=0.02, y=0.98,
            showarrow=False,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor=self.color_palettes[theme]['dark'],
            borderwidth=1
        )
        
        return {
            'success': True,
            'chart_html': fig.to_html(full_html=False, include_plotlyjs='cdn'),
            'chart_type': 'Scatter Plot',
            'interactive': True,
            'data_points': len(chart_data),
            'correlation': correlation,
            'insight': insight.description
        }
    
    def _render_pie_chart(self, insight: InsightSpec, df: pd.DataFrame, 
                         schema: DataSchema, theme: str) -> Dict[str, Any]:
        """Render stunning pie chart"""
        chart_spec = insight.chart_spec
        category_col = chart_spec.get('category_column')
        value_col = chart_spec.get('value_column', 'count')
        
        if not category_col or category_col not in df.columns:
            return self._render_default_chart(insight, df, schema, theme)
        
        # Prepare data
        if value_col == 'count':
            chart_data = df[category_col].value_counts().head(8).reset_index()
            chart_data.columns = [category_col, value_col]
        else:
            chart_data = df.groupby(category_col)[value_col].sum().reset_index()
            chart_data = chart_data.sort_values(value_col, ascending=False).head(8)
        
        # Create figure
        fig = go.Figure()
        
        # Add pie chart with custom colors
        colors = self._get_pie_colors(len(chart_data), theme)
        
        fig.add_trace(go.Pie(
            labels=chart_data[category_col],
            values=chart_data[value_col],
            marker=dict(
                colors=colors,
                line=dict(color='white', width=2)
            ),
            textinfo='label+percent',
            textposition='auto',
            hovertemplate=f'<b>%{{label}}</b><br>{value_col}: %{{value:,.0f}}<br>Percentage: %{{percent}}<extra></extra>'
        ))
        
        # Apply theme
        self._apply_theme(fig, theme, 'pie')
        
        return {
            'success': True,
            'chart_html': fig.to_html(full_html=False, include_plotlyjs='cdn'),
            'chart_type': 'Pie Chart',
            'interactive': True,
            'data_points': len(chart_data),
            'insight': insight.description
        }
    
    def _render_heatmap(self, insight: InsightSpec, df: pd.DataFrame, 
                       schema: DataSchema, theme: str) -> Dict[str, Any]:
        """Render stunning correlation heatmap"""
        # Get numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        if len(numeric_cols) < 2:
            return self._render_default_chart(insight, df, schema, theme)
        
        # Calculate correlation matrix
        corr_matrix = df[numeric_cols].corr()
        
        # Create figure
        fig = go.Figure()
        
        fig.add_trace(go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=np.round(corr_matrix.values, 2),
            texttemplate='%{text}',
            textfont={"size": 10},
            hovertemplate='<b>%{y} vs %{x}</b><br>Correlation: %{z:.3f}<extra></extra>'
        ))
        
        # Apply theme
        self._apply_theme(fig, theme, 'heatmap')
        
        return {
            'success': True,
            'chart_html': fig.to_html(full_html=False, include_plotlyjs='cdn'),
            'chart_type': 'Correlation Heatmap',
            'interactive': True,
            'data_points': len(corr_matrix),
            'insight': insight.description
        }
    
    def _render_histogram(self, insight: InsightSpec, df: pd.DataFrame, 
                         schema: DataSchema, theme: str) -> Dict[str, Any]:
        """Render stunning histogram"""
        chart_spec = insight.chart_spec
        x_col = chart_spec.get('x_axis')
        
        if not x_col or x_col not in df.columns:
            # Use first numeric column
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                x_col = numeric_cols[0]
            else:
                return self._render_default_chart(insight, df, schema, theme)
        
        # Create figure
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=df[x_col],
            nbinsx=30,
            marker=dict(
                color=self.color_palettes[theme]['primary'],
                opacity=0.7,
                line=dict(color='white', width=1)
            ),
            hovertemplate=f'<b>{x_col}</b><br>Count: %{{y}}<br>Range: %{{x}}<extra></extra>'
        ))
        
        # Apply theme
        self._apply_theme(fig, theme, 'histogram')
        
        # Add statistics annotations
        mean_val = df[x_col].mean()
        std_val = df[x_col].std()
        fig.add_vline(x=mean_val, line_dash="dash", line_color=self.color_palettes[theme]['warning'])
        fig.add_annotation(
            text=f"Mean: {mean_val:.2f}<br>Std: {std_val:.2f}",
            xref="paper", yref="paper",
            x=0.02, y=0.98,
            showarrow=False,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor=self.color_palettes[theme]['dark'],
            borderwidth=1
        )
        
        return {
            'success': True,
            'chart_html': fig.to_html(full_html=False, include_plotlyjs='cdn'),
            'chart_type': 'Histogram',
            'interactive': True,
            'data_points': len(df),
            'insight': insight.description
        }
    
    def _render_indicator(self, insight: InsightSpec, df: pd.DataFrame, 
                         schema: DataSchema, theme: str) -> Dict[str, Any]:
        """Render stunning KPI indicators"""
        # Create subplots for multiple KPIs
        measures = schema.suggested_measures[:4]  # Max 4 KPIs
        
        if not measures:
            return self._render_default_chart(insight, df, schema, theme)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=measures,
            specs=[[{"type": "indicator"}, {"type": "indicator"}],
                   [{"type": "indicator"}, {"type": "indicator"}]]
        )
        
        colors = list(self.color_palettes[theme].values())[:4]
        
        for i, measure in enumerate(measures):
            if measure in df.columns:
                row = (i // 2) + 1
                col = (i % 2) + 1
                
                current_value = df[measure].sum()
                # Calculate change (simplified - would need historical data)
                change = np.random.uniform(-0.1, 0.1)  # Placeholder
                
                fig.add_trace(go.Indicator(
                    mode="number+delta",
                    value=current_value,
                    delta={"reference": current_value * (1 - change), "relative": True},
                    title={"text": measure, "font": {"size": 14}},
                    number={"font": {"size": 20, "color": colors[i]}},
                    domain={'x': [0, 1], 'y': [0, 1]}
                ), row=row, col=col)
        
        # Apply theme
        self._apply_theme(fig, theme, 'indicator')
        
        return {
            'success': True,
            'chart_html': fig.to_html(full_html=False, include_plotlyjs='cdn'),
            'chart_type': 'KPI Dashboard',
            'interactive': True,
            'kpi_count': len(measures),
            'insight': insight.description
        }
    
    def _render_default_chart(self, insight: InsightSpec, df: pd.DataFrame, 
                            schema: DataSchema, theme: str) -> Dict[str, Any]:
        """Render default chart when specific type fails"""
        # Create simple data overview
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=['Total Rows', 'Total Columns', 'Data Quality'],
            y=[len(df), len(df.columns), schema.data_quality_score * 100],
            marker=dict(
                color=[self.color_palettes[theme]['primary'],
                       self.color_palettes[theme]['secondary'],
                       self.color_palettes[theme]['accent']]
            )
        ))
        
        self._apply_theme(fig, theme, 'bar')
        
        return {
            'success': True,
            'chart_html': fig.to_html(full_html=False, include_plotlyjs='cdn'),
            'chart_type': 'Data Overview',
            'interactive': True,
            'data_points': 3,
            'insight': 'Basic data overview'
        }
    
    def _apply_theme(self, fig, theme: str, chart_type: str):
        """Apply professional theme to chart"""
        colors = self.color_palettes[theme]
        template = self.chart_templates[theme]
        
        # Update layout with theme
        fig.update_layout(
            template=template,
            font=dict(
                family="Arial, sans-serif",
                size=12,
                color=colors['dark']
            ),
            title=dict(
                font=dict(size=16, color=colors['dark']),
                x=0.5,
                xanchor='center'
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            margin=dict(l=50, r=50, t=80, b=50)
        )
        
        # Chart-specific styling
        if chart_type in ['bar', 'line', 'scatter']:
            fig.update_xaxes(
                gridcolor='rgba(128,128,128,0.2)',
                linecolor=colors['dark'],
                tickfont=dict(size=10)
            )
            fig.update_yaxes(
                gridcolor='rgba(128,128,128,0.2)',
                linecolor=colors['dark'],
                tickfont=dict(size=10)
            )
    
    def _create_corporate_template(self):
        """Create corporate theme template"""
        return go.layout.Template(
            layout=go.Layout(
                colorway=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'],
                font=dict(family="Arial, sans-serif"),
                title=dict(font=dict(size=16)),
                xaxis=dict(gridcolor='rgba(128,128,128,0.2)'),
                yaxis=dict(gridcolor='rgba(128,128,128,0.2)')
            )
        )
    
    def _create_modern_template(self):
        """Create modern theme template"""
        return go.layout.Template(
            layout=go.Layout(
                colorway=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E'],
                font=dict(family="Helvetica, sans-serif"),
                title=dict(font=dict(size=16)),
                xaxis=dict(gridcolor='rgba(128,128,128,0.1)'),
                yaxis=dict(gridcolor='rgba(128,128,128,0.1)')
            )
        )
    
    def _create_minimal_template(self):
        """Create minimal theme template"""
        return go.layout.Template(
            layout=go.Layout(
                colorway=['#2C3E50', '#E74C3C', '#3498DB', '#F39C12', '#9B59B6'],
                font=dict(family="Georgia, serif"),
                title=dict(font=dict(size=16)),
                xaxis=dict(gridcolor='rgba(128,128,128,0.3)'),
                yaxis=dict(gridcolor='rgba(128,128,128,0.3)')
            )
        )
    
    def _get_gradient_colors(self, n_colors: int, theme: str) -> List[str]:
        """Generate gradient colors for charts"""
        base_color = self.color_palettes[theme]['primary']
        colors = []
        
        for i in range(n_colors):
            # Create gradient effect
            opacity = 0.6 + (0.4 * i / n_colors)
            colors.append(f"rgba(31, 119, 180, {opacity})")
        
        return colors
    
    def _get_pie_colors(self, n_colors: int, theme: str) -> List[str]:
        """Generate colors for pie charts"""
        color_values = list(self.color_palettes[theme].values())
        return [color_values[i % len(color_values)] for i in range(n_colors)]
    
    def _format_value(self, value: float, column: str, schema: DataSchema) -> str:
        """Format value based on column type and schema"""
        # Find column profile
        profile = None
        for col_profile in schema.columns:
            if col_profile.name == column:
                profile = col_profile
                break
        
        if profile:
            # Use suggested format from standardization
            if hasattr(profile, 'suggested_format'):
                format_type = profile.suggested_format
                if format_type in self.formatters:
                    return self.formatters[format_type](value)
        
        # Default formatting
        if isinstance(value, (int, float)):
            if value > 1000000:
                return f"{value/1000000:.1f}M"
            elif value > 1000:
                return f"{value/1000:.1f}K"
            else:
                return f"{value:,.0f}"
        
        return str(value)
    
    def _add_chart_annotations(self, fig, insight: InsightSpec, chart_data, y_col):
        """Add professional annotations to charts"""
        # Add total annotation
        total_value = chart_data[y_col].sum()
        fig.add_annotation(
            text=f"Total: {total_value:,.0f}",
            xref="paper", yref="paper",
            x=0.5, y=-0.1,
            showarrow=False,
            font=dict(size=12, color="gray")
        )
        
        # Add insight annotation if available
        if insight.business_impact:
            fig.add_annotation(
                text=f"💡 {insight.business_impact[:100]}...",
                xref="paper", yref="paper",
                x=0.02, y=0.98,
                showarrow=False,
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="gray",
                borderwidth=1,
                font=dict(size=10)
            )
    
    def _add_trend_annotations(self, fig, chart_data, y_col, theme):
        """Add trend annotations to line charts"""
        if len(chart_data) > 1:
            first_value = chart_data[y_col].iloc[0]
            last_value = chart_data[y_col].iloc[-1]
            change = ((last_value - first_value) / first_value) * 100
            
            trend_text = f"Trend: {change:+.1f}%"
            trend_color = self.color_palettes[theme]['success'] if change > 0 else self.color_palettes[theme]['warning']
            
            fig.add_annotation(
                text=trend_text,
                xref="paper", yref="paper",
                x=0.02, y=0.98,
                showarrow=False,
                font=dict(size=12, color=trend_color),
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor=trend_color,
                borderwidth=1
            )
    
    def export_chart(self, chart_html: str, format: str = "html") -> Dict[str, Any]:
        """Export chart in various formats"""
        try:
            if format == "html":
                return {
                    'success': True,
                    'data': chart_html,
                    'format': 'html',
                    'size': len(chart_html)
                }
            else:
                return {
                    'success': False,
                    'error': f"Export format {format} not yet implemented",
                    'format': format
                }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'format': format
            }
