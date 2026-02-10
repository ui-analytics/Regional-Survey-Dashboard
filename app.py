"""
Charlotte Regional Well-Being Dashboard
A comprehensive interactive visualization tool for exploring survey data
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from scipy import stats

# Page configuration
st.set_page_config(
    page_title="State of the Region Survey - Well-Being Dashboard",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
    }
    h1 {
        color: #A49665;
        padding-bottom: 20px;
    }
    h2 {
        color: #899064;
        padding-top: 20px;
    }
    h3 {
        color: #899064;
    }
    .highlight-box {
        background-color: #4c4e4f;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Color palettes based on best practices
COLORS = {
    'primary': '#1f77b4',
    'secondary': '#ff7f0e',
    'success': '#2ca02c',
    'warning': '#d62728',
    'info': '#9467bd',
    'sequential': px.colors.sequential.Blues,
    'diverging': px.colors.diverging.RdYlGn,
    'categorical': px.colors.qualitative.Set2,
    'wellbeing': ['#d73027', '#f46d43', '#fdae61', '#fee090', '#e0f3f8', '#abd9e9', '#74add1', '#4575b4'],
}

# Load data with caching
@st.cache_data
def load_data():
    """Load and prepare the survey data"""
    df = pd.read_csv('charlotte_regional_survey_simulated_n5000.csv')
    
    # Create labeled versions of categorical variables
    df['age_group'] = df['D3'].map({
        1: '18-24', 2: '25-39', 3: '40-54', 4: '55-65', 5: '65+'
    })
    
    df['race_ethnicity'] = df['D4'].map({
        1: 'White', 2: 'Hispanic/Latino', 3: 'Black/African American', 
        4: 'Asian', 5: 'Other'
    })
    
    df['income_bracket'] = df['D2'].map({
        1: '<$15K', 2: '$15-25K', 3: '$25-35K', 4: '$35-50K',
        5: '$50-75K', 6: '$75-100K', 7: '$100-150K', 8: '$150K+',
        -99: 'Prefer not to answer', -98: "Don't know"
    })
    
    df['employment_status'] = df['Q65'].map({
        1: 'Full-time', 2: 'Part-time', 3: 'Unemployed', 4: 'Retired'
    })
    
    df['belonging_level'] = df['BELONGCOM'].map({
        1: 'Very Weak', 2: 'Somewhat Weak', 3: 'Somewhat Strong', 
        4: 'Very Strong', 5: 'No Opinion'
    })
    
    df['outlook'] = df['ReOUTLOOK'].map({
        1: 'Better', 2: 'Worse', 3: 'Same', 4: "Don't Know"
    })
    
    df['financial_change'] = df['Q7'].map({
        1: 'Better off', 2: 'Worse off', 3: 'About the same', 4: "Don't know"
    })
    
    # Create income numeric for analysis (excluding special codes)
    df['income_numeric'] = df['D2'].apply(lambda x: x if x > 0 else np.nan)
    
    return df

# Main dashboard
def main():
    # Load data
    df = load_data()
    
    # Sidebar for navigation and filters
    st.sidebar.title("🏙️ Charlotte Regional Well-Being")
    st.sidebar.markdown("---")
    
    page = st.sidebar.radio(
        "Modules",
        ["📊 Overview", "🏘️ Geographic Analysis", "👥 Demographics Deep Dive", 
         "😊 Well-Being Explorer", "🏠 Regional Issues", "📈 Correlations & Insights"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔍 Filters")
    
    # Global filters
    counties = ['All'] + sorted(df['COUNTY'].unique().tolist())
    selected_county = st.sidebar.selectbox("County", counties)
    
    age_groups = ['All'] + sorted(df['age_group'].unique().tolist())
    selected_age = st.sidebar.selectbox("Age Group", age_groups)
    
    races = ['All'] + sorted(df['race_ethnicity'].unique().tolist())
    selected_race = st.sidebar.selectbox("Race/Ethnicity", races)
    
    # Apply filters
    filtered_df = df.copy()
    if selected_county != 'All':
        filtered_df = filtered_df[filtered_df['COUNTY'] == selected_county]
    if selected_age != 'All':
        filtered_df = filtered_df[filtered_df['age_group'] == selected_age]
    if selected_race != 'All':
        filtered_df = filtered_df[filtered_df['race_ethnicity'] == selected_race]
    
    st.sidebar.markdown("---")
    st.sidebar.info(f"**Showing {len(filtered_df):,} of {len(df):,} responses**")
    
    # Page routing
    if page == "📊 Overview":
        show_overview(filtered_df, df)
    elif page == "🏘️ Geographic Analysis":
        show_geographic_analysis(filtered_df, df)
    elif page == "👥 Demographics Deep Dive":
        show_demographics(filtered_df, df)
    elif page == "😊 Well-Being Explorer":
        show_wellbeing(filtered_df, df)
    elif page == "🏠 Regional Issues":
        show_regional_issues(filtered_df, df)
    else:
        show_correlations(filtered_df, df)

def show_overview(filtered_df, full_df):
    """Overview page with key metrics and summary visualizations"""
    st.title("📊 Charlotte Regional Well-Being Dashboard")
    st.markdown("### Survey Overview: 5,000 Residents Across 14 Counties (Read the State of the Region Report)")
    
    # Key metrics row
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "Avg Life Satisfaction",
            f"{filtered_df['LIFESAS'].mean():.2f}",
            f"{filtered_df['LIFESAS'].mean() - full_df['LIFESAS'].mean():.2f}",
            delta_color="normal"
        )
    
    with col2:
        st.metric(
            "Avg Trust Level",
            f"{filtered_df['TRUST'].mean():.2f}",
            f"{filtered_df['TRUST'].mean() - full_df['TRUST'].mean():.2f}"
        )
    
    with col3:
        strong_belonging = (filtered_df['BELONGCOM'] >= 3).mean() * 100
        st.metric(
            "Strong Belonging",
            f"{strong_belonging:.1f}%",
            f"{strong_belonging - (full_df['BELONGCOM'] >= 3).mean() * 100:.1f}%"
        )
    
    with col4:
        optimistic = (filtered_df['ReOUTLOOK'] == 1).mean() * 100
        st.metric(
            "Optimistic Outlook",
            f"{optimistic:.1f}%",
            f"{optimistic - (full_df['ReOUTLOOK'] == 1).mean() * 100:.1f}%"
        )
    
    with col5:
        st.metric(
            "Sample Size",
            f"{len(filtered_df):,}",
            f"{len(filtered_df) - len(full_df):,}"
        )
    
    st.markdown("---")
    
    # Two-column layout for main visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Life Satisfaction Distribution", 
                   help="This histogram shows how life satisfaction scores (0-10 scale) are distributed across respondents. The red dashed line indicates the average score. Higher bars on the right indicate more people with higher satisfaction. A score of 0 means 'not at all satisfied' while 10 means 'completely satisfied'.")
        
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=filtered_df['LIFESAS'],
            nbinsx=11,
            marker_color=COLORS['primary'],
            marker_line_color='white',
            marker_line_width=2,
            name='Life Satisfaction',
            hovertemplate='Score: %{x}<br>Count: %{y}<extra></extra>'
        ))
        fig.add_vline(
            x=filtered_df['LIFESAS'].mean(),
            line_dash="dash",
            line_color=COLORS['warning'],
            annotation_text=f"Mean: {filtered_df['LIFESAS'].mean():.2f}"
        )
        fig.update_layout(
            xaxis_title="Life Satisfaction Score (0-10)",
            yaxis_title="Number of Respondents",
            plot_bgcolor='white',
            showlegend=False,
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### Sample Distribution by County",
                   help="This horizontal bar chart shows the number of survey respondents from each county. Darker blue indicates more respondents. The numbers at the end of each bar show the exact count. Counties are sorted from most to least respondents.")
        
        county_counts = filtered_df['COUNTY'].value_counts().reset_index()
        county_counts.columns = ['County', 'Count']
        
        fig = px.bar(
            county_counts.sort_values('Count', ascending=True),
            x='Count',
            y='County',
            orientation='h',
            color='Count',
            color_continuous_scale='Blues',
            text='Count'
        )
        fig.update_traces(textposition='outside')
        fig.update_layout(
            xaxis_title="Number of Respondents",
            yaxis_title="",
            plot_bgcolor='white',
            showlegend=False,
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Well-being quadrant
    st.markdown("---")
    st.markdown("#### Well-Being Dimensions: Life Satisfaction vs. Social Support",
               help="This scatter plot shows the relationship between life satisfaction (vertical axis) and social support (horizontal axis). Each dot represents a respondent. The color indicates trust level (darker = higher trust). Dot size represents worthwhileness. The dotted lines divide the chart into four quadrants: Thriving (high on both), Satisfied but Isolated (high satisfaction, low support), Connected but Struggling (low satisfaction, high support), and Vulnerable (low on both).")
    
    fig = px.scatter(
        filtered_df,
        x='BELONGNEED',
        y='LIFESAS',
        color='TRUST',
        size='LIFEWW',
        hover_data=['COUNTY', 'age_group', 'income_bracket'],
        color_continuous_scale='Viridis',
        labels={
            'BELONGNEED': 'Social Support (0-10)',
            'LIFESAS': 'Life Satisfaction (0-10)',
            'TRUST': 'Trust Level',
            'LIFEWW': 'Worthwhileness'
        },
        opacity=0.6
    )
    
    # Add quadrant lines
    mid_x = filtered_df['BELONGNEED'].median()
    mid_y = filtered_df['LIFESAS'].median()
    
    fig.add_hline(y=mid_y, line_dash="dot", line_color="gray", opacity=0.5)
    fig.add_vline(x=mid_x, line_dash="dot", line_color="gray", opacity=0.5)
    
    fig.update_layout(
        plot_bgcolor='white',
        height=500,
        annotations=[
            dict(x=mid_x+1.5, y=mid_y+1.5, text="Thriving", showarrow=False, font=dict(size=14, color='green')),
            dict(x=mid_x-1.5, y=mid_y+1.5, text="Satisfied but Isolated", showarrow=False, font=dict(size=14, color='orange')),
            dict(x=mid_x+1.5, y=mid_y-1.5, text="Connected but Struggling", showarrow=False, font=dict(size=14, color='orange')),
            dict(x=mid_x-1.5, y=mid_y-1.5, text="Vulnerable", showarrow=False, font=dict(size=14, color='red'))
        ]
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Demographics summary
    st.markdown("---")
    st.markdown("#### Sample Demographics",
               help="These pie charts show the demographic composition of survey respondents. Each slice represents a different group, with percentages showing the proportion of the total sample. Hover over slices to see exact counts and percentages.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age_dist = filtered_df['age_group'].value_counts()
        fig = px.pie(
            values=age_dist.values,
            names=age_dist.index,
            title="Age Distribution",
            color_discrete_sequence=COLORS['categorical']
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        race_dist = filtered_df['race_ethnicity'].value_counts()
        fig = px.pie(
            values=race_dist.values,
            names=race_dist.index,
            title="Race/Ethnicity Distribution",
            color_discrete_sequence=COLORS['categorical']
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        emp_dist = filtered_df['employment_status'].value_counts()
        fig = px.pie(
            values=emp_dist.values,
            names=emp_dist.index,
            title="Employment Status",
            color_discrete_sequence=COLORS['categorical']
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

def show_geographic_analysis(filtered_df, full_df):
    """Geographic analysis page"""
    st.title("🏘️ Geographic Analysis")
    st.markdown("### Regional Patterns Across the Charlotte Region")
    
    # County-level metrics selection
    metric_options = {
        'Life Satisfaction': 'LIFESAS',
        'Worthwhileness': 'LIFEWW',
        'Trust': 'TRUST',
        'Social Support': 'BELONGNEED',
        'Housing Concern': 'HOUSING1',
        'Community Belonging': 'BELONGCOM'
    }
    
    selected_metric = st.selectbox("Select Metric to Analyze", list(metric_options.keys()))
    metric_col = metric_options[selected_metric]
    
    # County comparison
    county_stats = filtered_df.groupby('COUNTY').agg({
        metric_col: ['mean', 'std', 'count'],
        'respondent_id': 'count'
    }).reset_index()
    county_stats.columns = ['County', 'Mean', 'Std Dev', 'Count', 'N']
    county_stats = county_stats.sort_values('Mean', ascending=False)
    
    # Add confidence intervals
    county_stats['CI_lower'] = county_stats['Mean'] - 1.96 * (county_stats['Std Dev'] / np.sqrt(county_stats['Count']))
    county_stats['CI_upper'] = county_stats['Mean'] + 1.96 * (county_stats['Std Dev'] / np.sqrt(county_stats['Count']))
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(f"#### {selected_metric} by County (with 95% CI)",
                   help=f"This horizontal bar chart ranks counties by average {selected_metric.lower()}. Colors indicate the score level (red=low, yellow=medium, green=high). Error bars show the 95% confidence interval - the true county average is likely within this range. Wider bars indicate more uncertainty. Numbers at the end show the exact mean score. Hover to see the sample size for each county.")
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=county_stats['Mean'],
            y=county_stats['County'],
            orientation='h',
            marker=dict(
                color=county_stats['Mean'],
                colorscale='RdYlGn',
                showscale=True,
                colorbar=dict(title=selected_metric)
            ),
            text=county_stats['Mean'].round(2),
            textposition='outside',
            error_x=dict(
                type='data',
                symmetric=False,
                array=county_stats['CI_upper'] - county_stats['Mean'],
                arrayminus=county_stats['Mean'] - county_stats['CI_lower'],
                color='rgba(0,0,0,0.3)',
                thickness=1.5
            ),
            hovertemplate='<b>%{y}</b><br>Mean: %{x:.2f}<br>N: %{customdata}<extra></extra>',
            customdata=county_stats['N']
        ))
        
        fig.update_layout(
            xaxis_title=selected_metric,
            yaxis_title="",
            plot_bgcolor='white',
            height=500,
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### County Rankings")
        st.dataframe(
            county_stats[['County', 'Mean', 'N']].style.background_gradient(
                subset=['Mean'],
                cmap='RdYlGn',
                vmin=county_stats['Mean'].min(),
                vmax=county_stats['Mean'].max()
            ).format({'Mean': '{:.2f}'}),
            height=500,
            use_container_width=True
        )
    
    # Multi-metric heatmap
    st.markdown("---")
    st.markdown("#### County Performance Across Multiple Metrics",
               help="This heatmap shows how each county performs across six different well-being metrics. Colors indicate relative performance (red=below average, yellow=average, green=above average). Each metric is normalized to a 0-1 scale for comparison. Look for patterns - do some counties excel across the board? Do others struggle on multiple dimensions?")
    
    metrics_to_plot = ['LIFESAS', 'LIFEWW', 'TRUST', 'BELONGNEED', 'HOUSING1', 'BELONGCOM']
    metric_names = ['Life Sat.', 'Worthwhile', 'Trust', 'Support', 'Housing', 'Belonging']
    
    county_multi = filtered_df.groupby('COUNTY')[metrics_to_plot].mean()
    
    # Normalize to 0-1 scale for comparison (reverse housing concern)
    county_multi_norm = county_multi.copy()
    for col in county_multi.columns:
        if col == 'HOUSING1':
            county_multi_norm[col] = 1 - (county_multi[col] - county_multi[col].min()) / (county_multi[col].max() - county_multi[col].min())
        else:
            county_multi_norm[col] = (county_multi[col] - county_multi[col].min()) / (county_multi[col].max() - county_multi[col].min())
    
    fig = px.imshow(
        county_multi_norm.T,
        labels=dict(x="County", y="Metric", color="Normalized Score"),
        x=county_multi_norm.index,
        y=metric_names,
        color_continuous_scale='RdYlGn',
        aspect='auto'
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Zip code analysis - TREEMAP FIXED
    if 'D1' in filtered_df.columns and len(filtered_df['D1'].unique()) > 1:
        st.markdown("---")
        st.markdown("#### Zip Code Level Analysis",
                   help="This treemap shows the top 20 zip codes by life satisfaction. The size of each rectangle represents the number of respondents from that zip code. The color indicates the average life satisfaction score (red=low, yellow=medium, green=high). Larger, greener rectangles indicate zip codes with many respondents and high satisfaction. Only zip codes with at least 10 responses are included for reliability.")
        
        zip_stats = filtered_df.groupby('D1').agg({
            'LIFESAS': 'mean',
            'respondent_id': 'count'
        }).reset_index()
        zip_stats.columns = ['Zip Code', 'Mean Life Satisfaction', 'Count']
        zip_stats = zip_stats[zip_stats['Count'] >= 10]
        
        # FIXED: Sort by Mean Life Satisfaction in descending order BEFORE taking top 20
        zip_stats = zip_stats.sort_values('Mean Life Satisfaction', ascending=False).head(20)
        
        # Add labels for display
        zip_stats['Label'] = zip_stats.apply(
            lambda row: f"{int(row['Zip Code'])}<br>{row['Mean Life Satisfaction']:.2f}", 
            axis=1
        )
        
        fig = px.treemap(
            zip_stats,
            path=['Label'],
            values='Count',
            color='Mean Life Satisfaction',
            color_continuous_scale='RdYlGn',
            hover_data={
                'Zip Code': True,
                'Mean Life Satisfaction': ':.2f',
                'Count': True,
                'Label': False
            },
            labels={
                'Mean Life Satisfaction': 'Avg Life Satisfaction',
                'Count': 'Sample Size'
            }
        )
        
        fig.update_traces(
            textposition='middle center',
            textfont=dict(size=11, color='white', family='Arial Black'),
            marker=dict(line=dict(width=2, color='white'))
        )
        
        fig.update_layout(
            title="Top 20 Zip Codes by Life Satisfaction (min. 10 responses, sorted highest to lowest)",
            height=500,
            margin=dict(t=50, l=0, r=0, b=0)
        )
        
        st.plotly_chart(fig, use_container_width=True)

def show_demographics(filtered_df, full_df):
    """Demographics deep dive page"""
    st.title("👥 Demographics Deep Dive")
    st.markdown("### Understanding Well-Being Across Demographic Groups")
    
    demo_dimension = st.selectbox(
        "Select Demographic Dimension",
        ["Age Group", "Race/Ethnicity", "Income", "Employment Status"]
    )
    
    dimension_map = {
        "Age Group": 'age_group',
        "Race/Ethnicity": 'race_ethnicity',
        "Income": 'income_bracket',
        "Employment Status": 'employment_status'
    }
    
    demo_col = dimension_map[demo_dimension]
    
    st.markdown(f"#### Well-Being Metrics by {demo_dimension}",
               help=f"This grouped bar chart compares average well-being scores across {demo_dimension.lower()} groups. Each group of bars represents one demographic category. The four colored bars show Life Satisfaction, Worthwhileness, Trust, and Social Support. Higher bars indicate better outcomes. Look for patterns - which groups score consistently higher or lower across metrics?")
    
    wellbeing_metrics = ['LIFESAS', 'LIFEWW', 'TRUST', 'BELONGNEED']
    metric_names = ['Life Satisfaction', 'Worthwhileness', 'Trust', 'Social Support']
    
    demo_stats = filtered_df.groupby(demo_col)[wellbeing_metrics].mean().reset_index()
    
    demo_melted = demo_stats.melt(id_vars=demo_col, var_name='Metric', value_name='Score')
    demo_melted['Metric'] = demo_melted['Metric'].map(dict(zip(wellbeing_metrics, metric_names)))
    
    fig = px.bar(
        demo_melted,
        x=demo_col,
        y='Score',
        color='Metric',
        barmode='group',
        color_discrete_sequence=COLORS['categorical'],
        title=f"Well-Being Scores by {demo_dimension}"
    )
    fig.update_layout(
        xaxis_title=demo_dimension,
        yaxis_title="Mean Score (0-10)",
        plot_bgcolor='white',
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    st.markdown(f"#### Life Satisfaction Distribution by {demo_dimension}",
               help=f"These violin plots show the full distribution of life satisfaction scores for each {demo_dimension.lower()} group. The width of each violin shows where most people's scores fall (wider = more people). The white box inside shows the middle 50% of scores. The white dot shows the average. This gives you a more complete picture than just averages - you can see if a group has very spread out scores or if they're tightly clustered.")
    
    fig = go.Figure()
    
    for group in sorted(filtered_df[demo_col].dropna().unique()):
        group_data = filtered_df[filtered_df[demo_col] == group]['LIFESAS']
        fig.add_trace(go.Violin(
            y=group_data,
            name=str(group),
            box_visible=True,
            meanline_visible=True,
            fillcolor=COLORS['categorical'][hash(str(group)) % len(COLORS['categorical'])],
            opacity=0.6
        ))
    
    fig.update_layout(
        yaxis_title="Life Satisfaction Score",
        xaxis_title=demo_dimension,
        plot_bgcolor='white',
        height=450,
        showlegend=True
    )
    st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Group Statistics")
        group_stats = filtered_df.groupby(demo_col).agg({
            'LIFESAS': ['mean', 'std', 'count']
        }).round(2)
        group_stats.columns = ['Mean', 'Std Dev', 'N']
        st.dataframe(group_stats, use_container_width=True)
    
    with col2:
        st.markdown("#### Key Insights")
        
        top_group = demo_stats.loc[demo_stats['LIFESAS'].idxmax(), demo_col]
        top_score = demo_stats['LIFESAS'].max()
        bottom_group = demo_stats.loc[demo_stats['LIFESAS'].idxmin(), demo_col]
        bottom_score = demo_stats['LIFESAS'].min()
        gap = top_score - bottom_score
        
        st.markdown(f"""
        <div class="highlight-box">
        <strong>Highest Life Satisfaction:</strong><br>
        {top_group}: {top_score:.2f}/10<br><br>
        <strong>Lowest Life Satisfaction:</strong><br>
        {bottom_group}: {bottom_score:.2f}/10<br><br>
        <strong>Gap:</strong> {gap:.2f} points
        </div>
        """, unsafe_allow_html=True)
    
    if demo_dimension != "Income":
        st.markdown("---")
        st.markdown(f"#### {demo_dimension} × Income: Life Satisfaction Heatmap",
                   help=f"This heatmap shows how life satisfaction varies by both {demo_dimension.lower()} AND income level. Each cell shows the average satisfaction for that combination (e.g., 25-39 year olds earning $50-75K). Red cells indicate lower satisfaction, yellow is medium, and green is higher. Look for patterns - does income matter equally for all groups? Do some groups have high satisfaction even at lower incomes?")
        
        valid_income = filtered_df[filtered_df['income_numeric'].notna()].copy()
        
        if len(valid_income) > 0:
            pivot = valid_income.pivot_table(
                values='LIFESAS',
                index=demo_col,
                columns='income_bracket',
                aggfunc='mean'
            )
            
            income_order = ['<$15K', '$15-25K', '$25-35K', '$35-50K', '$50-75K', '$75-100K', '$100-150K', '$150K+']
            pivot = pivot[[col for col in income_order if col in pivot.columns]]
            
            fig = px.imshow(
                pivot,
                labels=dict(x="Income Bracket", y=demo_dimension, color="Life Satisfaction"),
                color_continuous_scale='RdYlGn',
                aspect='auto',
                text_auto='.2f'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

def show_wellbeing(filtered_df, full_df):
    """Well-being explorer page"""
    st.title("😊 Well-Being Explorer")
    st.markdown("### Deep Dive into Quality of Life Indicators")
    
    col1, col2, col3, col4 = st.columns(4)
    
    wellbeing_vars = {
        col1: ('LIFESAS', 'Life Satisfaction', '😊'),
        col2: ('LIFEWW', 'Worthwhileness', '🎯'),
        col3: ('TRUST', 'Trust in Others', '🤝'),
        col4: ('BELONGNEED', 'Social Support', '❤️')
    }
    
    for col, (var, label, emoji) in wellbeing_vars.items():
        with col:
            mean_val = filtered_df[var].mean()
            median_val = filtered_df[var].median()
            st.metric(f"{emoji} {label}", f"{mean_val:.2f}", f"Median: {median_val:.1f}")
    
    st.markdown("---")
    st.markdown("#### Well-Being Score Distributions",
               help="These four histograms show how different well-being metrics are distributed across all respondents. Each histogram shows the count of people at each score level (0-10). The red dashed line shows the average score for each metric. Compare the shapes - are people clustered around certain scores, or spread out? Do different metrics have different patterns?")
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['Life Satisfaction', 'Worthwhileness', 'Trust', 'Social Support'],
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    metrics = ['LIFESAS', 'LIFEWW', 'TRUST', 'BELONGNEED']
    positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
    
    for metric, (row, col) in zip(metrics, positions):
        fig.add_trace(
            go.Histogram(
                x=filtered_df[metric],
                nbinsx=11,
                marker_color=COLORS['primary'],
                marker_line_color='white',
                marker_line_width=1,
                showlegend=False,
                name=metric
            ),
            row=row, col=col
        )
        
        fig.add_vline(
            x=filtered_df[metric].mean(),
            line_dash="dash",
            line_color=COLORS['warning'],
            row=row, col=col
        )
    
    fig.update_xaxes(title_text="Score (0-10)", row=2, col=1)
    fig.update_xaxes(title_text="Score (0-10)", row=2, col=2)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=2, col=1)
    
    fig.update_layout(height=600, plot_bgcolor='white', showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    st.markdown("#### Sense of Community Belonging",
               help="The left chart shows how many people report each level of community belonging (Very Weak to Very Strong). Colors match the strength (red=weak, blue=strong). The right chart compares belonging levels between urban/suburban and rural areas, showing percentages as stacked bars. This helps identify if certain area types have stronger or weaker community ties.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        belonging_dist = filtered_df['belonging_level'].value_counts()
        belonging_order = ['Very Weak', 'Somewhat Weak', 'Somewhat Strong', 'Very Strong', 'No Opinion']
        belonging_dist = belonging_dist.reindex(belonging_order, fill_value=0)
        
        colors_belonging = ['#d73027', '#fc8d59', '#91bfdb', '#4575b4', '#cccccc']
        
        fig = go.Figure(data=[go.Bar(
            x=belonging_order,
            y=belonging_dist.values,
            marker_color=colors_belonging,
            text=belonging_dist.values,
            textposition='outside'
        )])
        
        fig.update_layout(
            title="Distribution of Belonging Levels",
            xaxis_title="Belonging Level",
            yaxis_title="Number of Respondents",
            plot_bgcolor='white',
            showlegend=False,
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        urban_counties = ['Mecklenburg', 'Gaston', 'Cabarrus']
        filtered_df['area_type'] = filtered_df['COUNTY'].apply(
            lambda x: 'Urban/Suburban' if x in urban_counties else 'Rural/Small Town'
        )
        
        belonging_by_area = filtered_df.groupby(['area_type', 'belonging_level']).size().unstack(fill_value=0)
        belonging_by_area_pct = belonging_by_area.div(belonging_by_area.sum(axis=1), axis=0) * 100
        
        fig = go.Figure()
        for level in belonging_order:
            if level in belonging_by_area_pct.columns:
                fig.add_trace(go.Bar(
                    name=level,
                    x=belonging_by_area_pct.index,
                    y=belonging_by_area_pct[level],
                    marker_color=colors_belonging[belonging_order.index(level)]
                ))
        
        fig.update_layout(
            title="Belonging by Area Type (%)",
            xaxis_title="Area Type",
            yaxis_title="Percentage",
            barmode='stack',
            plot_bgcolor='white',
            height=400,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    st.markdown("#### Factors Associated with Life Satisfaction",
               help="This scatter plot shows the relationship between the selected factor and life satisfaction. Each dot represents one respondent. The blue line shows the overall trend (trendline). The correlation coefficient (r) measures the strength of the relationship from -1 to +1. Positive r means as the factor increases, satisfaction tends to increase. Higher absolute values (closer to 1 or -1) indicate stronger relationships.")
    
    factor_options = {
        'Income': 'income_numeric',
        'Age Group': 'D3',
        'Trust': 'TRUST',
        'Social Support': 'BELONGNEED',
        'Emergency Confidence': 'Q8',
        'Safety (Day)': 'Q11_day',
        'Housing Concern': 'HOUSING1'
    }
    
    selected_factor = st.selectbox("Select Factor to Compare", list(factor_options.keys()))
    factor_var = factor_options[selected_factor]
    
    valid_data = filtered_df[[factor_var, 'LIFESAS']].dropna()
    
    if len(valid_data) > 0:
        fig = px.scatter(
            valid_data,
            x=factor_var,
            y='LIFESAS',
            trendline="ols",
            opacity=0.4,
            labels={factor_var: selected_factor, 'LIFESAS': 'Life Satisfaction'},
            color_discrete_sequence=[COLORS['primary']]
        )
        
        corr = valid_data[factor_var].corr(valid_data['LIFESAS'])
        
        fig.update_layout(
            title=f"Life Satisfaction vs. {selected_factor} (r = {corr:.3f})",
            plot_bgcolor='white',
            height=450
        )
        st.plotly_chart(fig, use_container_width=True)
        
        if abs(corr) > 0.5:
            strength = "strong"
        elif abs(corr) > 0.3:
            strength = "moderate"
        else:
            strength = "weak"
        
        direction = "positive" if corr > 0 else "negative"
        
        st.info(f"📊 There is a **{strength} {direction}** correlation (r = {corr:.3f}) between {selected_factor} and Life Satisfaction.")

def show_regional_issues(filtered_df, full_df):
    """Regional issues and priorities page"""
    st.title("🏠 Regional Issues & Priorities")
    st.markdown("### What Matters Most to Charlotte Area Residents")
    
    st.markdown("#### Top Regional Priorities",
               help="This horizontal bar chart shows the percentage of respondents who mentioned each issue as one of their top 3 priorities for the region. Longer bars (darker red) indicate more people are concerned about that issue. Since respondents could choose 3 priorities, percentages may sum to more than 100%.")
    
    priority_cols = ['REGION_1', 'REGION_2', 'REGION_3']
    priorities_data = []
    
    for col in priority_cols:
        priorities_data.extend(filtered_df[col].dropna().tolist())
    
    priority_map = {
        1: 'Housing Affordability',
        2: 'Crime/Safety',
        3: 'Economic Opportunity',
        4: 'Public Education',
        5: 'Transportation',
        6: 'Racial Equity',
        7: 'Public Health',
        8: 'Climate/Environment',
        9: 'Taxes'
    }
    
    priority_counts = pd.Series(priorities_data).value_counts()
    priority_counts.index = priority_counts.index.map(priority_map)
    priority_pct = (priority_counts / len(filtered_df) * 100).sort_values(ascending=True)
    
    fig = go.Figure(go.Bar(
        x=priority_pct.values,
        y=priority_pct.index,
        orientation='h',
        marker=dict(
            color=priority_pct.values,
            colorscale='Reds',
            showscale=False
        ),
        text=[f"{val:.1f}%" for val in priority_pct.values],
        textposition='outside'
    ))
    
    fig.update_layout(
        title="Percentage of Respondents Mentioning Each Issue",
        xaxis_title="Percentage of Respondents (%)",
        yaxis_title="",
        plot_bgcolor='white',
        height=450
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    st.markdown("#### Priorities by Demographics",
               help="This chart shows the top priority for each demographic group based on which issue was mentioned most often. The bar height shows how many times that top issue was mentioned. Different colors indicate different top priorities. This reveals if different groups have different concerns.")
    
    demo_choice = st.selectbox(
        "Compare priorities across:",
        ["Age Group", "Income Level", "County", "Race/Ethnicity"]
    )
    
    demo_map = {
        "Age Group": 'age_group',
        "Income Level": 'income_bracket',
        "County": 'COUNTY',
        "Race/Ethnicity": 'race_ethnicity'
    }
    
    demo_col = demo_map[demo_choice]
    
    demo_priorities = []
    for group in filtered_df[demo_col].unique():
        if pd.isna(group):
            continue
        group_df = filtered_df[filtered_df[demo_col] == group]
        group_priorities = []
        for col in priority_cols:
            group_priorities.extend(group_df[col].dropna().tolist())
        if group_priorities:
            top_priority = pd.Series(group_priorities).value_counts().index[0]
            demo_priorities.append({
                'Group': group,
                'Top Priority': priority_map[top_priority],
                'Mentions': pd.Series(group_priorities).value_counts().iloc[0]
            })
    
    demo_priorities_df = pd.DataFrame(demo_priorities)
    
    if len(demo_priorities_df) > 0:
        fig = px.bar(
            demo_priorities_df,
            x='Group',
            y='Mentions',
            color='Top Priority',
            text='Top Priority',
            color_discrete_sequence=COLORS['categorical'],
            title=f"Top Priority by {demo_choice}"
        )
        fig.update_traces(textposition='inside', textangle=0)
        fig.update_layout(
            xaxis_title=demo_choice,
            yaxis_title="Number of Mentions",
            plot_bgcolor='white',
            height=450
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    st.markdown("#### Housing Affordability Deep Dive",
               help="Left chart: Shows average housing situation severity (0-10 scale) by county. Higher scores mean more severe problems. Right chart: Shows the top reasons residents believe contribute to housing challenges, as percentages of all respondents.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        housing_by_county = filtered_df.groupby('COUNTY')['HOUSING1'].mean().sort_values(ascending=False)
        
        fig = go.Figure(go.Bar(
            x=housing_by_county.index,
            y=housing_by_county.values,
            marker=dict(
                color=housing_by_county.values,
                colorscale='Reds',
                showscale=False
            ),
            text=housing_by_county.values.round(2),
            textposition='outside'
        ))
        
        fig.update_layout(
            title="Housing Situation Severity by County (0-10)",
            xaxis_title="County",
            yaxis_title="Mean Severity Score",
            plot_bgcolor='white',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        housing_reason_cols = ['HOUSING2_1', 'HOUSING2_2', 'HOUSING2_3', 'HOUSING2_4']
        housing_reasons_data = []
        
        for col in housing_reason_cols:
            housing_reasons_data.extend(filtered_df[col].dropna().tolist())
        
        housing_reason_map = {
            1: 'Corporations buying properties',
            2: 'Landlords raising rent',
            3: 'Opposition to affordable housing',
            4: 'Out-of-state migration',
            5: 'Only expensive homes built',
            6: 'Individuals buying rentals',
            7: 'Government not doing enough',
            8: 'Laws not enforced',
            9: 'Not enough housing built',
            10: "Don't know"
        }
        
        housing_reason_counts = pd.Series(housing_reasons_data).value_counts()
        housing_reason_counts.index = housing_reason_counts.index.map(housing_reason_map)
        housing_reason_pct = (housing_reason_counts / len(filtered_df) * 100).sort_values(ascending=False).head(8)
        
        fig = go.Figure(go.Bar(
            x=housing_reason_pct.values,
            y=housing_reason_pct.index,
            orientation='h',
            marker_color=COLORS['warning'],
            text=[f"{val:.1f}%" for val in housing_reason_pct.values],
            textposition='outside'
        ))
        
        fig.update_layout(
            title="Top Reasons for Housing Challenges",
            xaxis_title="% of Respondents",
            yaxis_title="",
            plot_bgcolor='white',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    st.markdown("#### Regional Outlook: Future Expectations",
               help="Left chart: Pie chart showing the proportion of respondents who think the Charlotte region will be better, worse, same, or don't know in 3-4 years. Right chart: Shows the current life satisfaction of people grouped by their outlook. This reveals if optimistic people are currently happier.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        outlook_dist = filtered_df['outlook'].value_counts()
        outlook_order = ['Better', 'Same', 'Worse', "Don't Know"]
        outlook_dist = outlook_dist.reindex(outlook_order, fill_value=0)
        outlook_colors = ['#2ca02c', '#ffcc00', '#d62728', '#999999']
        
        fig = go.Figure(data=[go.Pie(
            labels=outlook_order,
            values=outlook_dist.values,
            marker=dict(colors=outlook_colors),
            textinfo='label+percent',
            hovertemplate='%{label}<br>%{value} respondents<br>%{percent}<extra></extra>'
        )])
        
        fig.update_layout(
            title="Outlook for Charlotte Region (3-4 years)",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        outlook_by_satisfaction = filtered_df.groupby('outlook')['LIFESAS'].mean().reindex(outlook_order)
        
        fig = go.Figure(go.Bar(
            x=outlook_order,
            y=outlook_by_satisfaction.values,
            marker=dict(color=outlook_colors),
            text=outlook_by_satisfaction.values.round(2),
            textposition='outside'
        ))
        
        fig.update_layout(
            title="Current Life Satisfaction by Future Outlook",
            xaxis_title="Regional Outlook",
            yaxis_title="Mean Life Satisfaction",
            plot_bgcolor='white',
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    st.markdown("#### Financial Well-Being",
               help="Left chart: How people's financial situation has changed compared to one year ago. Right chart: How confident people feel about handling an unexpected $400 expense. Higher bars on the right indicate better financial security.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fin_change_dist = filtered_df['financial_change'].value_counts()
        fin_order = ['Better off', 'About the same', 'Worse off', "Don't know"]
        fin_change_dist = fin_change_dist.reindex(fin_order, fill_value=0)
        
        fig = px.bar(
            x=fin_order,
            y=fin_change_dist.values,
            labels={'x': 'Financial Change', 'y': 'Count'},
            title="Financial Situation vs. One Year Ago",
            color=fin_change_dist.values,
            color_continuous_scale=['red', 'yellow', 'green']
        )
        fig.update_layout(plot_bgcolor='white', height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        emergency_dist = filtered_df['Q8'].value_counts().sort_index()
        confidence_labels = ['Not at all', 'Slightly', 'Moderately', 'Very', 'Completely']
        
        fig = go.Figure(go.Bar(
            x=confidence_labels,
            y=emergency_dist.values,
            marker=dict(
                color=emergency_dist.values,
                colorscale='RdYlGn',
                showscale=False
            ),
            text=emergency_dist.values,
            textposition='outside'
        ))
        
        fig.update_layout(
            title="Confidence Handling Unexpected Expenses",
            xaxis_title="Confidence Level",
            yaxis_title="Number of Respondents",
            plot_bgcolor='white',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

def show_correlations(filtered_df, full_df):
    """Correlations and insights page"""
    st.title("📈 Correlations & Statistical Insights")
    st.markdown("### Understanding Relationships in the Data")
    
    st.markdown("#### Well-Being & Key Variables Correlation Matrix",
               help="This correlation matrix shows how different variables relate to each other. Each cell shows the correlation coefficient from -1 to +1. Red indicates negative correlation (as one increases, the other decreases). Blue indicates positive correlation (both increase together). Darker colors indicate stronger relationships. Numbers show exact correlation values.")
    
    corr_vars = ['LIFESAS', 'LIFEWW', 'TRUST', 'BELONGNEED', 'BELONGCOM', 
                 'Q8', 'HOUSING1', 'Q11_day', 'Q11_night']
    corr_var_names = ['Life Sat.', 'Worthwhile', 'Trust', 'Support', 'Belonging',
                      'Emergency Conf.', 'Housing Concern', 'Safety Day', 'Safety Night']
    
    corr_data = filtered_df[corr_vars].corr()
    
    fig = px.imshow(
        corr_data,
        labels=dict(color="Correlation"),
        x=corr_var_names,
        y=corr_var_names,
        color_continuous_scale='RdBu_r',
        zmin=-1,
        zmax=1,
        text_auto='.2f',
        aspect='auto'
    )
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    st.markdown("#### Strongest Correlations with Life Satisfaction",
               help="These charts show which variables have the strongest relationships with life satisfaction. Left (green): Variables that increase with life satisfaction. Right (red): Variables that decrease as life satisfaction increases. Longer bars indicate stronger relationships. Use this to identify key drivers of well-being.")
    
    lifesas_corr = corr_data['LIFESAS'].drop('LIFESAS').sort_values(ascending=False)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### Positive Correlations")
        pos_corr = lifesas_corr[lifesas_corr > 0].head(5)
        
        fig = go.Figure(go.Bar(
            x=pos_corr.values,
            y=[corr_var_names[corr_vars.index(var)] for var in pos_corr.index],
            orientation='h',
            marker_color=COLORS['success'],
            text=pos_corr.values.round(3),
            textposition='outside'
        ))
        fig.update_layout(
            xaxis_title="Correlation with Life Satisfaction",
            yaxis_title="",
            plot_bgcolor='white',
            height=300,
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("##### Negative Correlations")
        neg_corr = lifesas_corr[lifesas_corr < 0].tail(5)
        
        fig = go.Figure(go.Bar(
            x=neg_corr.values,
            y=[corr_var_names[corr_vars.index(var)] for var in neg_corr.index],
            orientation='h',
            marker_color=COLORS['warning'],
            text=neg_corr.values.round(3),
            textposition='outside'
        ))
        fig.update_layout(
            xaxis_title="Correlation with Life Satisfaction",
            yaxis_title="",
            plot_bgcolor='white',
            height=300,
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    st.markdown("#### Income Gradient in Well-Being",
               help="This line chart shows how well-being metrics change across income levels. Each colored line represents a different metric. Upward slopes indicate that higher income is associated with higher scores on that metric. Steeper slopes indicate stronger relationships. Look for which metrics show the steepest income gradients.")
    
    valid_income = filtered_df[filtered_df['income_numeric'].notna()].copy()
    
    if len(valid_income) > 0:
        income_wellbeing = valid_income.groupby('income_bracket').agg({
            'LIFESAS': 'mean',
            'LIFEWW': 'mean',
            'TRUST': 'mean',
            'BELONGNEED': 'mean',
            'respondent_id': 'count'
        }).reset_index()
        
        income_order = ['<$15K', '$15-25K', '$25-35K', '$35-50K', '$50-75K', '$75-100K', '$100-150K', '$150K+']
        income_wellbeing['income_bracket'] = pd.Categorical(
            income_wellbeing['income_bracket'],
            categories=income_order,
            ordered=True
        )
        income_wellbeing = income_wellbeing.sort_values('income_bracket')
        
        fig = go.Figure()
        
        metrics = [
            ('LIFESAS', 'Life Satisfaction', COLORS['primary']),
            ('LIFEWW', 'Worthwhileness', COLORS['secondary']),
            ('TRUST', 'Trust', COLORS['info']),
            ('BELONGNEED', 'Social Support', COLORS['success'])
        ]
        
        for metric, name, color in metrics:
            fig.add_trace(go.Scatter(
                x=income_wellbeing['income_bracket'],
                y=income_wellbeing[metric],
                name=name,
                mode='lines+markers',
                line=dict(color=color, width=3),
                marker=dict(size=8)
            ))
        
        fig.update_layout(
            title="Well-Being Metrics Across Income Levels",
            xaxis_title="Income Bracket",
            yaxis_title="Mean Score (0-10)",
            plot_bgcolor='white',
            height=450,
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # FIXED: Employment status comparison with proper spacing
    st.markdown("---")
    st.markdown("#### Well-Being by Employment Status",
               help="These three bar charts compare employment status groups across different well-being metrics. Left: Life satisfaction levels. Middle: Confidence in handling unexpected expenses. Right: Housing concern levels. Taller bars indicate higher average scores. Compare patterns - do unemployed individuals differ across all metrics?")
    
    employment_stats = filtered_df.groupby('employment_status').agg({
        'LIFESAS': 'mean',
        'Q8': 'mean',
        'HOUSING1': 'mean',
        'respondent_id': 'count'
    }).reset_index()
    
    # Create individual plots with proper spacing
    fig1 = go.Figure(go.Bar(
        x=employment_stats['employment_status'],
        y=employment_stats['LIFESAS'],
        marker_color=COLORS['primary'],
        text=employment_stats['LIFESAS'].round(2),
        textposition='outside',
        name='Life Satisfaction'
    ))
    fig1.update_layout(
        title="Life Satisfaction",
        xaxis_title="Employment Status",
        yaxis_title="Score",
        plot_bgcolor='white',
        height=400,
        showlegend=False
    )
    
    fig2 = go.Figure(go.Bar(
        x=employment_stats['employment_status'],
        y=employment_stats['Q8'],
        marker_color=COLORS['secondary'],
        text=employment_stats['Q8'].round(2),
        textposition='outside',
        name='Emergency Confidence'
    ))
    fig2.update_layout(
        title="Emergency Confidence",
        xaxis_title="Employment Status",
        yaxis_title="Score",
        plot_bgcolor='white',
        height=400,
        showlegend=False
    )
    
    fig3 = go.Figure(go.Bar(
        x=employment_stats['employment_status'],
        y=employment_stats['HOUSING1'],
        marker_color=COLORS['warning'],
        text=employment_stats['HOUSING1'].round(2),
        textposition='outside',
        name='Housing Concern'
    ))
    fig3.update_layout(
        title="Housing Concern",
        xaxis_title="Employment Status",
        yaxis_title="Score",
        plot_bgcolor='white',
        height=400,
        showlegend=False
    )
    
    # Display in three columns
    col1, col2, col3 = st.columns(3)
    with col1:
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        st.plotly_chart(fig2, use_container_width=True)
    with col3:
        st.plotly_chart(fig3, use_container_width=True)
    
    st.markdown("---")
    st.markdown("#### Statistical Significance Testing",
               help="ANOVA (Analysis of Variance) tests whether groups differ significantly on the selected metric. F-statistic: Higher values indicate larger differences between groups. P-value: Values below 0.05 indicate statistically significant differences (unlikely due to chance). Stars indicate significance level: *** (p<0.001) very strong, ** (p<0.01) strong, * (p<0.05) moderate.")
    
    st.markdown("""
    Compare well-being metrics across groups using ANOVA (Analysis of Variance).
    Lower p-values (< 0.05) indicate statistically significant differences between groups.
    """)
    
    # Two-column layout for selections
    col_select1, col_select2 = st.columns(2)
    
    with col_select1:
        test_metric = st.selectbox(
            "Select metric to test:",
            ["Life Satisfaction", "Worthwhileness", "Trust", "Social Support", 
             "Community Belonging", "Emergency Confidence", "Housing Concern"]
        )
    
    with col_select2:
        test_variable = st.selectbox(
            "Select grouping variable:",
            ["Age Group", "Employment Status", "County Type", "Income Level", "Race/Ethnicity", "County"]
        )
    
    # Map metric names to column names
    metric_map = {
        "Life Satisfaction": "LIFESAS",
        "Worthwhileness": "LIFEWW",
        "Trust": "TRUST",
        "Social Support": "BELONGNEED",
        "Community Belonging": "BELONGCOM",
        "Emergency Confidence": "Q8",
        "Housing Concern": "HOUSING1"
    }
    
    # Map grouping variables to column names
    test_map = {
        "Age Group": 'age_group',
        "Employment Status": 'employment_status',
        "County Type": 'area_type',
        "Income Level": 'income_bracket',
        "Race/Ethnicity": 'race_ethnicity',
        "County": 'COUNTY'
    }
    
    test_col = test_map[test_variable]
    metric_col = metric_map[test_metric]
    
    # Create area_type if needed
    if test_col == 'area_type' and 'area_type' not in filtered_df.columns:
        urban_counties = ['Mecklenburg', 'Gaston', 'Cabarrus']
        filtered_df['area_type'] = filtered_df['COUNTY'].apply(
            lambda x: 'Urban/Suburban' if x in urban_counties else 'Rural/Small Town'
        )
    
    # Collect groups for ANOVA
    groups = []
    group_names = []
    for name, group in filtered_df.groupby(test_col):
        if not pd.isna(name) and len(group) > 0:
            group_data = group[metric_col].dropna()
            if len(group_data) > 0:  # Only include groups with data
                groups.append(group_data)
                group_names.append(name)
    
    if len(groups) >= 2:
        # Perform ANOVA
        f_stat, p_value = stats.f_oneway(*groups)
        
        # Calculate effect size (eta-squared)
        # Total sum of squares
        all_data = pd.concat(groups)
        grand_mean = all_data.mean()
        ss_total = ((all_data - grand_mean) ** 2).sum()
        
        # Between-group sum of squares
        ss_between = sum([len(g) * ((g.mean() - grand_mean) ** 2) for g in groups])
        
        # Eta-squared
        eta_squared = ss_between / ss_total if ss_total > 0 else 0
        
        # Display results
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("F-Statistic", f"{f_stat:.4f}")
        with col2:
            st.metric("P-Value", f"{p_value:.6f}")
        with col3:
            st.metric("η² (Effect Size)", f"{eta_squared:.4f}")
        with col4:
            if p_value < 0.001:
                sig = "***"
                sig_text = "p < 0.001"
                color = "green"
            elif p_value < 0.01:
                sig = "**"
                sig_text = "p < 0.01"
                color = "green"
            elif p_value < 0.05:
                sig = "*"
                sig_text = "p < 0.05"
                color = "orange"
            else:
                sig = "ns"
                sig_text = "Not significant"
                color = "red"
            
            st.markdown(f"<h3 style='color: {color};'>{sig}</h3>", unsafe_allow_html=True)
            st.caption(sig_text)
        
        # Interpretation
        if p_value < 0.05:
            effect_size_interpretation = ""
            if eta_squared < 0.01:
                effect_size_interpretation = "small"
            elif eta_squared < 0.06:
                effect_size_interpretation = "medium"
            else:
                effect_size_interpretation = "large"
            
            st.success(f"✓ There are **statistically significant** differences in {test_metric.lower()} across {test_variable.lower()} groups (p {sig_text}). The effect size is **{effect_size_interpretation}** (η² = {eta_squared:.4f}).")
        else:
            st.info(f"The differences in {test_metric.lower()} across {test_variable.lower()} groups are **not statistically significant** (p = {p_value:.4f}).")
        
        # Show group means
        st.markdown("---")
        st.markdown("##### Group Means and Sample Sizes")
        
        group_summary = []
        for name, group_data in zip(group_names, groups):
            group_summary.append({
                'Group': str(name),
                'N': len(group_data),
                'Mean': group_data.mean(),
                'SD': group_data.std(),
                'Min': group_data.min(),
                'Max': group_data.max()
            })
        
        group_summary_df = pd.DataFrame(group_summary).sort_values('Mean', ascending=False)
        
        st.dataframe(
            group_summary_df.style.background_gradient(
                subset=['Mean'],
                cmap='RdYlGn',
                vmin=group_summary_df['Mean'].min(),
                vmax=group_summary_df['Mean'].max()
            ).format({
                'Mean': '{:.2f}',
                'SD': '{:.2f}',
                'Min': '{:.2f}',
                'Max': '{:.2f}'
            }),
            use_container_width=True
        )
        
        # Visualization of group means
        st.markdown("##### Visual Comparison")
        
        fig = go.Figure()
        
        # Sort by mean for better visualization
        sorted_summary = group_summary_df.sort_values('Mean', ascending=True)
        
        fig.add_trace(go.Bar(
            y=sorted_summary['Group'],
            x=sorted_summary['Mean'],
            orientation='h',
            marker=dict(
                color=sorted_summary['Mean'],
                colorscale='RdYlGn',
                showscale=True,
                colorbar=dict(title=test_metric)
            ),
            text=sorted_summary['Mean'].round(2),
            textposition='outside',
            error_x=dict(
                type='data',
                array=1.96 * sorted_summary['SD'] / np.sqrt(sorted_summary['N']),  # 95% CI
                color='rgba(0,0,0,0.3)',
                thickness=2
            ),
            hovertemplate='<b>%{y}</b><br>' +
                         test_metric + ': %{x:.2f}<br>' +
                         'N: %{customdata}<extra></extra>',
            customdata=sorted_summary['N']
        ))
        
        chart_title = f"{test_metric} by {test_variable} (with 95% Confidence Intervals)"
        x_axis_title = f"Mean {test_metric}"
        
        fig.update_layout(
            title=chart_title,
            xaxis_title=x_axis_title,
            yaxis_title=test_variable,
            plot_bgcolor='white',
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.warning(f"Not enough groups with data to perform ANOVA. Need at least 2 groups with valid {test_metric.lower()} scores.")

if __name__ == "__main__":
    main()
