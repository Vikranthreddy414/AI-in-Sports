import streamlit as st
import pandas as pd
import os
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import altair as alt
import networkx as nx
from streamlit_agraph import agraph, Node, Edge, Config
import numpy as np
from scipy import stats

# Set up logging
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(layout="wide", page_title="Enhanced Email Analysis Dashboard", page_icon="📧")

st.markdown("""
    <style>
    .main, .sidebar .sidebar-content, .stApp, .css-18e3th9 {
        background-color: white;
    }
    .stApp {
        max-width: 1900px;
        margin: 0 auto;
    }
    h1, h2, h3 {
        color: #1f77b4;
    }
    .stButton>button {
        background-color: #1f77b4;
        color: white;
    }
    .streamlit-expanderHeader {
        background-color: white;
        color: #1f77b4;
    }
    .stDateInput>div>div>input {
        background-color: white;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data
def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file with caching."""
    try:
        data = pd.read_csv(file_path)
        logger.info(f"Successfully loaded data from {file_path}")
        return data
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        st.error(f"File not found: {file_path}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {e}")
        st.error(f"Error loading data from {file_path}")
        return pd.DataFrame()

def display_filtered_data(df: pd.DataFrame, label: str):
    """Display filtered data by label."""
    filtered_data = df[df['label'] == label][['Topic', 'Count', 'OpenAI', 'KeyBert']]
    st.dataframe(filtered_data, use_container_width=True)
    for topic in filtered_data['Topic']:
        if st.button(f'Click me for Topic {topic}', key=f'clickme_{label}_{topic}'):
            st.session_state.selected_topic = topic
            st.experimental_rerun()

def display_emails_by_topic(df: pd.DataFrame, topic_number: int):
    """Display emails filtered by topic number."""
    filtered_emails = df[df['topic_number'] == topic_number][['cleaned_subject', 'cleaned_body']]
    st.dataframe(filtered_emails.head(5), use_container_width=True)
    csv = filtered_emails.to_csv(index=False)
    st.download_button(
        label=f"Download all emails for Topic {topic_number}",
        data=csv,
        file_name=f'filtered_emails_topic_{topic_number}.csv',
        mime='text/csv',
    )

def get_last_n_days_data(selected_date, n_days):
    all_data = []
    for i in range(n_days):
        date = selected_date - timedelta(days=i)
        year, month, day = date.year, f"{date.month:02d}", f"{date.day:02d}"
        yyyymmdd = f"{year}{month}{day}"
        base_folder = f"C:\\Users\\vikra\\Downloads\\angry_emails_analysis\\{year}\\{month}\\{day}\\"
        topic_summary_file_path = os.path.join(base_folder, f"topic_summary_{yyyymmdd}.csv")
        if os.path.exists(topic_summary_file_path):
            data = load_data(topic_summary_file_path)
            if not data.empty:
                data['date'] = date
                all_data.append(data)
    return pd.concat(all_data) if all_data else pd.DataFrame()

def generate_wordcloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    return fig

def create_topic_network(topic_info):
    G = nx.Graph()
    for _, row in topic_info.iterrows():
        G.add_node(row['Topic'], size=row['Count'], group=row['label'])
    
    # Add edges based on some criteria (e.g., shared words in representation)
    for i, row1 in topic_info.iterrows():
        for j, row2 in topic_info.iloc[i+1:].iterrows():
            if len(set(row1['Representation'].split()) & set(row2['Representation'].split())) > 0:
                G.add_edge(row1['Topic'], row2['Topic'])
    
    nodes = [Node(id=node, label=f"Topic {node}", size=G.nodes[node]['size']/5, color=G.nodes[node]['group']) 
             for node in G.nodes()]
    edges = [Edge(source=edge[0], target=edge[1]) for edge in G.edges()]
    
    config = Config(width=700, height=500, directed=False, physics=True, hierarchical=False)
    return nodes, edges, config

def main_dashboard():
    st.sidebar.header("📅 Date Selection")
    selected_date = st.sidebar.date_input("Select a date", value=None, key="main_date")

    st.markdown("<h1 style='text-align: center;'>📧 Enhanced Email Analysis Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

    if not selected_date:
        st.info("👈 Please select a date from the left sidebar to begin your analysis.")
        return

    year, month, day = selected_date.year, f"{selected_date.month:02d}", f"{selected_date.day:02d}"
    yyyymmdd = f"{year}{month}{day}"

    base_folder = f"C:\\Users\\vikra\\Downloads\\angry_emails_analysis\\{year}\\{month}\\{day}\\"
    results_file_path = os.path.join(base_folder, f"results_{yyyymmdd}.csv")
    topic_summary_file_path = os.path.join(base_folder, f"topic_summary_{yyyymmdd}.csv")

    if not (os.path.exists(results_file_path) and os.path.exists(topic_summary_file_path)):
        st.error("🚫 Sorry, I don't have data for the selected date")
        return

    topic_info = load_data(topic_summary_file_path)
    df_results = load_data(results_file_path)

    if topic_info.empty or df_results.empty:
        st.error("❌ Error loading data, please check the logs.")
        return

    # Calculate metrics
    total_emails_today = int(topic_info['Count'].sum())
    previous_date = selected_date - timedelta(days=1)
    prev_yyyymmdd = f"{previous_date.year}{previous_date.month:02d}{previous_date.day:02d}"
    prev_topic_summary_file_path = f"C:\\Users\\vikra\\Downloads\\angry_emails_analysis\\{previous_date.year}\\{previous_date.month:02d}\\{previous_date.day:02d}\\topic_summary_{prev_yyyymmdd}.csv"

    if os.path.exists(prev_topic_summary_file_path):
        prev_topic_info = load_data(prev_topic_summary_file_path)
        total_emails_previous_day = int(prev_topic_info['Count'].sum())
        prev_label_counts = prev_topic_info.groupby('label')['Count'].sum().reset_index()
        email_difference = int(total_emails_today - total_emails_previous_day)
    else:
        total_emails_previous_day = 0
        prev_label_counts = pd.DataFrame({'label': topic_info['label'].unique(), 'Count': [0] * len(topic_info['label'].unique())})
        email_difference = None  # Set to None when we don't have previous day's data

    # Display metrics
    st.markdown("### 📊 Key Metrics")
    col1, col2, col3= st.columns(3)
    with col2:
        if email_difference is not None:
            st.metric("📬 Total Emails", value=total_emails_today, delta=email_difference)
        else:
            st.metric("📬 Total Emails", value=total_emails_today)
    with col1:
        st.metric("📅 Date", value=selected_date.strftime("%Y-%m-%d"))

    with col3:
        if st.button("Reset Category"):
            st.session_state.selected_category = None

    # Create category buttons with arrows
    st.markdown("### 📁 Email Categories")
    label_counts = topic_info.groupby('label')['Count'].sum().reset_index()
    label_diff = label_counts.merge(prev_label_counts, on='label', how='left', suffixes=('_today', '_previous'))
    label_diff['Count_diff'] = label_diff['Count_today'] - label_diff['Count_previous'].fillna(0)

    cols = st.columns(3)  # Changed to 4 columns to accommodate the reset button
    for idx, row in label_diff.iterrows():
        with cols[idx % 3]:
            if email_difference is not None:
                st.metric(
                    label=row['label'],
                    value=int(row['Count_today']),
                    delta=int(row['Count_diff']),
                    delta_color="normal"
                )
            else:
                st.metric(
                    label=row['label'],
                    value=int(row['Count_today'])
                )
            if st.button(f"Explore {row['label']}", key=f"label_{row['label']}"):
                st.session_state.selected_category = row['label']

    # Filter data based on selected category
    if 'selected_category' in st.session_state and st.session_state.selected_category:
        filtered_topic_info = topic_info[topic_info['label'] == st.session_state.selected_category]
        filtered_df_results = df_results[df_results['topic_number'].isin(filtered_topic_info['Topic'])]
    else:
        filtered_topic_info = topic_info
        filtered_df_results = df_results

    st.markdown("### 📊 Email Analysis Visualizations")
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Distribution", "📉 Trends", "🕸️ Topic Network", "🔍 Email Explorer"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            # Pie chart for category distribution
            fig_pie = px.pie(filtered_topic_info.groupby('label')['Count'].sum().reset_index(), 
                             values='Count', names='label', title='Email Distribution by Category')
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)

        with col2:
            # Bar chart for top 10 topics
            top_topics = filtered_topic_info.nlargest(10, 'Count')
            fig_bar = px.bar(top_topics, x='Topic', y='Count', color='label', title='Top 10 Topics')
            fig_bar.update_layout(xaxis_title='Topic', yaxis_title='Count')
            st.plotly_chart(fig_bar, use_container_width=True)

    with tab2:
        # Load data for the last 7 days
        last_7_days_data = get_last_n_days_data(selected_date, 7)

        if not last_7_days_data.empty:
            if 'selected_category' in st.session_state and st.session_state.selected_category:
                last_7_days_data = last_7_days_data[last_7_days_data['label'] == st.session_state.selected_category]

            # Line chart for overall trend
            trend_data = last_7_days_data.groupby('date')['Count'].sum().reset_index()
            fig_trend = px.line(trend_data, x='date', y='Count', title='Overall Email Count for Last 7 Days')
            fig_trend.update_layout(xaxis_title='Date', yaxis_title='Count')
            st.plotly_chart(fig_trend, use_container_width=True)

            # Stacked area chart for category trends
            category_trend = last_7_days_data.groupby(['date', 'label'])['Count'].sum().reset_index()
            fig_stacked = px.area(category_trend, x='date', y='Count', color='label', title='Email Categories Trend')
            fig_stacked.update_layout(xaxis_title='Date', yaxis_title='Count')
            st.plotly_chart(fig_stacked, use_container_width=True)

            # Heatmap of email volume by day and category
            pivot_data = category_trend.pivot(index='date', columns='label', values='Count').fillna(0)
            fig_heatmap = px.imshow(pivot_data.T, 
                                    labels=dict(x="Date", y="Category", color="Email Count"),
                                    title="Email Volume Heatmap by Day and Category")
            st.plotly_chart(fig_heatmap, use_container_width=True)
        else:
            st.info("No trend data available for the last 7 days.")

    with tab3:
        # Word cloud
        st.subheader("🔠 Word Cloud of Email Content")
        all_content = " ".join(filtered_df_results['cleaned_body'].astype(str))
        fig_wordcloud = generate_wordcloud(all_content)
        st.pyplot(fig_wordcloud)

        # Most common words
        st.subheader("📊 Most Common Words")
        words = all_content.split()
        word_freq = Counter(words).most_common(20)
        fig_common_words = px.bar(x=[word for word, _ in word_freq], y=[freq for _, freq in word_freq],
                                  title="20 Most Common Words")
        fig_common_words.update_layout(xaxis_title="Word", yaxis_title="Frequency")
        st.plotly_chart(fig_common_words, use_container_width=True)

        st.subheader("🕸️ Topic Network Visualization")
        nodes, edges, config = create_topic_network(filtered_topic_info)
        agraph(nodes=nodes, edges=edges, config=config)

    with tab4:
        st.markdown("### 🔍 Email Explorer")
        if 'selected_category' in st.session_state and st.session_state.selected_category:
            selected_category = st.session_state.selected_category
            st.write(f"Selected Category: {selected_category}")
            
            st.markdown(f"<b>{selected_category} related emails:</b>", unsafe_allow_html=True)
            display_filtered_data(filtered_topic_info, selected_category)
            
            if 'selected_topic' in st.session_state and st.session_state.selected_topic is not None:
                st.markdown(f'<div class="section-header">Emails for Topic {st.session_state.selected_topic}</div>', unsafe_allow_html=True)
                display_emails_by_topic(filtered_df_results, st.session_state.selected_topic)
        else:
            st.info("Please select a category to explore emails.")


        # Add JavaScript for scrolling
    
    
    scroll_script = """
        <script>
            var scrollToVisualizations = %s;
            if (scrollToVisualizations) {
                var visualizationsElement = document.getElementById('email-analysis-visualizations');
                if (visualizationsElement) {
                    visualizationsElement.scrollIntoView({ behavior: 'smooth' });
                }
            }
        </script>
    """ % str(st.session_state.get('scroll_to_visualizations', False)).lower()

    st.markdown(scroll_script, unsafe_allow_html=True)
    
    # Reset the scroll flag
    st.session_state.scroll_to_visualizations = False


def trends():
    st.sidebar.header("📅 Date Range")
    start_date = st.sidebar.date_input("Start date", key="start_date", value=None)
    end_date = st.sidebar.date_input("End date", key="end_date", value=None)

    st.markdown("<h1 style='text-align: center;'>📈 Advanced Trend Analysis</h1>", unsafe_allow_html=True)

    if not start_date or not end_date:
        st.info("👈 Please select the date range from the left sidebar")
    else:
        if start_date > end_date:
            st.error("❌ End date must be after start date")
            return
        all_data = []
        current_date = start_date

        with st.spinner("Loading data..."):
            while current_date <= end_date:
                year, month, day = current_date.year, f"{current_date.month:02d}", f"{current_date.day:02d}"
                yyyymmdd = f"{year}{month}{day}"

                topic_summary_file_path = f"C:\\Users\\vikra\\Downloads\\angry_emails_analysis\\{year}\\{month}\\{day}\\topic_summary_{yyyymmdd}.csv"
                if os.path.exists(topic_summary_file_path):
                    data = load_data(topic_summary_file_path)
                    if not data.empty:
                        data['date'] = current_date
                        all_data.append(data)
                else:
                    st.warning(f"No data available for {current_date}")

                current_date += timedelta(days=1)

        if all_data:
            aggregated_data = pd.concat(all_data)
            st.success("Data loaded successfully!")

            # Interactive label selection
            selected_labels = st.multiselect("Select Labels for Analysis", 
                                             options=aggregated_data['label'].unique(), 
                                             default=aggregated_data['label'].unique())
            
            # Add a "Clear Filter" button
            if st.button("Clear Filter"):
                selected_labels = aggregated_data['label'].unique()
            
            if selected_labels:
                display_trends(aggregated_data, selected_labels, start_date, end_date)
            else:
                st.warning("Please select at least one label to analyze.")
        else:
            st.error("No data available for the selected date range.")

def display_trends(data: pd.DataFrame, selected_labels: list, start_date: datetime, end_date: datetime):
    trend_data = data[data['label'].isin(selected_labels)].groupby(['date', 'label'])['Count'].sum().reset_index()
    trend_data['date'] = pd.to_datetime(trend_data['date'])

    # Total count of emails
    total_emails = trend_data['Count'].sum()
    st.metric("📬 Total Emails", f"{total_emails:,}")

    # Daily average
    days = (end_date - start_date).days + 1
    daily_avg = total_emails / days
    st.metric("📅 Daily Average", f"{daily_avg:.2f}")

    # Total volume per label
    volume_per_label = trend_data.groupby('label')['Count'].sum().reset_index()
    
    # Horizontal bar chart for total volume per label
    fig_volume = px.bar(volume_per_label, y='label', x='Count', title='Total Volume per Label', orientation='h')
    fig_volume.update_traces(marker_color='dodgerblue')
    fig_volume.update_layout(
        yaxis_title='Label',
        xaxis_title='Count',
        height=400,
        template='plotly_white',
        title_x=0.5,
        showlegend=False
    )
    st.plotly_chart(fig_volume, use_container_width=True)

    # Trend analysis
    st.markdown("### 📈 Trend Analysis")

    # Line chart for trend of emails per day by label
    fig_trend = px.line(trend_data, x='date', y='Count', color='label', title='Trend of Selected Labels over Time', line_shape='spline')
    fig_trend.update_layout(
        xaxis_title='Date',
        yaxis_title='Count',
        height=500,
        template='plotly_white',
        title_x=0.5,
        showlegend=True
    )
    fig_trend.update_xaxes(dtick="D1", tickformat="%Y-%m-%d")
    st.plotly_chart(fig_trend, use_container_width=True)

    # Stacked area chart
    fig_stacked = px.area(trend_data, x='date', y='Count', color='label', title='Cumulative Email Volume by Label')
    fig_stacked.update_layout(
        xaxis_title='Date',
        yaxis_title='Cumulative Count',
        height=500,
        template='plotly_white',
        title_x=0.5
    )
    st.plotly_chart(fig_stacked, use_container_width=True)

    # Weekly aggregation
    trend_data['week'] = trend_data['date'].dt.to_period('W')
    weekly_data = trend_data.groupby(['week', 'label'])['Count'].sum().reset_index()
    weekly_data['week'] = weekly_data['week'].astype(str)
    
    fig_weekly = px.bar(weekly_data, x='week', y='Count', color='label', title='Weekly Email Volume by Label')
    fig_weekly.update_layout(
        xaxis_title='Week',
        yaxis_title='Count',
        height=500,
        template='plotly_white',
        title_x=0.5,
        barmode='stack'
    )
    st.plotly_chart(fig_weekly, use_container_width=True)

    # Percentage change analysis
    st.markdown("### 📊 Percentage Change Analysis")
    
    def calculate_percentage_change(group):
        group['pct_change'] = group['Count'].pct_change() * 100
        return group

    pct_change_data = trend_data.groupby('label').apply(calculate_percentage_change).reset_index(drop=True)
    pct_change_data = pct_change_data.dropna()

    fig_pct_change = px.line(pct_change_data, x='date', y='pct_change', color='label', 
                             title='Daily Percentage Change in Email Volume by Label')
    fig_pct_change.update_layout(
        xaxis_title='Date',
        yaxis_title='Percentage Change (%)',
        height=500,
        template='plotly_white',
        title_x=0.5
    )
    st.plotly_chart(fig_pct_change, use_container_width=True)

    # Heatmap of percentage changes
    pivot_pct_change = pct_change_data.pivot(index='date', columns='label', values='pct_change')
    fig_heatmap = px.imshow(pivot_pct_change.T, 
                            labels=dict(x="Date", y="Label", color="% Change"),
                            title="Heatmap of Daily Percentage Changes",
                            color_continuous_scale="RdYlGn")
    fig_heatmap.update_layout(height=400)
    st.plotly_chart(fig_heatmap, use_container_width=True)

def main():
    st.sidebar.title("📧 Email Analysis")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["📊 Main Dashboard", "📈 Trend Analysis"],
        key="page_selection"
    )

    if page == "📊 Main Dashboard":
        main_dashboard()
    elif page == "📈 Trend Analysis":
        trends()

if __name__ == "__main__":
    main()
