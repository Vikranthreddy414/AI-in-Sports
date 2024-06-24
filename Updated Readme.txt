import streamlit as st
import pandas as pd
import os
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go

# Set up logging
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
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

def display_emails_by_topic(df: pd.DataFrame, topic_number: int):
    """Display emails filtered by topic number."""
    filtered_emails = df[df['topic_number'] == topic_number][['cleaned_subject', 'cleaned_body']]
    st.write(filtered_emails.head(5))
    csv = filtered_emails.to_csv(index=False)
    st.download_button(
        label=f"Download all emails for Topic {topic_number}",
        data=csv,
        file_name=f'filtered_emails_topic_{topic_number}.csv',
        mime='text/csv',
    )

def display_filtered_data(df: pd.DataFrame, label: str):
    """Display filtered data by label."""
    filtered_data = df[df['label'] == label][['Topic', 'Count', 'OpenAI', 'KeyBert']]
    st.write(filtered_data)
    for topic in filtered_data['Topic']:
        if st.button(f'Click me for Topic {topic}', key=f'clickme_{label}_{topic}'):
            st.session_state.selected_topic = topic
            st.experimental_rerun()

def main():
    st.sidebar.header("Filters")
    selected_date = st.sidebar.date_input("Select a date", value=None, key="main_date")

    st.markdown("<h1 style='text-align: center; color: black;'>Angry Email Analysis</h1>", unsafe_allow_html=True)

    if 'date_selected' not in st.session_state:
        st.session_state.date_selected = False

    if selected_date:
        st.session_state.date_selected = True

    if not st.session_state.date_selected:
        st.markdown("<p style='text-align: center; font-style: italic; color: black;'>Please select the date from the left side bar</p>", unsafe_allow_html=True)

    if st.session_state.date_selected and selected_date:
        year = selected_date.year
        month = f"{selected_date.month:02d}"
        day = f"{selected_date.day:02d}"
        yyyymmdd = f"{year}{month}{day}"

        base_folder = f"C:\\Users\\s_analysis\\{year}\\{month}\\{day}\\"
        results_file_path = os.path.join(base_folder, f"results_{yyyymmdd}.csv")
        topic_summary_file_path = os.path.join(base_folder, f"topic_summary_{yyyymmdd}.csv")

        if os.path.exists(results_file_path) and os.path.exists(topic_summary_file_path):
            topic_info = load_data(topic_summary_file_path)
            df_results = load_data(results_file_path)

            if topic_info.empty or df_results.empty:
                st.error("Error loading data, please check the logs.")
                return

            candidate_labels = []

            label_counts = pd.DataFrame({'label': candidate_labels, 'Count': [0] * len(candidate_labels)})

            label_counts_actual = topic_info.groupby('label')['Count'].sum().reset_index()

            label_counts = label_counts.merge(label_counts_actual, on='label', how='left').fillna(0)
            label_counts['Count'] = label_counts['Count_x'] + label_counts['Count_y']
            label_counts = label_counts[['label', 'Count']].astype({'Count': 'int'})

            st.markdown("""
                <style>
                .main-title {
                    font-size: 2.5rem;
                    font-weight: bold;
                    color: #333;
                    text-align: center;
                }
                .sub-header {
                    font-size: 1.5rem;
                    color: #555;
                    text-align: center;
                }
                .section-header {
                    font-size: 1.25rem;
                    color: #777;
                    margin-top: 2rem;
                    margin-bottom: 1rem;
                    text-align: center;
                }
                .count-box {
                    border: 1px solid #ddd;
                    border-radius: 5px;
                    padding: 10px;
                    text-align: center;
                    font-size: 1.1rem;
                    font-weight: bold;
                    color: #333;
                    margin-bottom: 10px;
                    width: 120px;
                    background-color: #f9f9f9;
                    cursor: pointer;
                }
                .count-box:hover {
                    background-color: #e6e6e6;
                }
                .button-grid {
                    display: flex;
                    flex-wrap: wrap;
                    justify-content: center;
                }
                .button-grid > div {
                    margin: 5px;
                }
                </style>
            """, unsafe_allow_html=True)

            if 'selected_topic' not in st.session_state:
                st.session_state.selected_topic = None
            if 'selected_category' not in st.session_state:
                st.session_state.selected_category = None

            st.markdown('<div class="sub-header">Email Counts by Label</div>', unsafe_allow_html=True)

            cols = st.columns(3)
            buttons_per_col = -(-len(label_counts) // 3)

            for idx, (label, count) in enumerate(label_counts.itertuples(index=False)):
                col = cols[idx % 3]
                with col:
                    if st.button(f"{label}: {int(count)}", key=f"label_{label}"):
                        st.session_state.selected_category = label
                        st.session_state.selected_topic = None
                        st.experimental_rerun()

            if st.session_state.selected_category:
                st.write(f"{st.session_state.selected_category} related angry emails:")
                display_filtered_data(topic_info, st.session_state.selected_category)

            if st.session_state.selected_topic is not None:
                st.markdown(f'<div class="section-header">Emails for Topic {st.session_state.selected_topic}</div>', unsafe_allow_html=True)
                display_emails_by_topic(df_results, st.session_state.selected_topic)

        else:
            st.error("Sorry, I don't have data for the selected date")

def trends():
    st.sidebar.header("Trends")
    start_date = st.sidebar.date_input("Start date", key="start_date", value=None)
    end_date = st.sidebar.date_input("End date", key="end_date", value=None)

    st.markdown("<h1 style='text-align: center; color: black;'>Trend Analysis</h1>", unsafe_allow_html=True)

    if not start_date or not end_date:
        st.markdown("<p style='text-align: center; font-style: italic; color: black;'>Please select the date range from the left side bar</p>", unsafe_allow_html=True)
    else:
        if start_date > end_date:
            st.error("End date must be after start date")
            return

        all_data = []
        current_date = start_date

        while current_date <= end_date:
            year = current_date.year
            month = f"{current_date.month:02d}"
            day = f"{current_date.day:02d}"
            yyyymmdd = f"{year}{month}{day}"

            topic_summary_file_path = f"C:\\User\\s_analysis\\{year}\\{month}\\{day}\\topic_summary_{yyyymmdd}.csv"
            if os.path.exists(topic_summary_file_path):
                data = load_data(topic_summary_file_path)
                if not data.empty:
                    data['date'] = current_date
                    all_data.append(data)
            else:
                st.error(f"Sorry, I don't have data for the selected date range")
                return

            current_date += timedelta(days=1)

        if all_data:
            aggregated_data = pd.concat(all_data)
            selected_labels = st.multiselect("Select Labels", options=aggregated_data['label'].unique(), default=aggregated_data['label'].unique())
            if selected_labels:
                display_trends(aggregated_data, selected_labels)

def display_trends(data: pd.DataFrame, selected_labels: list):
    trend_data = data[data['label'].isin(selected_labels)].groupby(['date', 'label'])['Count'].sum().reset_index()
    trend_data['date'] = pd.to_datetime(trend_data['date']).dt.date  # Ensure date format is correct

    # Total count of emails
    total_emails = trend_data['Count'].sum()
    st.markdown(f"### Total Emails: {total_emails}")

    # Total volume per label
    volume_per_label = trend_data.groupby('label')['Count'].sum().reset_index()
    st.markdown("### Total Volume per Label")
    fig_volume = px.bar(volume_per_label, x='label', y='Count', title='Total Volume per Label', text='Count')
    fig_volume.update_traces(marker_color='dodgerblue')
    fig_volume.update_layout(
        xaxis_title='Label',
        yaxis_title='Count',
        template='plotly_white',
        title_x=0.5,
        showlegend=False
    )
    st.plotly_chart(fig_volume, use_container_width=True)

    # Total count per day with respect to label - trend graph
    st.markdown("### Trend of Emails per Day by Label")
    fig_trend = px.line(trend_data, x='date', y='Count', color='label', title='Trend of Selected Labels over time', line_shape='spline')
    fig_trend.update_layout(
        xaxis_title='Date',
        yaxis_title='Count',
        template='plotly_white',
        title_x=0.5,
        showlegend=True
    )
    fig_trend.update_xaxes(dtick="D1", tickformat="%Y-%m-%d")
    st.plotly_chart(fig_trend, use_container_width=True)

    # Difference from last day for each label
    trend_data['previous_day'] = trend_data.groupby('label')['Count'].shift(1)
    trend_data['difference'] = trend_data['Count'] - trend_data['previous_day']
    difference_per_label = trend_data.groupby('label').apply(lambda x: x.iloc[-1])[['label', 'difference']].reset_index(drop=True)
    st.markdown("### Difference from Last Day for Each Label")
    fig_difference = px.bar(difference_per_label, x='label', y='difference', title='Difference from Last Day for Each Label', text='difference')
    fig_difference.update_traces(marker_color='lightcoral')
    fig_difference.update_layout(
        xaxis_title='Label',
        yaxis_title='Difference',
        template='plotly_white',
        title_x=0.5,
        showlegend=False
    )
    st.plotly_chart(fig_difference, use_container_width=True)

if __name__ == "__main__":
    st.sidebar.title("Angry Email Analysis")
    page = st.sidebar.selectbox("Choose a page", ["Main", "Trends"])

    if page == "Main":
        main()
    elif page == "Trends":
        trends()
