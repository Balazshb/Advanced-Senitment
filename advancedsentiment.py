import pandas as pd
import streamlit as st
import plotly.express as px
from urllib.parse import urlparse, parse_qs, urlunparse
from datetime import datetime

# ===================== PASSWORD PROTECTION =====================
def check_password():
    """Returns `True` if the user enters the correct password."""
    
    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == "SocialCare2025":
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store the password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show password input
        st.text_input(
            "Enter Password", 
            type="password", 
            on_change=password_entered, 
            key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error
        st.text_input(
            "Enter Password", 
            type="password", 
            on_change=password_entered, 
            key="password"
        )
        st.error("Incorrect password. Please try again.")
        return False
    else:
        # Password correct
        return True

if not check_password():
    st.stop()  # Stop execution if password is wrong
# ===================== END PASSWORD PROTECTION =====================

# Function to clean URLs (handle duplicates)
def clean_url(url):
    if not isinstance(url, str) or not url.startswith('http'):
        return url
    
    try:
        parsed = urlparse(url)
        # Remove common tracking parameters
        query_params = parse_qs(parsed.query)
        for param in ['utm_source', 'utm_medium', 'utm_campaign', 'fbclid']:
            if param in query_params:
                del query_params[param]
        
        # Reconstruct URL without tracking parameters
        cleaned = parsed._replace(query="")
        return urlunparse(cleaned).split('?')[0]
    except:
        return url

# Function to load and process data
def load_data(file_path):
    df = pd.read_csv(file_path)
    
    # Filter for inbound public comments only
    df = df[(df['Direction'] == 'inbound') & 
            (df['Message Type'].str.contains('comment|post', case=False, na=False)) &
            (~df['Post URL'].isna())]
    
    # Clean URLs
    df['Cleaned Post URL'] = df['Post URL'].apply(clean_url)
    
    # Extract content preview for post cards
    df['Content Preview'] = df['Content'].str[:100] + '...'
    
    # Convert 'Created at' to datetime
    df['Created at'] = pd.to_datetime(df['Created at'])
    
    return df

# Function to get top posts by comment volume
def get_top_posts(df, top_n=20):
    comments_per_post = df.groupby('Cleaned Post URL').agg({
        'Message ID': 'count',
        'Sentiment': lambda x: x.value_counts().to_dict(),
        'Content': 'first',
        'Created at': 'first',
        'Channel': 'first'
    }).reset_index()
    
    comments_per_post.columns = ['Post URL', 'Total Comments', 'Sentiment Breakdown', 
                               'Content Preview', 'First Comment Date', 'Channel']
    
    # Sort and select top posts
    top_posts = comments_per_post.sort_values('Total Comments', ascending=False).head(top_n)
    
    # Create display names for dropdown
    top_posts['Display Name'] = top_posts.apply(
        lambda row: f"{row['Channel']} - {row['Content Preview'][:50]}... ({row['Total Comments']} comments)", 
        axis=1
    )
    
    return top_posts

# Function to create detailed post analysis
def create_post_analysis(df, selected_post_url, comments_to_show=10):
    post_details = df[df['Cleaned Post URL'] == selected_post_url]
    if post_details.empty:
        st.warning("No data available for the selected post.")
        return
    
    first_comment = post_details.iloc[0]
    
    # Display post header with basic info
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f"""
            <h3 style="margin-bottom: 0;">{first_comment['Channel']}</h3>
            <p style="margin-top: 0;"><small>Posted on: {first_comment['Created at'].strftime('%Y-%m-%d %H:%M')}</small></p>
            <p><strong>Platform:</strong> {first_comment['Medium']}</p>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
            <div style="text-align: right;">
                <a href="{selected_post_url}" target="_blank">
                    <button style="
                        background-color: #4CAF50;
                        color: white;
                        padding: 8px 16px;
                        border: none;
                        border-radius: 4px;
                        cursor: pointer;
                    ">View Original Post</button>
                </a>
            </div>
        """, unsafe_allow_html=True)
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Comments", len(post_details))
    col2.metric("Unique Commenters", post_details['Contact Primary Identifier'].nunique())
    col3.metric("First Comment Date", first_comment['Created at'].strftime('%Y-%m-%d'))
    
    st.markdown("---")
    
    # Sentiment analysis section
    st.subheader("Sentiment Analysis")
    
    # Define sentiment colors
    sentiment_colors = {
        'positive': '#4CAF50',  # Green
        'negative': '#F44336',  # Red
        'neutral': '#9E9E9E',   # Grey
        'Without sentiment': '#FF9800'  # Orange
    }
    
    # Replace NaN with "Without sentiment"
    post_details['Sentiment'] = post_details['Sentiment'].fillna('Without sentiment')
    
    # Create toggle buttons for each sentiment
    sentiments = ['positive', 'negative', 'neutral', 'Without sentiment']
    cols = st.columns(len(sentiments))
    sentiment_toggles = {}
    
    for i, sentiment in enumerate(sentiments):
        with cols[i]:
            sentiment_toggles[sentiment] = st.checkbox(
                f"Show {sentiment}", 
                value=True,
                key=f"post_sentiment_{sentiment}",
                help=f"Toggle {sentiment} comments in the chart"
            )
    
    # Filter sentiments based on toggles
    filtered_sentiments = [s for s in sentiments if sentiment_toggles[s]]
    filtered_df = post_details[post_details['Sentiment'].isin(filtered_sentiments)]
    
    # Checkbox to show comment counts on pie chart
    show_counts = st.checkbox("Show comment counts on chart", value=True)
    
    if not filtered_df.empty:
        post_sentiment = filtered_df['Sentiment'].value_counts().reset_index()
        post_sentiment.columns = ['Sentiment', 'Count']
        
        # Create pie chart
        fig = px.pie(post_sentiment, 
                    values='Count', 
                    names='Sentiment',
                    title=f"Sentiment Distribution",
                    color='Sentiment',
                    color_discrete_map=sentiment_colors)
        
        if show_counts:
            fig.update_traces(textposition='inside', 
                             textinfo='percent+label+value',
                             texttemplate='%{label}<br>%{value} (%{percent})')
        else:
            fig.update_traces(textposition='inside', 
                             textinfo='percent+label',
                             texttemplate='%{label}<br>%{percent}')
            
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No sentiments selected. Please enable at least one sentiment type.")
    
    st.markdown("---")
    
    # Comments table section
    st.subheader("Comments Analysis")
    
    # Let user select how many comments to show
    max_comments = min(50, len(post_details))
    comments_to_show = st.slider(
        "Number of comments to display per table", 
        min_value=1, 
        max_value=max_comments, 
        value=min(10, max_comments),
        help="Adjust how many comments to show in each table"
    )
    
    # Prepare comments data
    comments_display = post_details[['Created at', 'Contact Primary Identifier', 'Content', 'Sentiment']]
    comments_display = comments_display.sort_values('Created at', ascending=False)
    comments_display['Created at'] = comments_display['Created at'].dt.strftime('%Y-%m-%d %H:%M')
    
    # Function to apply background color
    def color_sentiment(val):
        if val == 'positive':
            return 'background-color: #4CAF50; color: white'
        elif val == 'negative':
            return 'background-color: #F44336; color: white'
        elif val == 'neutral':
            return 'background-color: #9E9E9E; color: white'
        elif val == 'Without sentiment':
            return 'background-color: #FF9800; color: white'
        return ''
    
    # Positive comments table
    st.subheader("Positive Comments", divider='green')
    positive_comments = comments_display[comments_display['Sentiment'] == 'positive'].head(comments_to_show)
    
    if not positive_comments.empty:
        st.dataframe(
            positive_comments.style.applymap(color_sentiment, subset=['Sentiment']),
            height=min(400, len(positive_comments) * 35 + 100),
            use_container_width=True
        )
    else:
        st.warning("No positive comments found.")
    
    # Negative comments table
    st.subheader("Negative Comments", divider='red')
    negative_comments = comments_display[comments_display['Sentiment'] == 'negative'].head(comments_to_show)
    
    if not negative_comments.empty:
        st.dataframe(
            negative_comments.style.applymap(color_sentiment, subset=['Sentiment']),
            height=min(400, len(negative_comments) * 35 + 100),
            use_container_width=True
        )
    else:
        st.warning("No negative comments found.")
        
    # Neutral comments table
    st.subheader("Neutral Comments", divider='grey')
    neutral_comments = comments_display[comments_display['Sentiment'] == 'neutral'].head(comments_to_show)
    
    if not neutral_comments.empty:
        st.dataframe(
            neutral_comments.style.applymap(color_sentiment, subset=['Sentiment']),
            height=min(400, len(neutral_comments) * 35 + 100),
            use_container_width=True
        )
    else:
        st.warning("No neutral comments found.")
        
    # Without sentiment comments table
    st.subheader("Without Sentiment Comments", divider='orange')
    nosentiment_comments = comments_display[comments_display['Sentiment'] == 'Without sentiment'].head(comments_to_show)
    
    if not nosentiment_comments.empty:
        st.dataframe(
            nosentiment_comments.style.applymap(color_sentiment, subset=['Sentiment']),
            height=min(400, len(nosentiment_comments) * 35 + 100),
            use_container_width=True
        )
    else:
        st.warning("No comments without sentiment found.")

# Function to create visualizations
def create_visualizations(df):
    st.title("Advanced Sentiment Analysis")
    
    # Overall stats
    st.header("Overall Statistics")
    col1, col2, col3 = st.columns(3)
    unique_posts = df['Cleaned Post URL'].nunique()
    col1.metric("Total Posts", unique_posts)
    col2.metric("Total Comments", len(df))
    col3.metric("Unique Commenters", df['Contact Primary Identifier'].nunique())
    
    # Sentiment distribution with customizable colors and toggles
    st.subheader("Comment Sentiment Distribution")
    
    # Define sentiment colors
    sentiment_colors = {
        'positive': '#4CAF50',  # Green
        'negative': '#F44336',  # Red
        'neutral': '#9E9E9E',   # Grey
        'Without sentiment': '#FF9800'  # Orange
    }
    
    # Replace NaN with "Without sentiment"
    df['Sentiment'] = df['Sentiment'].fillna('Without sentiment')
    
    # Create toggle buttons for each sentiment
    sentiments = ['positive', 'negative', 'neutral', 'Without sentiment']
    cols = st.columns(len(sentiments))
    sentiment_toggles = {}
    
    for i, sentiment in enumerate(sentiments):
        with cols[i]:
            sentiment_toggles[sentiment] = st.checkbox(
                f"Show {sentiment}", 
                value=True,
                key=f"overall_sentiment_{sentiment}",
                help=f"Toggle {sentiment} comments in the chart"
            )
    
    # Filter sentiments based on toggles
    filtered_sentiments = [s for s in sentiments if sentiment_toggles[s]]
    filtered_df = df[df['Sentiment'].isin(filtered_sentiments)]
    
    if not filtered_df.empty:
        sentiment_counts = filtered_df['Sentiment'].value_counts().reset_index()
        sentiment_counts.columns = ['Sentiment', 'Count']
        
        # Apply color mapping
        color_map = {s: sentiment_colors[s] for s in filtered_sentiments}
        
        fig2 = px.pie(sentiment_counts, values='Count', names='Sentiment', 
                     title="Overall Sentiment Distribution",
                     color='Sentiment',
                     color_discrete_map=color_map)
        st.plotly_chart(fig2)
    else:
        st.warning("No sentiments selected. Please enable at least one sentiment type.")
    
    # Comments per post with adjustable count
    st.subheader("Comments per Post")
    
    # Let user select how many posts to show
    max_posts = min(100, df['Cleaned Post URL'].nunique())
    post_count = st.slider(
        "Number of posts to display", 
        min_value=5, 
        max_value=max_posts, 
        value=20,
        help="Adjust how many posts to show in the chart"
    )
    
    comments_per_post = df.groupby('Cleaned Post URL').agg({
        'Message ID': 'count',
        'Sentiment': lambda x: x.value_counts().to_dict(),
        'Content': 'first',
        'Created at': 'first',
        'Channel': 'first'
    }).reset_index()
    comments_per_post.columns = ['Post URL', 'Total Comments', 'Sentiment Breakdown', 
                               'Content Preview', 'First Comment Date', 'Channel']
    
    # Expand sentiment breakdown into separate columns
    for sentiment in sentiment_colors.keys():
        comments_per_post[sentiment] = comments_per_post['Sentiment Breakdown'].apply(
            lambda x: x.get(sentiment, 0) if isinstance(x, dict) else 0)
    
    # Sort and select top posts
    top_posts = comments_per_post.sort_values('Total Comments', ascending=False).head(post_count)
    
    # Create interactive bar chart with visible URLs
    fig3 = px.bar(top_posts, 
                 x='Post URL', 
                 y='Total Comments',
                 hover_data=['positive', 'negative', 'neutral', 'Without sentiment', 'Channel'],
                 title=f"Top {post_count} Posts by Comment Volume",
                 labels={'Post URL': 'Post URL', 'Total Comments': 'Comment Count'})
    
    # Make URLs visible on x-axis
    fig3.update_xaxes(
        tickangle=45,
        tickmode='array',
        tickvals=top_posts['Post URL'],
        ticktext=[url.split('//')[-1][:30] + '...' for url in top_posts['Post URL']]
    )
    
    fig3.update_traces(hovertemplate=(
        "<b>%{customdata[4]}</b><br>"
        "URL: %{x}<br>"
        "Total Comments: %{y}<br>"
        "Positive: %{customdata[0]}<br>"
        "Negative: %{customdata[1]}<br>"
        "Neutral: %{customdata[2]}<br>"
        "Without Sentiment: %{customdata[3]}<br>"
        "<extra></extra>"
    ))
    st.plotly_chart(fig3, use_container_width=True)
    
    # Detailed Post Analysis section
    st.header("Detailed Post Analysis")
    
    # Get top posts for dropdown
    top_posts = get_top_posts(df, top_n=50)
    
    # Create dropdown for post selection
    selected_post = st.selectbox(
        "Select a post to analyze",
        options=top_posts['Post URL'],
        format_func=lambda url: top_posts[top_posts['Post URL'] == url]['Display Name'].iloc[0],
        help="Select a post from the top posts by comment volume"
    )
    
    # Show detailed analysis for selected post
    create_post_analysis(df, selected_post)

# Main function
def main():
    st.set_page_config(layout="centered", page_title="Advanced Sentiment Analysis")
    
    st.sidebar.title("Upload Data")
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            df = load_data(uploaded_file)
            create_visualizations(df)
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
    else:
        st.info("Please upload a CSV file to begin analysis")

if __name__ == "__main__":
    main()
