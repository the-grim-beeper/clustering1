# app.py - ArXiv Paper Clustering Web App
import streamlit as st
import feedparser
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import plotly.graph_objects as go
import nltk
from nltk.corpus import stopwords
import time
import base64

# Set page configuration
st.set_page_config(
    page_title="ArXiv Paper Clustering",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Download NLTK resources
@st.cache_resource
def download_nltk_resources():
    nltk.download('stopwords')
    nltk.download('punkt')
    
download_nltk_resources()

class ArxivClusteringApp:
    def __init__(self):
        self.papers_df = None
        self.clusters = None
        self.outliers = None
        self.vectorizer = None
        self.tfidf_matrix = None
        self.pca_result = None
        
    def fetch_arxiv_rss(self, rss_url):
        """Fetch and parse an ArXiv RSS feed"""
        with st.spinner(f"Fetching papers from {rss_url}..."):
            feed = feedparser.parse(rss_url)
            
            papers = []
            for entry in feed.entries:
                # Extract paper information
                paper_id = entry.id.split('/')[-1]
                title = entry.title.replace('\n', ' ').strip()
                
                # Clean the summary/abstract
                summary = entry.summary.replace('\n', ' ')
                summary = re.sub(r'<.*?>', '', summary)  # Remove HTML tags
                summary = re.sub(r'\s+', ' ', summary).strip()
                
                # Get categories
                categories = [tag['term'] for tag in entry.tags] if 'tags' in entry else []
                
                # Add to papers list
                papers.append({
                    'id': paper_id,
                    'title': title,
                    'abstract': summary,
                    'categories': categories,
                    'link': entry.link
                })
            
            # Convert to DataFrame
            self.papers_df = pd.DataFrame(papers)
            return self.papers_df
    
    def vectorize_text(self):
        """Convert paper text to TF-IDF vectors"""
        if self.papers_df is None or len(self.papers_df) == 0:
            raise ValueError("No papers loaded. Fetch papers first.")
        
        # Combine title and abstract for better vectorization
        self.papers_df['text'] = self.papers_df['title'] + ' ' + self.papers_df['abstract']
        
        # Create TF-IDF vectors
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words=stopwords.words('english'),
            min_df=2,
            ngram_range=(1, 2)
        )
        
        self.tfidf_matrix = self.vectorizer.fit_transform(self.papers_df['text'])        
        return self.tfidf_matrix
    
    def cluster_papers(self, n_clusters=None):
        """Cluster papers using K-means"""
        if self.tfidf_matrix is None:
            self.vectorize_text()
        
        # Determine number of clusters if not specified
        if n_clusters is None:
            n_clusters = min(max(2, int(np.sqrt(len(self.papers_df)))), 10)
        
        # Apply K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.papers_df['cluster'] = kmeans.fit_predict(self.tfidf_matrix)
        
        # Calculate distance to cluster center (for outlier detection)
        cluster_centers = kmeans.cluster_centers_
        
        # For each paper, calculate similarity to its cluster center
        similarities = []
        for i, paper_idx in enumerate(range(len(self.papers_df))):
            paper_vector = self.tfidf_matrix[paper_idx]
            cluster_id = self.papers_df.iloc[paper_idx]['cluster']
            center_vector = cluster_centers[cluster_id]
            
            # Use cosine similarity (higher means more similar)
            sim = cosine_similarity(paper_vector, center_vector.reshape(1, -1))[0][0]
            similarities.append(sim)
        
        self.papers_df['similarity_to_center'] = similarities
        
        return self.papers_df
    
    def detect_outliers(self, threshold=1.5):
        """Identify outliers based on similarity to cluster center"""
        if 'similarity_to_center' not in self.papers_df.columns:
            raise ValueError("Papers not clustered yet. Run cluster_papers first.")
        
        # Find outliers for each cluster
        outliers = []
        
        for cluster_id in self.papers_df['cluster'].unique():
            cluster_papers = self.papers_df[self.papers_df['cluster'] == cluster_id]
            
            if len(cluster_papers) <= 1:
                continue  # Skip clusters with only one paper
            
            # Calculate threshold based on mean and std dev
            similarities = cluster_papers['similarity_to_center']
            mean_sim = similarities.mean()
            std_sim = similarities.std()
            
            if std_sim == 0:  # All papers identical
                continue
                
            outlier_threshold = mean_sim - threshold * std_sim
            
            # Find papers below threshold
            cluster_outliers = cluster_papers[cluster_papers['similarity_to_center'] < outlier_threshold]
            outliers.append(cluster_outliers)
        
        # Combine all outliers
        if outliers:
            self.outliers = pd.concat(outliers)
            self.papers_df['is_outlier'] = self.papers_df.index.isin(self.outliers.index)
        else:
            self.outliers = pd.DataFrame()
            self.papers_df['is_outlier'] = False
            
        return self.outliers
    
    def reduce_dimensions(self, n_components=2):
        """Reduce dimensionality for visualization"""
        if self.tfidf_matrix is None:
            raise ValueError("Text not vectorized yet. Run vectorize_text first.")
        
        # Apply PCA
        pca = PCA(n_components=n_components)
        self.pca_result = pca.fit_transform(self.tfidf_matrix.toarray())
        
        # Add coordinates to DataFrame
        self.papers_df['x'] = self.pca_result[:, 0]
        self.papers_df['y'] = self.pca_result[:, 1]
        
        return self.pca_result
    
    def create_plotly_visualization(self):
        """Create Plotly visualization of clusters"""
        if 'x' not in self.papers_df.columns:
            self.reduce_dimensions()
        
        # Prepare hover text
        self.papers_df['hover_text'] = self.papers_df.apply(
            lambda row: f"<b>{row['title']}</b><br><br>" + 
                       f"Categories: {', '.join(row['categories'])}<br>" +
                       f"Similarity: {row['similarity_to_center']:.3f}<br><br>" +
                       f"{row['abstract'][:200]}...",
            axis=1
        )
        
        # Create figure
        fig = px.scatter(
            self.papers_df, 
            x='x', 
            y='y',
            color='cluster',
            hover_name='title',
            hover_data={
                'x': False,
                'y': False,
                'cluster': True,
                'similarity_to_center': ':.3f',
                'is_outlier': True,
                'hover_text': False
            },
            custom_data=['id', 'link'],
            color_continuous_scale=px.colors.qualitative.G10,
            title='ArXiv Papers Clustering'
        )
        
        # Style the points
        fig.update_traces(
            marker=dict(
                size=10,
                line=dict(width=1, color='darkgray'),
            ),
            hovertemplate='%{hovertext}<extra></extra>'
        )
        
        # Add outlines to outliers
        outliers = self.papers_df[self.papers_df['is_outlier']]
        if len(outliers) > 0:
            fig.add_trace(
                go.Scatter(
                    x=outliers['x'],
                    y=outliers['y'],
                    mode='markers',
                    marker=dict(
                        size=15,
                        color='rgba(255, 0, 0, 0.7)',
                        symbol='star',
                        line=dict(width=2, color='black')
                    ),
                    name='Outliers',
                    text=outliers['title'],
                    hoverinfo='text',
                    hovertemplate='%{text}<br>(OUTLIER)<extra></extra>'
                )
            )
        
        # Style the layout
        fig.update_layout(
            legend_title_text='Clusters',
            hovermode='closest',
            height=600,
            plot_bgcolor='rgba(240, 240, 240, 0.8)',
            margin=dict(t=50, l=25, r=25, b=25)
        )
        
        return fig
    
    def run_analysis(self, rss_url, n_clusters=None, outlier_threshold=1.5):
        """Run the complete analysis pipeline"""
        # Fetch and process
        self.fetch_arxiv_rss(rss_url)
        
        if len(self.papers_df) == 0:
            st.error("No papers found in the RSS feed. Please check the URL and try again.")
            return None
            
        with st.spinner("Analyzing paper content..."):
            self.vectorize_text()
            self.cluster_papers(n_clusters)
            self.detect_outliers(outlier_threshold)
            self.reduce_dimensions()
        
        return self.papers_df

# Function to export to CSV
def download_link(df, filename):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV file</a>'
    return href

# Main Streamlit app
def main():
    st.title("📚 ArXiv Paper Clustering")
    st.markdown("""
    This app fetches papers from ArXiv RSS feeds, clusters them based on their content similarity,
    and identifies unusual papers (outliers) that may be worth exploring further.
    """)
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["📊 Analyze Papers", "ℹ️ About", "❓ Help"])
    
    with tab1:
        st.header("Paper Analysis")
        
        # Initialize app instance
        if 'app' not in st.session_state:
            st.session_state.app = ArxivClusteringApp()
            st.session_state.results = None
        
        # Sidebar inputs
        st.sidebar.header("Input Parameters")
        
        # Example RSS URLs
        example_urls = {
            "Artificial Intelligence": "http://export.arxiv.org/rss/cs.AI",
            "Computer Vision": "http://export.arxiv.org/rss/cs.CV",
            "Natural Language Processing": "http://export.arxiv.org/rss/cs.CL",
            "Machine Learning": "http://export.arxiv.org/rss/cs.LG",
            "Quantum Physics": "http://export.arxiv.org/rss/quant-ph",
            "Astrophysics": "http://export.arxiv.org/rss/astro-ph",
            "High Energy Physics": "http://export.arxiv.org/rss/hep-th",
            "Mathematics": "http://export.arxiv.org/rss/math"
        }
        
        selected_option = st.sidebar.selectbox(
            "Select an ArXiv category:",
            list(example_urls.keys())
        )
        
        # Option to enter custom URL
        custom_url = st.sidebar.text_input(
            "Or enter a custom ArXiv RSS URL:",
            ""
        )
        
        # User parameters
        st.sidebar.subheader("Clustering Parameters")
        
        n_clusters = st.sidebar.slider(
            "Number of clusters (0 = auto):",
            0, 15, 0,
            help="Set to 0 to automatically determine the optimal number of clusters based on data size"
        )
        
        outlier_threshold = st.sidebar.slider(
            "Outlier sensitivity:",
            0.5, 3.0, 1.5, 0.1,
            help="Lower values find more outliers, higher values are more restrictive"
        )
        
        # Process input
        rss_url = custom_url if custom_url else example_urls[selected_option]
        
        # Run button
        if st.sidebar.button("Fetch and Analyze Papers"):
            try:
                with st.spinner("Processing..."):
                    app = st.session_state.app
                    result = app.run_analysis(
                        rss_url=rss_url,
                        n_clusters=None if n_clusters == 0 else n_clusters,
                        outlier_threshold=outlier_threshold
                    )
                    
                    if result is not None:
                        st.session_state.results = True
                        st.success(f"Successfully analyzed {len(app.papers_df)} papers")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
        
        # Display results if available
        if st.session_state.results:
            app = st.session_state.app
            
            # Display visualization
            st.subheader("Paper Clusters Visualization")
            st.plotly_chart(app.create_plotly_visualization(), use_container_width=True)
            
            # Instructions
            st.info("👆 Hover over points to see paper details. Outliers are marked with red stars.")
            
            # Summary statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Papers", len(app.papers_df))
            with col2:
                st.metric("Number of Clusters", len(app.papers_df['cluster'].unique()))
            with col3:
                st.metric("Outliers Found", len(app.outliers))
            
            # Export options
            st.markdown("### Export Results")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(download_link(app.papers_df, "arxiv_papers_clusters.csv"), unsafe_allow_html=True)
            with col2:
                if len(app.outliers) > 0:
                    st.markdown(download_link(app.outliers, "arxiv_outliers.csv"), unsafe_allow_html=True)
            
            # Display outliers
            if len(app.outliers) > 0:
                st.subheader("📌 Outlier Papers")
                
                for _, paper in app.outliers.iterrows():
                    with st.expander(f"{paper['title']}"):
                        st.markdown(f"**Cluster:** {paper['cluster']} | **Similarity:** {paper['similarity_to_center']:.3f}")
                        st.markdown(f"**Categories:** {', '.join(paper['categories'])}")
                        st.markdown(f"**Abstract:** {paper['abstract']}")
                        st.markdown(f"**Link:** [{paper['link']}]({paper['link']})")
            else:
                st.info("No outliers were found with the current threshold.")
            
            # Display all clusters
            st.subheader("📊 Cluster Details")
            
            # Group by cluster
            cluster_groups = app.papers_df.groupby('cluster')
            
            for cluster_id, group in cluster_groups:
                outliers_count = sum(group['is_outlier'])
                
                with st.expander(f"Cluster {cluster_id}: {len(group)} papers ({outliers_count} outliers)"):
                    # Get the most common categories in this cluster
                    all_categories = []
                    for cats in group['categories']:
                        all_categories.extend(cats)
                    
                    if all_categories:
                        category_counts = pd.Series(all_categories).value_counts()
                        top_categories = category_counts.head(5)
                        
                        st.markdown("**Top categories:**")
                        for cat, count in top_categories.items():
                            st.markdown(f"- {cat}: {count} papers")
                    
                    # Show papers in this cluster
                    st.markdown("**Papers in this cluster:**")
                    
                    for _, paper in group.iterrows():
                        title_prefix = "🌟 " if paper['is_outlier'] else ""
                        st.markdown(f"{title_prefix}[{paper['title']}]({paper['link']})")
    
    with tab2:
        st.header("About This App")
        st.markdown("""
        ### How It Works

        This application analyzes ArXiv research papers using natural language processing and machine learning techniques:

        1. **Data Collection**: Fetches the latest papers from ArXiv RSS feeds
        2. **Text Processing**: Converts paper titles and abstracts into numerical vectors using TF-IDF
        3. **Clustering**: Groups similar papers together using the K-means algorithm
        4. **Outlier Detection**: Identifies papers that don't fit well with others in their cluster
        5. **Visualization**: Displays the relationships between papers in a 2D interactive plot

        ### Why Look for Outliers?

        Outlier papers often represent:
        - Novel research directions
        - Interdisciplinary work
        - Unusual applications of common techniques
        - Papers that may have been miscategorized

        By identifying these papers, researchers can discover interesting work they might otherwise miss.
        """)
        
        st.subheader("Technical Details")
        st.markdown("""
        - **Vectorization**: TF-IDF (Term Frequency-Inverse Document Frequency)
        - **Dimensionality Reduction**: Principal Component Analysis (PCA)
        - **Clustering Algorithm**: K-means
        - **Outlier Detection**: Statistical distance from cluster centers
        - **Visualization**: Interactive 2D scatter plot with Plotly
        """)
    
    with tab3:
        st.header("Help & FAQ")
        
        with st.expander("How do I use this app?"):
            st.markdown("""
            1. Select an ArXiv category from the dropdown in the sidebar, or enter a custom RSS URL
            2. Adjust the clustering parameters if needed
            3. Click "Fetch and Analyze Papers"
            4. Explore the visualization, outliers, and cluster details
            """)
            
        with st.expander("What are the clustering parameters?"):
            st.markdown("""
            - **Number of clusters**: How many groups to divide the papers into. Set to 0 for automatic selection.
            - **Outlier sensitivity**: Controls how extreme a paper needs to be to be considered an outlier. Lower values find more outliers.
            """)
            
        with st.expander("How can I find a custom ArXiv RSS URL?"):
            st.markdown("""
            ArXiv provides RSS feeds for all categories and subcategories using this format:
            ```
            http://export.arxiv.org/rss/[category]
            ```
            
            For example:
            - Computer Science: http://export.arxiv.org/rss/cs
            - Machine Learning: http://export.arxiv.org/rss/cs.LG
            - Mathematics: http://export.arxiv.org/rss/math
            
            For a complete list of categories, visit: [arXiv.org](https://arxiv.org/)
            """)
            
        with st.expander("Can I export the results?"):
            st.markdown("""
            Yes! After analysis is complete, you'll see download links to export:
            - All papers with their cluster assignments
            - Just the outlier papers
            
            These export as CSV files that you can open in spreadsheet software or use for further analysis.
            """)
            
        with st.expander("Why are some papers considered outliers?"):
            st.markdown("""
            A paper is marked as an outlier if its content is significantly different from other papers in its assigned cluster.
            
            Specifically, we measure how similar each paper is to the center of its cluster. Papers with similarity scores that fall below a threshold (based on the mean and standard deviation of the cluster) are flagged as outliers.
            
            This could indicate papers that:
            - Combine multiple research areas
            - Use unusual terminology
            - Apply techniques from one field to problems in another
            - Focus on niche applications
            """)

# Run the app
if __name__ == "__main__":
    main()
