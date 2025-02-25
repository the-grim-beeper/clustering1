# app.py - Enhanced ArXiv Paper Clustering Web App
import streamlit as st
import feedparser
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, SpectralClustering
from sklearn.decomposition import PCA, TruncatedSVD, NMF
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
import plotly.express as px
import plotly.graph_objects as go
import nltk
from nltk.corpus import stopwords
import time
import base64

# Handle optional dependencies
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

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
        self.feature_matrix = None
        self.embedding = None
        
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
                
                # Get authors
                try:
                    authors = [author.name for author in entry.authors]
                except (AttributeError, KeyError):
                    authors = []
                
                # Add to papers list
                papers.append({
                    'id': paper_id,
                    'title': title,
                    'abstract': summary,
                    'categories': categories,
                    'authors': authors,
                    'link': entry.link
                })
            
            # Convert to DataFrame
            self.papers_df = pd.DataFrame(papers)
            return self.papers_df
    
    def extract_features(self, feature_method="tfidf", feature_source="title_abstract", max_features=1000):
        """Extract features from paper text using various methods"""
        if self.papers_df is None or len(self.papers_df) == 0:
            raise ValueError("No papers loaded. Fetch papers first.")
        
        # Prepare text based on selected source
        if feature_source == "title_only":
            self.papers_df['text'] = self.papers_df['title']
        elif feature_source == "abstract_only":
            self.papers_df['text'] = self.papers_df['abstract']
        elif feature_source == "title_abstract":
            self.papers_df['text'] = self.papers_df['title'] + ' ' + self.papers_df['abstract']
        elif feature_source == "categories":
            self.papers_df['text'] = self.papers_df['categories'].apply(lambda x: ' '.join(x))
        elif feature_source == "authors":
            self.papers_df['text'] = self.papers_df['authors'].apply(lambda x: ' '.join(x) if x else '')
        
        # Choose vectorization method
        if feature_method == "tfidf":
            self.vectorizer = TfidfVectorizer(
                max_features=max_features,
                stop_words=stopwords.words('english'),
                min_df=2,
                ngram_range=(1, 2)
            )
        elif feature_method == "bow":
            self.vectorizer = CountVectorizer(
                max_features=max_features,
                stop_words=stopwords.words('english'),
                min_df=2,
                ngram_range=(1, 2)
            )
        
        # Apply vectorization
        self.feature_matrix = self.vectorizer.fit_transform(self.papers_df['text'])
        
        return self.feature_matrix
    
    def cluster_papers(self, clustering_method="kmeans", n_clusters=None, eps=0.5, min_samples=5):
        """Cluster papers using various algorithms"""
        if self.feature_matrix is None:
            raise ValueError("Features not extracted yet. Extract features first.")
        
        # Determine number of clusters if not specified
        if n_clusters is None:
            n_clusters = min(max(2, int(np.sqrt(len(self.papers_df)))), 10)
        
        # Apply clustering algorithm
        if clustering_method == "kmeans":
            clusterer = KMeans(n_clusters=n_clusters, random_state=42)
            self.papers_df['cluster'] = clusterer.fit_predict(self.feature_matrix)
            
            # Calculate distance to cluster center (for outlier detection)
            cluster_centers = clusterer.cluster_centers_
            
            # For each paper, calculate similarity to its cluster center
            similarities = []
            for i, paper_idx in enumerate(range(len(self.papers_df))):
                paper_vector = self.feature_matrix[paper_idx]
                cluster_id = self.papers_df.iloc[paper_idx]['cluster']
                center_vector = cluster_centers[cluster_id]
                
                # Use cosine similarity (higher means more similar)
                sim = cosine_similarity(paper_vector, center_vector.reshape(1, -1))[0][0]
                similarities.append(sim)
            
            self.papers_df['similarity_to_center'] = similarities
            
        elif clustering_method == "hierarchical":
            clusterer = AgglomerativeClustering(n_clusters=n_clusters)
            self.papers_df['cluster'] = clusterer.fit_predict(self.feature_matrix.toarray())
            
            # For hierarchical clustering, we'll use distance to cluster centroid as similarity measure
            # Calculate centroids for each cluster
            cluster_centroids = {}
            for cluster_id in range(n_clusters):
                mask = self.papers_df['cluster'] == cluster_id
                if sum(mask) > 0:  # Ensure cluster is not empty
                    cluster_papers = self.feature_matrix[mask].toarray()
                    centroid = np.mean(cluster_papers, axis=0)
                    cluster_centroids[cluster_id] = centroid
            
            # Calculate similarity to centroid
            similarities = []
            for i, row in self.papers_df.iterrows():
                cluster_id = row['cluster']
                paper_vector = self.feature_matrix[i]
                centroid = cluster_centroids[cluster_id]
                sim = cosine_similarity(paper_vector, centroid.reshape(1, -1))[0][0]
                similarities.append(sim)
            
            self.papers_df['similarity_to_center'] = similarities
            
        elif clustering_method == "dbscan":
            clusterer = DBSCAN(eps=eps, min_samples=min_samples)
            self.papers_df['cluster'] = clusterer.fit_predict(self.feature_matrix.toarray())
            
            # For DBSCAN, -1 indicates outliers
            self.papers_df['is_outlier'] = self.papers_df['cluster'] == -1
            
            # Calculate distance to nearest core point for non-outliers
            core_samples_mask = np.zeros_like(clusterer.labels_, dtype=bool)
            core_samples_mask[clusterer.core_sample_indices_] = True
            
            similarities = []
            for i, row in self.papers_df.iterrows():
                if row['is_outlier']:
                    # Outliers get a similarity of 0
                    similarities.append(0)
                else:
                    # For non-outliers, find distance to nearest core point in same cluster
                    cluster_id = row['cluster']
                    core_points = np.where((clusterer.labels_ == cluster_id) & core_samples_mask)[0]
                    paper_vector = self.feature_matrix[i].toarray().flatten()
                    
                    max_sim = 0
                    for core_idx in core_points:
                        core_vector = self.feature_matrix[core_idx].toarray().flatten()
                        sim = cosine_similarity(paper_vector.reshape(1, -1), core_vector.reshape(1, -1))[0][0]
                        max_sim = max(max_sim, sim)
                    
                    similarities.append(max_sim)
            
            self.papers_df['similarity_to_center'] = similarities
            
        elif clustering_method == "spectral":
            clusterer = SpectralClustering(n_clusters=n_clusters, random_state=42, affinity='nearest_neighbors')
            self.papers_df['cluster'] = clusterer.fit_predict(self.feature_matrix)
            
            # For spectral clustering, we can use a similar approach to kmeans
            # Calculate cluster centroids
            centroids = {}
            for cluster_id in range(n_clusters):
                mask = self.papers_df['cluster'] == cluster_id
                if sum(mask) > 0:
                    cluster_papers = self.feature_matrix[mask].toarray()
                    centroid = np.mean(cluster_papers, axis=0)
                    centroids[cluster_id] = centroid
            
            # Calculate similarity to centroid
            similarities = []
            for i, row in self.papers_df.iterrows():
                cluster_id = row['cluster']
                paper_vector = self.feature_matrix[i]
                centroid = centroids[cluster_id]
                sim = cosine_similarity(paper_vector, centroid.reshape(1, -1))[0][0]
                similarities.append(sim)
            
            self.papers_df['similarity_to_center'] = similarities
        
        return self.papers_df
    
    def detect_outliers(self, threshold=1.5, method="statistical"):
        """Identify outliers based on similarity to cluster center"""
        if 'cluster' not in self.papers_df.columns:
            raise ValueError("Papers not clustered yet. Run cluster_papers first.")
        
        # For DBSCAN, outliers are already identified
        if 'is_outlier' in self.papers_df.columns:
            self.outliers = self.papers_df[self.papers_df['is_outlier']]
            return self.outliers
        
        # Find outliers based on method
        if method == "statistical":
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
                
        elif method == "percentile":
            # Use percentile-based outlier detection
            outliers = []
            
            for cluster_id in self.papers_df['cluster'].unique():
                cluster_papers = self.papers_df[self.papers_df['cluster'] == cluster_id]
                
                if len(cluster_papers) <= 3:  # Need enough papers for percentile
                    continue
                
                # Get lowest 5% by similarity
                percentile_threshold = np.percentile(cluster_papers['similarity_to_center'], 5)
                cluster_outliers = cluster_papers[cluster_papers['similarity_to_center'] < percentile_threshold]
                outliers.append(cluster_outliers)
            
            # Combine all outliers
            if outliers:
                self.outliers = pd.concat(outliers)
                self.papers_df['is_outlier'] = self.papers_df.index.isin(self.outliers.index)
            else:
                self.outliers = pd.DataFrame()
                self.papers_df['is_outlier'] = False
                
        return self.outliers
    
    def reduce_dimensions(self, method="pca", n_components=2):
        """Reduce dimensionality for visualization using various methods"""
        if self.feature_matrix is None:
            raise ValueError("Features not extracted yet. Extract features first.")
        
        # Apply dimensionality reduction
        if method == "pca":
            reducer = PCA(n_components=n_components)
            self.embedding = reducer.fit_transform(self.feature_matrix.toarray())
            
        elif method == "tsne":
            reducer = TSNE(n_components=n_components, random_state=42, perplexity=min(30, len(self.papers_df)-1))
            self.embedding = reducer.fit_transform(self.feature_matrix.toarray())
            
        elif method == "svd":
            reducer = TruncatedSVD(n_components=n_components)
            self.embedding = reducer.fit_transform(self.feature_matrix)
            
        elif method == "nmf":
            reducer = NMF(n_components=n_components, random_state=42)
            self.embedding = reducer.fit_transform(self.feature_matrix)
            
        elif method == "umap":
            if UMAP_AVAILABLE:
                reducer = umap.UMAP(n_components=n_components, random_state=42)
                self.embedding = reducer.fit_transform(self.feature_matrix.toarray())
            else:
                st.warning("UMAP is not available. Please install it with 'pip install umap-learn'. Falling back to PCA.")
                reducer = PCA(n_components=n_components)
                self.embedding = reducer.fit_transform(self.feature_matrix.toarray())
        
        # Add coordinates to DataFrame
        self.papers_df['x'] = self.embedding[:, 0]
        self.papers_df['y'] = self.embedding[:, 1]
        
        return self.embedding
    
    def create_plotly_visualization(self):
        """Create Plotly visualization of clusters"""
        if 'x' not in self.papers_df.columns:
            self.reduce_dimensions()
        
        # Prepare hover text
        self.papers_df['hover_text'] = self.papers_df.apply(
            lambda row: f"<b>{row['title']}</b><br><br>" + 
                       f"Categories: {', '.join(row['categories'])}<br>" +
                       (f"Similarity: {row['similarity_to_center']:.3f}<br><br>" if 'similarity_to_center' in row else "") +
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
                'similarity_to_center': ':.3f' if 'similarity_to_center' in self.papers_df.columns else False,
                'is_outlier': True if 'is_outlier' in self.papers_df.columns else False,
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
        
        # Add outlines to outliers if they exist
        if 'is_outlier' in self.papers_df.columns:
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
    
    def run_analysis(self, rss_url, feature_params, clustering_params, dimension_params, outlier_params):
        """Run the complete analysis pipeline with specified parameters"""
        # Fetch data
        self.fetch_arxiv_rss(rss_url)
        
        if len(self.papers_df) == 0:
            st.error("No papers found in the RSS feed. Please check the URL and try again.")
            return None
            
        with st.spinner("Analyzing paper content..."):
            # Extract features
            self.extract_features(
                feature_method=feature_params['method'],
                feature_source=feature_params['source'],
                max_features=feature_params['max_features']
            )
            
            # Cluster papers
            if clustering_params['method'] == 'dbscan':
                self.cluster_papers(
                    clustering_method=clustering_params['method'],
                    eps=clustering_params['eps'],
                    min_samples=clustering_params['min_samples']
                )
            else:
                self.cluster_papers(
                    clustering_method=clustering_params['method'],
                    n_clusters=clustering_params['n_clusters']
                )
            
            # Detect outliers (if not DBSCAN which already marks outliers)
            if clustering_params['method'] != 'dbscan':
                self.detect_outliers(
                    threshold=outlier_params['threshold'],
                    method=outlier_params['method']
                )
            
            # Reduce dimensions for visualization
            self.reduce_dimensions(
                method=dimension_params['method'],
                n_components=2
            )
        
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
        
        # Feature extraction parameters
        st.sidebar.subheader("Feature Extraction")
        
        feature_method = st.sidebar.selectbox(
            "Feature extraction method:",
            ["tfidf", "bow"],
            format_func=lambda x: {
                "tfidf": "TF-IDF Vectorization",
                "bow": "Bag of Words"
            }[x],
            help="TF-IDF works well for most cases. Bag of Words is simpler but may be less effective."
        )
        
        feature_source = st.sidebar.selectbox(
            "Text source for features:",
            ["title_abstract", "title_only", "abstract_only", "categories", "authors"],
            format_func=lambda x: {
                "title_abstract": "Title + Abstract",
                "title_only": "Title Only",
                "abstract_only": "Abstract Only",
                "categories": "ArXiv Categories",
                "authors": "Authors"
            }[x],
            help="Select what text to use for clustering. Title+Abstract is recommended."
        )
        
        max_features = st.sidebar.slider(
            "Maximum features:",
            100, 2000, 1000, 100,
            help="Maximum number of terms to include in the vocabulary."
        )
        
        # Clustering parameters
        st.sidebar.subheader("Clustering Method")
        
        clustering_method = st.sidebar.selectbox(
            "Clustering algorithm:",
            ["kmeans", "hierarchical", "dbscan", "spectral"],
            format_func=lambda x: {
                "kmeans": "K-Means",
                "hierarchical": "Hierarchical",
                "dbscan": "DBSCAN",
                "spectral": "Spectral"
            }[x],
            help="Different algorithms have different strengths. K-Means is fast, DBSCAN is good at finding outliers."
        )
        
        # Show appropriate parameters based on clustering method
        if clustering_method == "dbscan":
            eps = st.sidebar.slider(
                "DBSCAN epsilon:",
                0.1, 1.0, 0.5, 0.05,
                help="Maximum distance between samples to be considered in the same neighborhood."
            )
            
            min_samples = st.sidebar.slider(
                "DBSCAN min_samples:",
                2, 10, 5, 1,
                help="Minimum number of samples in a neighborhood to form a core point."
            )
            
            n_clusters = None
        else:
            n_clusters = st.sidebar.slider(
                "Number of clusters (0 = auto):",
                0, 15, 0,
                help="Set to 0 to automatically determine the optimal number of clusters based on data size."
            )
            
            eps = 0.5
            min_samples = 5
        
        # Dimensionality reduction parameters
        st.sidebar.subheader("Visualization")
        
        dimension_method = st.sidebar.selectbox(
            "Dimensionality reduction method:",
            ["pca", "tsne", "svd", "nmf", "umap"],
            format_func=lambda x: {
                "pca": "PCA",
                "tsne": "t-SNE",
                "svd": "Truncated SVD",
                "nmf": "Non-Negative Matrix Factorization",
                "umap": "UMAP"
            }[x],
            help="Method to reduce dimensions for visualization. t-SNE and UMAP often produce better visualizations but are slower."
        )
        
        # Outlier detection parameters (not shown for DBSCAN)
        if clustering_method != "dbscan":
            st.sidebar.subheader("Outlier Detection")
            
            outlier_method = st.sidebar.selectbox(
                "Outlier detection method:",
                ["statistical", "percentile"],
                format_func=lambda x: {
                    "statistical": "Statistical (mean/std)",
                    "percentile": "Percentile-based"
                }[x],
                help="Method to identify outliers. Statistical uses distance from mean, percentile uses the bottom N%."
            )
            
            outlier_threshold = st.sidebar.slider(
                "Outlier sensitivity:",
                0.5, 3.0, 1.5, 0.1,
                help="Lower values find more outliers, higher values are more restrictive."
            )
        else:
            outlier_method = "dbscan"
            outlier_threshold = 0
        
        # Process input
        rss_url = custom_url if custom_url else example_urls[selected_option]
        
        # Package parameters
        feature_params = {
            'method': feature_method,
            'source': feature_source,
            'max_features': max_features
        }
        
        clustering_params = {
            'method': clustering_method,
            'n_clusters': None if n_clusters == 0 else n_clusters,
            'eps': eps,
            'min_samples': min_samples
        }
        
        dimension_params = {
            'method': dimension_method
        }
        
        outlier_params = {
            'method': outlier_method,
            'threshold': outlier_threshold
        }
        
        # Run button
        if st.sidebar.button("Fetch and Analyze Papers"):
            try:
                with st.spinner("Processing..."):
                    app = st.session_state.app
                    result = app.run_analysis(
                        rss_url=rss_url,
                        feature_params=feature_params,
                        clustering_params=clustering_params,
                        dimension_params=dimension_params,
                        outlier_params=outlier_params
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
            st.info("👆 Hover over points to see paper details. Outliers are marked with red stars (if any).")
            
            # Summary statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Papers", len(app.papers_df))
            with col2:
                st.metric("Number of Clusters", len(app.papers_df['cluster'].unique()))
            with col3:
                if 'is_outlier' in app.papers_df.columns:
                    outlier_count = app.papers_df['is_outlier'].sum()
                    st.metric("Outliers Found", outlier_count)
                else:
                    st.metric("Outliers Found", 0)
            
            # Export options
            st.markdown("### Export Results")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(download_link(app.papers_df, "arxiv_papers_clusters.csv"), unsafe_allow_html=True)
            with col2:
                if hasattr(app, 'outliers') and app.outliers is not None and len(app.outliers) > 0:
                    st.markdown(download_link(app.outliers, "arxiv_outliers.csv"), unsafe_allow_html=True)
            
            # Display outliers
            if hasattr(app, 'outliers') and app.outliers is not None and len(app.outliers) > 0:
                st.subheader("📌 Outlier Papers")
                
                for _, paper in app.outliers.iterrows():
                    with st.expander(f"{paper['title']}"):
                        if 'similarity_to_center' in paper:
                            st.markdown(f"**Cluster:** {paper['cluster']} | **Similarity:** {paper['similarity_to_center']:.3f}")
                        else:
                            st.markdown(f"**Cluster:** {paper['cluster']}")
                        st.markdown(f"**Categories:** {', '.join(paper['categories'])}")
                        st.markdown(f"**Abstract:** {paper['abstract']}")
                        st.markdown(f"**Link:** [{paper['link']}]({paper['link']})")
            else:
                st.info("No outliers were found with the current settings.")
            
            # Display all clusters
            st.subheader("📊 Cluster Details")
            
            # Group by cluster
            cluster_groups = app.papers_df.groupby('cluster')
            
            for cluster_id, group in cluster_groups:
                if 'is_outlier' in group:
                    outliers_count = sum(group['is_outlier'])
                else:
                    outliers_count = 0
                
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
