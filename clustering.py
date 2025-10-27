import pandas as pd
import ast
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, FastICA
from sklearn.manifold import TSNE, MDS
from umap import UMAP
import plotly.express as px
from tqdm import tqdm

def safe_convert_embedding(embedding_str):
    try:
        if isinstance(embedding_str, list):
            return embedding_str
        
        if isinstance(embedding_str, str):
            cleaned_str = embedding_str.strip()
            if cleaned_str.startswith('[') and cleaned_str.endswith(']'):
                return ast.literal_eval(cleaned_str)
        
        return None
    except (ValueError, SyntaxError):
        try:
            if isinstance(embedding_str, str):
                cleaned = embedding_str.replace('[', '').replace(']', '').strip()
                return [float(x.strip()) for x in cleaned.split(',') if x.strip()]
        except:
            return None
        return None

def clustering(df, input_algoritm, info_row, count_clusters=5):
    progress_bar = tqdm(total=100, desc="Clustering progress")
    
    df['embedding'] = df['embedding'].apply(safe_convert_embedding)
    progress_bar.update(10)
    
    df = df.dropna(subset=['embedding'])
    
    embedding_lengths = df['embedding'].apply(len)
    if embedding_lengths.nunique() > 1:
        print(f"Предупреждение: эмбеддинги разной длины: {embedding_lengths.unique()}")
        most_common_length = embedding_lengths.mode()[0]
        df = df[df['embedding'].apply(len) == most_common_length]
    
    embeddings = pd.DataFrame(df['embedding'].tolist())
    progress_bar.update(10)
    
    kmeans = KMeans(n_clusters=count_clusters, random_state=42)
    df['cluster'] = kmeans.fit_predict(embeddings)
    progress_bar.update(20)
    
    algorithms = {
        1: (FastICA(n_components=2, random_state=42), 'ICA'),
        2: (MDS(n_components=2, random_state=42), 'MDS'), 
        3: (PCA(n_components=2, random_state=42), 'PCA'),
        4: (TSNE(n_components=2, random_state=42), 'T-SNE'),
        5: (UMAP(n_components=2, random_state=42), 'UMAP')
    }
    
    alg, name = algorithms[input_algoritm]
    progress_bar.update(10)
    
    reduced_embeddings = alg.fit_transform(embeddings)
    progress_bar.update(20)
    
    result_df = pd.DataFrame(data=reduced_embeddings, columns=['x', 'y'])
    progress_bar.update(10)
    
    result_df[info_row] = df[info_row].values
    result_df['cluster'] = df['cluster'].values
    progress_bar.update(10)
    
    scientific_palette = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
        '#393b79', '#637939', '#8c6d31', '#843c39', '#7b4173',
        '#5254a3', '#6b6ecf', '#9c9ede', '#637939', '#8ca252',
        '#b5cf6b', '#cedb9c', '#8c6d31', '#bd9e39', '#e7ba52'
    ]
    
    fig = px.scatter(
        result_df,
        x='x',
        y='y', 
        color='cluster',
        hover_name=info_row,
        title=name,
        width=1200,
        height=800,
        color_discrete_sequence=scientific_palette[:count_clusters]
    )
    
    fig.update_layout(
        plot_bgcolor='rgba(255, 255, 255, 1)',
        paper_bgcolor='rgba(255, 255, 255, 1)',
        font=dict(family="Arial, sans-serif", size=12, color='#2C3E50'),
        title=dict(
            x=0.5,
            font=dict(size=24, color='#2C3E50', family="Arial, sans-serif")
        ),
        hoverlabel=dict(
            bgcolor='rgba(255, 255, 255, 0.95)',
            font_size=12,
            font_family="Arial, sans-serif",
            bordercolor='#BDC3C7'
        ),
        xaxis=dict(
            gridcolor='#ECF0F1',
            linecolor='#BDC3C7'
        ),
        yaxis=dict(
            gridcolor='#ECF0F1', 
            linecolor='#BDC3C7'
        )
    )
    
    fig.update_traces(
        marker=dict(
            size=10,
            opacity=0.8,
            line=dict(width=1, color='rgba(255, 255, 255, 0.8)')
        ),
        selector=dict(mode='markers')
    )
    
    fig.write_html(f"{name}.html")
    progress_bar.update(10)
    
    result_df.to_csv(f'{name}.csv', index=False)
    progress_bar.update(10)
    
    progress_bar.close()
    
    return result_df