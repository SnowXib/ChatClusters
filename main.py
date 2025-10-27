import pandas as pd
import asyncio

from embedding_gen import process_embeddings
from parser import chat_parsing
from defs import API_KEY, TARGET_ROW
from clustering import clustering

def main():
    chat_data = chat_parsing('result.json')
    df = pd.DataFrame({TARGET_ROW: chat_data['user_1'] + chat_data['user_2']})

    df.to_csv('combined_dataset.csv', index=False)

    df = df.dropna(subset=[TARGET_ROW])
    df_with_embeddings = asyncio.run(process_embeddings(df, API_KEY, TARGET_ROW))
    df_with_embeddings = df_with_embeddings.dropna(subset=['embedding'])

    df_with_embeddings.to_csv('combined_dataset_with_embeddings.csv', index=False)

    result = clustering(
        df=df_with_embeddings,
        info_row=TARGET_ROW,
        input_algoritm=2,  # 1-ICA, 2-MDS, 3-PCA, 4-TSNE, 5-UMAP
        count_clusters=50,   
    )

if __name__ == "__main__":
    main()