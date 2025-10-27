import pandas as pd
import asyncio
import aiohttp
import json
from aiohttp import ClientSession
import nest_asyncio
from tqdm.asyncio import tqdm_asyncio

async def get_embedding_async(text: str, api_key: str, session: ClientSession, model_name: str = 'text-embedding-3-small') -> list:
    try:
        clean_text = text.replace("\n", " ") if isinstance(text, str) else ""
        url = "https://api.proxyapi.ru/openai/v1/embeddings"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "input": [clean_text],
            "model": model_name
        }
        
        async with session.post(url, headers=headers, json=payload) as response:
            if response.status == 200:
                data = await response.json()
                return data['data'][0]['embedding']
            else:
                print(f"Error: {response.status} - {await response.text()}")
                return None
    except Exception as e:
        print(f"Exception: {e}")
        return None
    

async def process_embeddings(df: pd.DataFrame, api_key: str, row_for_embedding: str, batch_size: int = 100) -> pd.DataFrame:
    if 'embedding' in df.columns:
        nan_mask = df['embedding'].isna()
        nan_count = nan_mask.sum()
        
        if nan_count == 0:
            print("Нет NaN значений, возвращаем исходный датафрейм")
            return df
            
        print(f"Обрабатываем {nan_count} NaN значений")
        nan_df = df[nan_mask].copy()
        
        async with ClientSession() as session:
            tasks = []
            for i, row in nan_df.iterrows():
                text = row[row_for_embedding]
                tasks.append(get_embedding_async(text, api_key, session))
            
            embeddings = []
            for i in range(0, len(tasks), batch_size):
                batch_tasks = tasks[i:i+batch_size]
                batch_results = await tqdm_asyncio.gather(*batch_tasks, desc=f"Processing NaN batch {i//batch_size + 1}")
                embeddings.extend(batch_results)
        
        df.loc[nan_mask, 'embedding'] = embeddings
        return df
    
    else:
        async with ClientSession() as session:
            tasks = []
            for i, row in df.iterrows():
                text = row[row_for_embedding]
                tasks.append(get_embedding_async(text, api_key, session))
            
            embeddings = []
            for i in range(0, len(tasks), batch_size):
                batch_tasks = tasks[i:i+batch_size]
                batch_results = await tqdm_asyncio.gather(*batch_tasks, desc=f"Processing batch {i//batch_size + 1}")
                embeddings.extend(batch_results)
        
        df['embedding'] = embeddings
        return df