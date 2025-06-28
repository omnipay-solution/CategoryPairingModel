from fastapi import FastAPI, Query
from pydantic import BaseModel
from gensim.models import Word2Vec
import pandas as pd
import pyodbc
import os

app = FastAPI()

MODEL_PATH = "category_pairing_model.model"

# === Function to fetch data from SQL Server ===
def fetch_data_from_sql():
    connection_string = (
        "DRIVER={ODBC Driver 17 for SQL Server};"
        "SERVER=omnipos-sql-server.database.windows.net,1433;"
        "DATABASE=OmniPOS;"
        "UID=omnipos;"
        "PWD=Bik1984att@;"
        "Encrypt=yes;"
        "TrustServerCertificate=no;"
        "Connection Timeout=30;"
    )
    conn = pyodbc.connect(connection_string)
    query = "SELECT Name, Category FROM Items"
    df = pd.read_sql(query, conn)
    conn.close()
    return df

# === Train a new model from SQL data ===
def train_model(basket_size=3):
    df = fetch_data_from_sql()

    print(" Sample rows fetched from SQL:")
    print(df.head())
    print(" Rows fetched from SQL:", len(df))

    if df.empty:
        raise ValueError(" No data fetched from SQL.")

    df = df.dropna(subset=['Category'])
    df['Category'] = df['Category'].str.upper()

    recipes = []
    for i in range(0, len(df), basket_size):
        basket = df.iloc[i:i + basket_size]['Category'].unique().tolist()
        if len(basket) > 1:
            recipes.append(basket)

    print(" Total recipes created:", len(recipes))

    if not recipes:
        raise ValueError(" No valid category baskets found for training.")

    model = Word2Vec(
        sentences=recipes,
        vector_size=50,
        window=2,
        min_count=1,
        sg=1,
        workers=4,
        epochs=100
    )
    model.save(MODEL_PATH)
    return model

# === Load Existing or Train New Model ===
if os.path.exists(MODEL_PATH):
    print(" Loading existing model...")
    model = Word2Vec.load(MODEL_PATH)
else:
    print(" No existing model found. Training...")
    model = train_model()

# === Define POST Request Body ===
class CategoryRequest(BaseModel):
    category: str

# === API Endpoint ===
@app.post("/recommend")
def recommend(request: CategoryRequest):
    category = request.category.upper()
    if category in model.wv:
        recommendations = model.wv.most_similar(category, topn=3)
        return {
            "base_category": category,
            "recommendations": [
                {"category": rec, "score": round(score, 3)} for rec, score in recommendations
            ]
        }
    return {"error": f"Category '{category}' not found in model"}
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
