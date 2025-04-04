from fastapi import FastAPI
import pandas as pd

app = FastAPI()

@app.get("/scraped-data/")
def get_scraped_data():
    try:
        df = pd.read_csv("Data/research_titles_cleaned.csv")
        return df.to_dict(orient="records")
    except Exception as e:
        return {"error": str(e)}
