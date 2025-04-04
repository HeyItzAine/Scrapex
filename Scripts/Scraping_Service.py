from fastapi import FastAPI, BackgroundTasks
import os

app = FastAPI()

# Function to trigger scraping
def run_scraper():
    os.system("python Scripts/Scrapex.py")  # Calls your existing Scrapex script

@app.post("/start-scraping/")
def start_scraping(background_tasks: BackgroundTasks):
    background_tasks.add_task(run_scraper)
    return {"message": "Scraping started in the background!"}
