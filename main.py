import asyncio
from fastapi import FastAPI
import uvicorn
from .scanner import scan_all_pairs

app = FastAPI()

@app.get("/")
def read_root():
    return {"status": "bot active"}

async def run_bot_loop():
    while True:
        try:
            scan_all_pairs()
        except Exception as e:
            print(f"Scan error: {e}")
        await asyncio.sleep(60)

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(run_bot_loop())

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)
