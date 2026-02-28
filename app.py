"""FastAPI application entry point for the Epistemic Verification Pipeline."""
from fastapi import FastAPI
from api.routes import router

app = FastAPI(title="Epistemic Verification Pipeline")
app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    import config
    uvicorn.run(app, host="0.0.0.0", port=config.PORT)
