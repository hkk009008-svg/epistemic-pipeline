from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from api.routes import router

app = FastAPI(title="Epistemic Verification Pipeline", version="2.0.0")
app.include_router(router)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    from fastapi import HTTPException
    if isinstance(exc, HTTPException):
        return JSONResponse(status_code=exc.status_code, content={"error": True, "detail": exc.detail})
    return JSONResponse(status_code=500, content={"error": True, "detail": f"Internal server error: {str(exc)[:200]}"})
