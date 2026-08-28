from fastapi import FastAPI

from backend.app.api.routes import router

app = FastAPI(
    title="NER-Nav API",
    version="0.1.0",
    description=(
        "AI-Based Smart Logistics and Accessibility Intelligence "
        "Platform for the North Eastern Region."
    ),
)
app.include_router(router)


@app.get("/health", tags=["system"])
async def health() -> dict[str, str]:
    return {"status": "ok"}
