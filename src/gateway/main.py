import os
from pathlib import Path

import httpx
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

TARGET_BASE_URL = os.getenv("TARGET_BASE_URL", "http://localhost:8001").rstrip("/")

app = FastAPI(title="Proxy Gateway")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = httpx.AsyncClient(base_url=TARGET_BASE_URL, timeout=30.0)


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"])
async def proxy(request: Request, path: str) -> Response:
    body = await request.body()
    params = dict(request.query_params)
    headers = dict(request.headers)
    headers.pop("host", None)

    resp = await client.request(
        method=request.method,
        url=f"/{path}",
        params=params,
        headers=headers,
        content=body,
    )

    return Response(
        content=resp.content,
        status_code=resp.status_code,
        headers=dict(resp.headers),
    )


@app.on_event("shutdown")
async def shutdown():
    await client.aclose()


if __name__ == "__main__":
    port = int(os.getenv("PROXY_PORT", "8002"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
