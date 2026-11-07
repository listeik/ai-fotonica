from __future__ import annotations

import base64
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal

import anyio
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from doe.models import load_model
from doe.physics import FresnelPropagator, gray_to_png, png_to_gray
from doe.synthesis import synthesize
from doe.training.dataset import synthetic_targets

LOGGER = logging.getLogger(__name__)
MAX_FILE_BYTES = 64 * 1024
MAX_BODY_BYTES = 4 * MAX_FILE_BYTES + 8 * 1024
MAX_ITERATIONS = 2000
Method = Literal["classical", "neural", "hybrid"]
DEMOS = [
    {"id": 0, "name": "Набор 01"},
    {"id": 1, "name": "Набор 02"},
    {"id": 2, "name": "Набор 03"},
]


class LimitSynthesisBody:
    """Bound uploads before multipart parsing, including chunked requests."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if (
            scope["type"] != "http"
            or scope["method"] != "POST"
            or scope["path"].rstrip("/") != "/api/synthesize"
        ):
            await self.app(scope, receive, send)
            return

        too_large = JSONResponse({"detail": "Request exceeds the upload limit."}, 413)
        content_length = dict(scope["headers"]).get(b"content-length", b"")
        if content_length.isdigit():
            significant_digits = content_length.lstrip(b"0") or b"0"
            if len(significant_digits) > 6 or int(significant_digits) > MAX_BODY_BYTES:
                await too_large(scope, receive, send)
                return

        body = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            chunk = message.get("body", b"")
            if len(body) + len(chunk) > MAX_BODY_BYTES:
                await too_large(scope, receive, send)
                return
            body.extend(chunk)
            if not message.get("more_body", False):
                break

        replayed = False

        async def replay() -> Message:
            nonlocal replayed
            if not replayed:
                replayed = True
                buffered = bytes(body)
                body.clear()
                return {"type": "http.request", "body": buffered, "more_body": False}
            return await receive()

        await self.app(scope, replay, send)


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def create_app(
    checkpoint: str | Path | None = None,
    device: str | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        configured_device = device or os.getenv("DOE_DEVICE", "auto")
        if configured_device == "auto":
            configured_device = "cuda" if torch.cuda.is_available() else "cpu"
        app.state.device = torch.device(configured_device)
        if app.state.device.type == "cpu":
            torch.set_num_threads(int(os.getenv("DOE_CPU_THREADS", "4")))
        app.state.propagator = FresnelPropagator().to(app.state.device)
        app.state.limiter = anyio.CapacityLimiter(1)
        app.state.model = None
        app.state.model_version = None
        app.state.model_status = "missing"
        weights = Path(
            checkpoint or os.getenv("DOE_CHECKPOINT", "artifacts/doe-unet.pt")
        )
        if weights.is_file():
            try:
                app.state.model, app.state.model_version = load_model(
                    weights, app.state.device
                )
                app.state.model_status = "ready"
            except Exception:
                LOGGER.exception("Could not load the configured model checkpoint")
                app.state.model_status = "load_error"
        yield

    app = FastAPI(
        title="AI Photonica · DOE synthesis",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.add_middleware(LimitSynthesisBody)
    origins = os.getenv(
        "DOE_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[origin.strip() for origin in origins.split(",") if origin.strip()],
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    def optics() -> dict:
        propagator = app.state.propagator
        return {
            "size_px": propagator.size,
            "wavelength_nm": round(propagator.wavelength_m * 1e9, 4),
            "pixel_pitch_um": round(propagator.pitch_m * 1e6, 4),
            "distances_mm": [round(float(z) * 1000, 4) for z in propagator.distances_m],
            "reconstruction": "fresnel_fft_amplitude",
        }

    @app.get("/api/health")
    async def health() -> dict:
        return {"status": "ok", "device": str(app.state.device), "optics": optics()}

    @app.get("/api/model/info")
    async def model_info() -> dict:
        return {
            "available": app.state.model is not None,
            "version": app.state.model_version,
            "status": app.state.model_status,
            "device": str(app.state.device),
        }

    @app.get("/api/methods")
    async def methods() -> dict:
        neural_ready = app.state.model is not None
        return {
            "methods": [
                {"id": "classical", "available": True, "uses_iterations": True},
                {"id": "neural", "available": neural_ready, "uses_iterations": False},
                {"id": "hybrid", "available": neural_ready, "uses_iterations": True},
            ],
            "max_iterations": MAX_ITERATIONS,
        }

    @app.get("/api/demo")
    async def demos() -> list[dict]:
        return DEMOS

    @app.get("/api/demo/{index}")
    def demo(index: int) -> dict:
        if not 0 <= index < len(DEMOS):
            raise HTTPException(404, "Demo set does not exist.")
        images = [_b64(gray_to_png(target)) for target in synthetic_targets(index, split="demo")]
        return {**DEMOS[index], "images": images}

    @app.post("/api/synthesize")
    async def synthesis(
        images: Annotated[list[UploadFile], File(min_length=4, max_length=4)],
        method: Annotated[Method, Form()] = "classical",
        iterations: Annotated[int | None, Form(ge=0, le=MAX_ITERATIONS)] = None,
        seed: Annotated[int, Form(ge=0, le=2**31 - 1)] = 0,
    ) -> dict:
        if method != "classical" and app.state.model is None:
            raise HTTPException(503, "Neural synthesis requires a loaded checkpoint.")
        if method == "neural" and iterations is not None:
            raise HTTPException(422, "Neural synthesis does not use iterations.")
        if method == "classical" and iterations == 0:
            raise HTTPException(422, "Classical synthesis requires at least one iteration.")

        target_images = []
        for plane, upload in enumerate(images, start=1):
            content = await upload.read(MAX_FILE_BYTES + 1)
            if len(content) > MAX_FILE_BYTES:
                raise HTTPException(413, f"Plane {plane} exceeds the PNG upload limit.")
            try:
                target_images.append(png_to_gray(content, app.state.propagator.size))
            except ValueError as exc:
                raise HTTPException(422, f"Plane {plane}: {exc}") from exc

        limiter = app.state.limiter
        try:
            limiter.acquire_nowait()
        except anyio.WouldBlock as exc:
            raise HTTPException(
                429, "Synthesis is busy. Try again after the active run finishes.",
                headers={"Retry-After": "2"},
            ) from exc

        try:
            result = await run_in_threadpool(
                synthesize,
                torch.stack(target_images),
                method,
                app.state.propagator,
                model=app.state.model if method != "classical" else None,
                iterations=iterations,
                seed=seed,
            )
        finally:
            limiter.release()

        return {
            "method": method,
            "model_version": app.state.model_version if method != "classical" else None,
            "phase_png": _b64(result.phase_png),
            "reconstruction_pngs": [_b64(png) for png in result.reconstruction_pngs],
            "metrics": result.metrics,
            "seconds": result.seconds,
            "iterations": result.iterations,
            "parameters": {**optics(), "seed": seed, "device": str(app.state.device)},
        }

    return app


app = create_app()
