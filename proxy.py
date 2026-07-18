import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from typing import Any
from urllib.parse import quote

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("pii-proxy")

OC_URL = os.environ.get(
    "OPENCONNECTOR_URL",
    "http://openconnector:3000",
).rstrip("/")

ANALYZER_URL = os.environ.get(
    "PRESIDIO_ANALYZER_URL",
    "http://presidio-analyzer:5001",
).rstrip("/")

ANON_URL = os.environ.get(
    "PRESIDIO_ANONYMIZER_URL",
    "http://presidio-anonymizer:5001",
).rstrip("/")

MAX_REQUEST_BYTES = int(os.environ.get("MAX_REQUEST_BYTES", "1048576"))
MAX_RESPONSE_BYTES = int(os.environ.get("MAX_RESPONSE_BYTES", "5242880"))
MAX_STRING_LENGTH = int(os.environ.get("MAX_STRING_LENGTH", "250000"))
REDACTION_CONCURRENCY = int(os.environ.get("REDACTION_CONCURRENCY", "8"))

# Add only headers OpenConnector genuinely requires.
ALLOWED_REQUEST_HEADERS = {
    "accept",
    "authorization",
    "content-type",
    "user-agent",
}

# Headers returned to the AI are generated locally.
SAFE_RESPONSE_HEADERS = {
    "cache-control": "no-store",
    "x-content-type-options": "nosniff",
}


class RedactionFailure(Exception):
    """Raised whenever safe redaction cannot be confirmed."""


@asynccontextmanager
async def lifespan(app: FastAPI):
    timeout = httpx.Timeout(
        connect=5.0,
        read=60.0,
        write=30.0,
        pool=5.0,
    )

    limits = httpx.Limits(
        max_connections=50,
        max_keepalive_connections=20,
    )

    # trust_env=False prevents proxy and .netrc environment settings from
    # unexpectedly redirecting or authenticating internal requests.
    app.state.client = httpx.AsyncClient(
        timeout=timeout,
        limits=limits,
        follow_redirects=False,
        trust_env=False,
    )

    app.state.redaction_semaphore = asyncio.Semaphore(
        REDACTION_CONCURRENCY
    )

    try:
        yield
    finally:
        await app.state.client.aclose()


app = FastAPI(
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
    lifespan=lifespan,
)


def safe_error(status_code: int, code: str) -> JSONResponse:
    return JSONResponse(
        content={"error": code},
        status_code=status_code,
        headers=SAFE_RESPONSE_HEADERS,
    )


@app.get("/health")
async def health() -> dict[str, str]:
    # Do not expose dependency URLs or diagnostic details here.
    return {"status": "ok"}


async def redact_text(app: FastAPI, value: str) -> str:
    if not isinstance(value, str):
        raise RedactionFailure("Invalid redaction input")

    if not value:
        return value

    if len(value) > MAX_STRING_LENGTH:
        raise RedactionFailure("String exceeds redaction limit")

    client: httpx.AsyncClient = app.state.client
    semaphore: asyncio.Semaphore = app.state.redaction_semaphore

    try:
        async with semaphore:
            analyzer_response = await client.post(
                f"{ANALYZER_URL}/analyze",
                json={
                    "text": value,
                    "language": "en",
                },
            )

        if analyzer_response.status_code != 200:
            raise RedactionFailure("Analyzer returned an error")

        results = analyzer_response.json()

        if not isinstance(results, list):
            raise RedactionFailure("Invalid analyzer response")

        # An empty result is accepted only as Presidio's explicit determination
        # that no configured recognizer detected PII.
        if not results:
            return value

        async with semaphore:
            anonymizer_response = await client.post(
                f"{ANON_URL}/anonymize",
                json={
                    "text": value,
                    "analyzer_results": results,
                    "anonymizers": {
                        "DEFAULT": {
                            "type": "replace",
                            "new_value": "<REDACTED>",
                        }
                    },
                },
            )

        if anonymizer_response.status_code != 200:
            raise RedactionFailure("Anonymizer returned an error")

        payload = anonymizer_response.json()
        redacted = payload.get("text")

        if not isinstance(redacted, str):
            raise RedactionFailure("Invalid anonymizer response")

        # If PII was detected but the text was not changed, fail closed.
        if redacted == value:
            raise RedactionFailure("Anonymizer did not alter detected PII")

        return redacted

    except RedactionFailure:
        raise
    except (
        httpx.HTTPError,
        json.JSONDecodeError,
        TypeError,
        ValueError,
    ) as exc:
        raise RedactionFailure("Redaction service failure") from exc


async def redact_json(app: FastAPI, value: Any) -> Any:
    if isinstance(value, str):
        return await redact_text(app, value)

    if value is None or isinstance(value, bool):
        return value

    if isinstance(value, dict):
        redacted: dict[str, Any] = {}

        for key, item in value.items():
            # JSON keys are strings, but normalize defensively.
            safe_key = await redact_text(app, str(key))
            safe_value = await redact_json(app, item)

            # Prevent collisions when several PII keys become <REDACTED>.
            candidate = safe_key
            suffix = 2
            while candidate in redacted:
                candidate = f"{safe_key}_{suffix}"
                suffix += 1

            redacted[candidate] = safe_value

        return redacted

    if isinstance(value, list):
        return [
            await redact_json(app, item)
            for item in value
        ]

    # Numbers may be phone numbers, account numbers, SSNs without punctuation,
    # timestamps, coordinates, or organization-specific identifiers.
    #
    # Blocking them is restrictive, but it is safer than claiming that arbitrary
    # numeric values can always be classified correctly.
    if isinstance(value, (int, float)):
        return "<REDACTED_NUMERIC_VALUE>"

    raise RedactionFailure("Unsupported JSON value type")


def filtered_request_headers(request: Request) -> dict[str, str]:
    headers: dict[str, str] = {}

    for name, value in request.headers.items():
        lower_name = name.lower()
        if lower_name in ALLOWED_REQUEST_HEADERS:
            headers[lower_name] = value

    # HTTPX calculates the correct value after receiving the body.
    headers.pop("content-length", None)
    return headers


@app.api_route(
    "/{path:path}",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
)
async def proxy(request: Request, path: str):
    body = await request.body()

    if len(body) > MAX_REQUEST_BYTES:
        return safe_error(413, "request_too_large")

    # Quote each path segment rather than allowing arbitrary URL syntax.
    safe_path = "/".join(
        quote(segment, safe="")
        for segment in path.split("/")
    )

    url = f"{OC_URL}/{safe_path}"

    try:
        upstream = await request.app.state.client.request(
            method=request.method,
            url=url,
            params=list(request.query_params.multi_items()),
            headers=filtered_request_headers(request),
            content=body,
        )
    except httpx.HTTPError:
        logger.exception("OpenConnector request failed")
        return safe_error(502, "upstream_unavailable")

    if len(upstream.content) > MAX_RESPONSE_BYTES:
        return safe_error(502, "upstream_response_too_large")

    content_type = upstream.headers.get("content-type", "")
    media_type = content_type.split(";", 1)[0].strip().lower()

    # Never pass redirects, errors, files, streams, HTML, text, XML, or CSV.
    if not 200 <= upstream.status_code < 300:
        return safe_error(502, "upstream_rejected")

    if media_type not in {
        "application/json",
        "application/problem+json",
    } and not media_type.endswith("+json"):
        return safe_error(502, "non_json_response_rejected")

    try:
        decoded = upstream.json()
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
        return safe_error(502, "invalid_json_response")

    try:
        redacted = await redact_json(request.app, decoded)
    except RedactionFailure:
        logger.exception("Response blocked because redaction failed")
        return safe_error(502, "redaction_failed")

    return JSONResponse(
        content=redacted,
        status_code=upstream.status_code,
        headers=SAFE_RESPONSE_HEADERS,
    )
