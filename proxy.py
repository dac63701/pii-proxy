import asyncio
import json
import logging
import os
import re
from contextlib import asynccontextmanager
from typing import Any
from urllib.parse import quote

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("pii-proxy")

OC_URL = os.environ.get(
    "OPENCONNECTOR_URL",
    "http://openconnector:3000",
).rstrip("/")

# Separate credentials for the two trust boundaries. Never forward the
# caller's proxy bearer token to OpenConnector.
PII_PROXY_AUTH_TOKEN = os.environ.get("PII_PROXY_AUTH_TOKEN", "").strip()
OPENCONNECTOR_RUNTIME_TOKEN = os.environ.get(
    "OPENCONNECTOR_RUNTIME_TOKEN",
    "",
).strip()

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

# Keep ordinary names and email addresses available to the agent, but redact
# high-risk identifiers and contact/payment data from provider responses.
SENSITIVE_PII_ENTITIES = tuple(
    item.strip()
    for item in os.environ.get(
        "SENSITIVE_PII_ENTITIES",
        "CREDIT_CARD,US_SSN,US_BANK_NUMBER,US_DRIVER_LICENSE,US_PASSPORT,"
        "IBAN_CODE,PHONE_NUMBER,IP_ADDRESS,CRYPTO",
    ).split(",")
    if item.strip()
)

# Defense-in-depth for long bare numeric identifiers that Presidio may not
# classify because the provider response omits context (for example an ID
# number returned as a standalone JSON string). Dates and short IDs are left
# alone to avoid breaking MCP/Gmail identifiers.
LONG_NUMERIC_IDENTIFIER = re.compile(
    r"(?<![A-Za-z0-9])(?:\d[ -]?){9,18}\d(?![A-Za-z0-9])"
)

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
                    "entities": list(SENSITIVE_PII_ENTITIES),
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
            return redact_numeric_identifiers(value)

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

        # If sensitive PII was detected but the text was not changed, fail closed.
        if redacted == value:
            raise RedactionFailure("Anonymizer did not alter detected PII")

        return redact_numeric_identifiers(redacted)

    except RedactionFailure:
        raise
    except (
        httpx.HTTPError,
        json.JSONDecodeError,
        TypeError,
        ValueError,
    ) as exc:
        raise RedactionFailure("Redaction service failure") from exc


def redact_numeric_identifiers(value: str) -> str:
    """Mask long standalone numeric identifiers missed by recognizers."""
    return LONG_NUMERIC_IDENTIFIER.sub("<REDACTED_NUMERIC_IDENTIFIER>", value)


async def redact_json(app: FastAPI, value: Any) -> Any:
    if isinstance(value, str):
        return await redact_text(app, value)

    if value is None or isinstance(value, bool):
        return value

    if isinstance(value, dict):
        redacted: dict[str, Any] = {}

        for key, item in value.items():
            # Preserve JSON/MCP field names; only redact their values.
            safe_key = str(key)
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

    # Numeric JSON values may be protocol IDs, timestamps, or amounts. Preserve
    # them because changing MCP envelope numbers can break request correlation;
    # string-form identifiers are handled by Presidio and the fallback regex.
    if isinstance(value, (int, float)):
        # Preserve JSON/MCP protocol numbers. Sensitive identifiers are
        # detected when represented as strings and handled above.
        return value

    raise RedactionFailure("Unsupported JSON value type")


async def redact_mcp_body(
    app: FastAPI,
    body: bytes,
    content_type: str,
    status_code: int | None = None,
) -> bytes:
    """Redact MCP JSON or SSE data without changing the MCP framing.

    OpenConnector may acknowledge an asynchronous MCP request with an empty
    ``202 Accepted`` response. There is no provider payload to redact in that
    case, but an empty response with any other status remains invalid and is
    blocked by the fail-closed behavior below.
    """
    if status_code == 202 and not body:
        return b""

    media_type = content_type.split(";", 1)[0].strip().lower()

    if media_type in {"application/json", "application/problem+json"}:
        try:
            decoded = json.loads(body)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise RedactionFailure("Invalid MCP JSON response") from exc
        redacted = await redact_json(app, decoded)
        return json.dumps(redacted, ensure_ascii=False, separators=(",", ":")).encode()

    if media_type == "text/event-stream":
        text = body.decode("utf-8")
        output: list[str] = []
        for line in text.splitlines(keepends=True):
            if not line.startswith("data:"):
                output.append(line)
                continue

            prefix, raw = line.split(":", 1)
            payload = raw[1:] if raw.startswith(" ") else raw
            newline = "\n" if line.endswith("\n") else ""
            try:
                decoded = json.loads(payload.rstrip("\r\n"))
                safe_payload = json.dumps(
                    await redact_json(app, decoded),
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            except json.JSONDecodeError:
                safe_payload = redact_numeric_identifiers(payload.rstrip("\r\n"))
            output.append(f"{prefix}: {safe_payload}{newline}")
        return "".join(output).encode("utf-8")

    # MCP implementations sometimes omit a precise content type. Fail closed
    # rather than return an unscanned provider response.
    raise RedactionFailure("Unsupported MCP response content type")


def filtered_request_headers(request: Request) -> dict[str, str]:
    headers: dict[str, str] = {}

    for name, value in request.headers.items():
        lower_name = name.lower()
        if lower_name in ALLOWED_REQUEST_HEADERS:
            headers[lower_name] = value

    # HTTPX calculates the correct value after receiving the body.
    headers.pop("content-length", None)
    return headers


def authenticated_request_headers(request: Request) -> dict[str, str] | None:
    """Validate the proxy token and return sanitized upstream headers."""
    if not PII_PROXY_AUTH_TOKEN or not OPENCONNECTOR_RUNTIME_TOKEN:
        logger.error("Required proxy authentication tokens are not configured")
        return None

    if request.headers.get("authorization", "") != (
        f"Bearer {PII_PROXY_AUTH_TOKEN}"
    ):
        return None

    headers = filtered_request_headers(request)
    headers["authorization"] = f"Bearer {OPENCONNECTOR_RUNTIME_TOKEN}"
    return headers


@app.api_route(
    "/mcp",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
)
async def proxy_mcp_stream(request: Request):
    """MCP Streamable HTTP passthrough.

    OpenConnector serves MCP over Streamable HTTP. Server-to-client
    messages may use text/event-stream, which cannot be redacted without
    destroying the stream. We therefore stream the upstream bytes through
    verbatim and fail closed only on transport errors, non-2xx statuses,
    or redirects.

    This is a deliberate, documented exception to the JSON redaction rule:
    the isolation guarantee (the agent cannot reach raw OpenConnector) still
    holds because every byte still transits this proxy.
    """
    body = await request.body()

    if len(body) > MAX_REQUEST_BYTES:
        return safe_error(413, "request_too_large")

    headers = authenticated_request_headers(request)
    if headers is None:
        return safe_error(401, "unauthorized")

    # OpenConnector's MCP SDK (v1.29+) requires both "application/json" AND
    # "text/event-stream" in the Accept header for POST requests or it returns
    # 406 Not Acceptable.  Always set the canonical value so the proxy never
    # accidentally passes a header that doesn't pass the check (e.g. when the
    # incoming request has only "application/json" or a client sends */*).
    headers["accept"] = "application/json, text/event-stream"

    url = f"{OC_URL}/mcp"

    try:
        upstream = await request.app.state.client.request(
            method=request.method,
            url=url,
            params=list(request.query_params.multi_items()),
            headers=headers,
            content=body,
        )
    except httpx.HTTPError:
        logger.exception("OpenConnector MCP request failed")
        return safe_error(502, "upstream_unavailable")

    if not 200 <= upstream.status_code < 300:
        return safe_error(502, "upstream_rejected")

    content_type = upstream.headers.get("content-type", "application/json")
    if len(upstream.content) > MAX_RESPONSE_BYTES:
        return safe_error(502, "upstream_response_too_large")
    try:
        redacted_body = await redact_mcp_body(
            request.app,
            upstream.content,
            content_type,
            status_code=upstream.status_code,
        )
    except RedactionFailure:
        logger.exception("MCP response blocked because redaction failed")
        return safe_error(502, "redaction_failed")

    return StreamingResponse(
        iter([redacted_body]),
        status_code=upstream.status_code,
        headers={
            "content-type": content_type,
            "cache-control": "no-store",
            "x-content-type-options": "nosniff",
        },
        background=None,
    )


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
