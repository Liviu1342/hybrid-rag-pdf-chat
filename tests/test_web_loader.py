"""Tests for the safe web download, with a fake DNS and a fake website (no internet)."""

import socket
from collections.abc import Iterator

import pytest

import web_loader
from web_loader import MAX_BYTES, WebPageError, fetch_page, validate_url

# What our fake DNS answers for each host name
FAKE_DNS = {
    "example.com": "93.184.215.14",  # a public address
    "router.local": "192.168.1.1",  # a home network
    "metadata.cloud": "169.254.169.254",  # the cloud metadata service
    "localhost": "127.0.0.1",
}


def fake_getaddrinfo(host: str, port: object) -> list[tuple[object, ...]]:
    """Like the real one: an IP address is returned as it is, a name is looked up."""
    address = FAKE_DNS.get(host, host)
    if not address[0].isdigit():
        raise socket.gaierror("unknown host")
    return [(None, None, None, "", (address, 0))]


@pytest.fixture(autouse=True)
def fake_dns(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every test in this file uses the fake DNS instead of the real one."""
    monkeypatch.setattr(web_loader.socket, "getaddrinfo", fake_getaddrinfo)


class FakeResponse:
    """The parts of requests.Response that fetch_page uses."""

    def __init__(self, status: int = 200, headers: dict[str, str] | None = None, body: bytes = b"") -> None:
        self.status_code = status
        self.headers = headers or {"Content-Type": "text/html; charset=utf-8"}
        self.body = body
        self.is_redirect = status in (301, 302, 303, 307, 308)

    def iter_content(self, chunk_size: int) -> Iterator[bytes]:
        for start in range(0, len(self.body), chunk_size):
            yield self.body[start : start + chunk_size]

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *exc_info: object) -> None:
        pass


def serve(monkeypatch: pytest.MonkeyPatch, responses: dict[str, FakeResponse]) -> None:
    """Make requests.get answer from a dictionary of URL -> response."""

    def fake_get(url: str, **kwargs: object) -> FakeResponse:
        # With automatic redirects, requests would follow them without our checks
        assert kwargs.get("allow_redirects") is False, "redirects must be checked by our code"
        return responses[url]

    monkeypatch.setattr(web_loader.requests, "get", fake_get)


def test_public_https_link_is_allowed() -> None:
    assert validate_url(" https://example.com/article ") == "https://example.com/article"


@pytest.mark.parametrize(
    "url",
    ["http://router.local/admin", "http://metadata.cloud/latest", "http://localhost:8501", "http://127.0.0.1/"],
)
def test_private_and_local_addresses_are_refused(url: str) -> None:
    with pytest.raises(WebPageError, match="private or local"):
        validate_url(url)


@pytest.mark.parametrize("url", ["file:///C:/Windows/win.ini", "ftp://example.com/file", "data:text/plain,hello"])
def test_other_schemes_are_refused(url: str) -> None:
    with pytest.raises(WebPageError, match="Only http"):
        validate_url(url)


def test_page_is_downloaded(monkeypatch: pytest.MonkeyPatch) -> None:
    serve(monkeypatch, {"https://example.com/a": FakeResponse(body=b"<p>Hello</p>")})
    assert fetch_page("https://example.com/a") == ("https://example.com/a", b"<p>Hello</p>")


def test_redirect_to_a_private_address_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    redirect = FakeResponse(status=302, headers={"Location": "http://metadata.cloud/latest"})
    serve(monkeypatch, {"https://example.com/a": redirect})
    with pytest.raises(WebPageError, match="private or local"):
        fetch_page("https://example.com/a")


def test_too_large_page_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    serve(monkeypatch, {"https://example.com/big": FakeResponse(body=b"x" * (MAX_BYTES + 1))})
    with pytest.raises(WebPageError, match="larger than 5 MB"):
        fetch_page("https://example.com/big")


def test_non_html_content_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    pdf = FakeResponse(headers={"Content-Type": "application/pdf"}, body=b"%PDF")
    serve(monkeypatch, {"https://example.com/file": pdf})
    with pytest.raises(WebPageError, match="not a web page"):
        fetch_page("https://example.com/file")