"""Download a public web page safely.

A link typed by the user makes OUR computer send a request. Without checks,
someone could make it fetch internal addresses (SSRF), e.g. a router admin page
or the cloud metadata service at 169.254.169.254, or a huge file.
"""

import ipaddress
import socket
from urllib.parse import urljoin, urlparse

import requests

MAX_BYTES = 5 * 1024 * 1024  # 5 MB: far more than any article, far less than a disk
TIMEOUT_SECONDS = 10
MAX_REDIRECTS = 3
ALLOWED_CONTENT_TYPES = ("text/html", "application/xhtml+xml", "text/plain")


class WebPageError(Exception):
    """Error with a user-friendly message that can be shown directly in the UI."""


def validate_url(url: str) -> str:
    """Return the URL if it is a public http(s) address; raise WebPageError otherwise."""
    parsed = urlparse(url.strip())
    if parsed.scheme not in ("http", "https"):
        raise WebPageError("Only http:// and https:// links are allowed.")
    if not parsed.hostname:
        raise WebPageError("This link has no host name.")
    try:
        # A name can point to several addresses: every one of them must be public
        addresses = {str(info[4][0]) for info in socket.getaddrinfo(parsed.hostname, None)}
    except socket.gaierror:
        raise WebPageError(f"Cannot find the website '{parsed.hostname}'.") from None
    for address in addresses:
        ip = ipaddress.ip_address(address.split("%")[0])  # drop an IPv6 zone like "%eth0"
        if not ip.is_global:  # private, loopback, link-local, reserved...
            raise WebPageError("Links to private or local network addresses are not allowed.")
    return url.strip()


def fetch_page(url: str) -> tuple[str, bytes]:
    """Download a page; return (final_url, content). Every redirect is checked again."""
    for _ in range(MAX_REDIRECTS + 1):
        url = validate_url(url)
        try:
            response = requests.get(
                url,
                timeout=TIMEOUT_SECONDS,
                allow_redirects=False,  # we follow redirects ourselves, checking each one
                stream=True,  # do not download the body before we check its size
                headers={"User-Agent": "hybrid-rag-chat/1.0"},
            )
        except requests.RequestException as error:
            raise WebPageError(f"Could not download the page ({type(error).__name__}).") from error
        with response:  # closes the connection whatever happens
            if response.is_redirect:
                url = urljoin(url, response.headers.get("Location", ""))
                continue  # validate the new address in the next round
            if response.status_code != 200:
                raise WebPageError(f"The website answered with HTTP {response.status_code}.")
            content_type = response.headers.get("Content-Type", "").split(";")[0].strip().lower()
            if content_type not in ALLOWED_CONTENT_TYPES:
                raise WebPageError(f"This link is not a web page ({content_type or 'unknown type'}).")
            content = bytearray()
            for piece in response.iter_content(chunk_size=64 * 1024):
                content.extend(piece)
                if len(content) > MAX_BYTES:
                    raise WebPageError("The page is larger than 5 MB.")
            return url, bytes(content)
    raise WebPageError("Too many redirects.")