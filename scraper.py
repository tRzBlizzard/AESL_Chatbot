
import time
import xml.etree.ElementTree as ET
from collections import deque
from urllib.parse import (
    parse_qsl,
    urlencode,
    urldefrag,
    urljoin,
    urlparse,
    urlunparse,
)
from urllib.robotparser import RobotFileParser

import requests
import trafilatura
from bs4 import BeautifulSoup

from config import (
    ALLOWED_HOST,
    MAX_PAGES,
    REQUEST_DELAY_SECONDS,
    REQUEST_TIMEOUT,
    START_URL,
    USER_AGENT,
)
from database import initialize_database, save_page, database_stats


SKIP_EXTENSIONS = (
    ".pdf", ".jpg", ".jpeg", ".png", ".gif", ".webp",
    ".zip", ".doc", ".docx", ".xls", ".xlsx", ".ppt",
    ".pptx", ".mp4", ".mp3", ".xml",
)

SKIP_PATH_PARTS = (
    "/wp-admin", "/wp-login", "/cart", "/checkout",
    "/my-account", "/feed/",
)


def normalize_url(url):
    if not url:
        return None

    url, _fragment = urldefrag(url)
    parts = urlparse(url)

    if parts.scheme not in ("http", "https"):
        return None

    host = (parts.hostname or "").lower()

    if host != ALLOWED_HOST and not host.endswith("." + ALLOWED_HOST):
        return None

    if parts.username or parts.password:
        return None

    path = parts.path or "/"
    lower_path = path.lower()

    if lower_path.endswith(SKIP_EXTENSIONS):
        return None

    if any(part in lower_path for part in SKIP_PATH_PARTS):
        return None

    # Keep only query parameters that may affect public content.
    # Most tracking parameters are discarded.
    kept_params = [
        (key, value)
        for key, value in parse_qsl(parts.query, keep_blank_values=False)
        if key.lower() in {"page", "lang"}
    ]

    return urlunparse((
        parts.scheme.lower(),
        parts.netloc.lower(),
        path,
        "",
        urlencode(kept_params),
        "",
    ))


def make_session():
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    return session


def load_robots(session):
    robots_url = urljoin(START_URL, "/robots.txt")
    parser = RobotFileParser()
    parser.set_url(robots_url)

    try:
        response = session.get(
            robots_url, timeout=REQUEST_TIMEOUT
        )

        if response.status_code == 404:
            parser.parse([])
        else:
            response.raise_for_status()
            parser.parse(response.text.splitlines())

    except requests.RequestException as exc:
        # Fail closed if the site's crawl policy cannot be checked.
        raise RuntimeError(
            f"Could not check robots.txt at {robots_url}: {exc}"
        ) from exc

    return parser


def sitemap_urls(session, start_url):
    """Collect URLs from common sitemap locations and sitemap indexes."""
    candidates = [
        urljoin(start_url, "/sitemap.xml"),
        urljoin(start_url, "/sitemap_index.xml"),
    ]

    robots_url = urljoin(start_url, "/robots.txt")
    try:
        response = session.get(
            robots_url, timeout=REQUEST_TIMEOUT
        )
        if response.ok:
            for line in response.text.splitlines():
                if line.lower().startswith("sitemap:"):
                    candidates.append(line.split(":", 1)[1].strip())
    except requests.RequestException:
        pass

    found = set()
    seen_sitemaps = set()
    queue = deque(candidates)

    while queue and len(seen_sitemaps) < 50:
        sitemap = queue.popleft()
        if sitemap in seen_sitemaps:
            continue
        seen_sitemaps.add(sitemap)

        try:
            response = session.get(
                sitemap, timeout=REQUEST_TIMEOUT
            )
            if not response.ok:
                continue

            root = ET.fromstring(response.content)

            for element in root.iter():
                if element.tag.endswith("loc") and element.text:
                    loc = element.text.strip()

                    if element is root or not loc:
                        continue

                    if loc.lower().endswith(".xml"):
                        queue.append(loc)
                    else:
                        normalized = normalize_url(loc)
                        if normalized:
                            found.add(normalized)

        except (requests.RequestException, ET.ParseError):
            continue

    return sorted(found)


def extract_links(html, base_url):
    soup = BeautifulSoup(html, "html.parser")
    result = set()

    for anchor in soup.find_all("a", href=True):
        candidate = normalize_url(urljoin(base_url, anchor["href"]))
        if candidate:
            result.add(candidate)

    return result


def extract_title(html, fallback_url):
    soup = BeautifulSoup(html, "html.parser")

    title_tag = soup.find("title")
    if title_tag:
        title = title_tag.get_text(" ", strip=True)
        if title:
            return title

    heading = soup.find("h1")
    if heading:
        title = heading.get_text(" ", strip=True)
        if title:
            return title

    return fallback_url


def extract_menu_text(html):
    """
    trafilatura drops navigation menus as boilerplate, but the Products
    page's menu is where the product categories are listed. Keep those
    link labels so the category lookup can find them.
    """
    soup = BeautifulSoup(html, "html.parser")
    labels = []

    for container in soup.find_all(["nav", "header"]):
        for anchor in container.find_all("a"):
            label = anchor.get_text(" ", strip=True)

            if label and len(label) <= 60 and label not in labels:
                labels.append(label)

    return " | ".join(labels)


def crawl_website():
    initialize_database()

    session = make_session()
    robots = load_robots(session)

    start = normalize_url(START_URL)
    sitemap = sitemap_urls(session, START_URL)

    queue = deque([start, *sitemap])
    queued = set(queue)
    visited = set()

    saved = 0
    failed = 0
    blocked = 0

    print(f"Found {len(sitemap)} candidate URLs in sitemaps.")
    print(f"Maximum page visits this run: {MAX_PAGES}")

    while queue and len(visited) < MAX_PAGES:
        url = queue.popleft()

        if not url or url in visited:
            continue

        visited.add(url)

        if not robots.can_fetch(USER_AGENT, url):
            print(f"Blocked by robots.txt: {url}")
            blocked += 1
            continue

        try:
            response = session.get(
                url,
                timeout=REQUEST_TIMEOUT,
                allow_redirects=True,
            )
            response.raise_for_status()

            final_url = normalize_url(response.url)
            if not final_url:
                continue

            content_type = response.headers.get("Content-Type", "")
            if "text/html" not in content_type.lower():
                continue

            html = response.text

            for link in extract_links(html, final_url):
                if link not in queued and link not in visited:
                    queued.add(link)
                    queue.append(link)

            title = extract_title(html, final_url)
            content = trafilatura.extract(
                html,
                include_tables=True,
                include_links=False,
                favor_precision=False,
            )

            if content and len(content.strip()) >= 80:
                content = content.strip()

                if urlparse(final_url).path.rstrip("/").lower().endswith(
                    "/products"
                ):
                    menu = extract_menu_text(html)

                    if menu:
                        content += "\n\nSite menu: " + menu

                save_page(final_url, title, content)
                saved += 1
                print(f"[saved {saved}] {title} | {final_url}")
            else:
                print(f"No useful main text: {final_url}")

        except requests.RequestException as exc:
            failed += 1
            print(f"Download failed: {url} | {exc}")
        except Exception as exc:
            failed += 1
            print(f"Page processing failed: {url} | {exc}")

        time.sleep(REQUEST_DELAY_SECONDS)

    stats = database_stats()

    print("\n--- Crawl summary ---")
    print(f"URLs visited: {len(visited)}")
    print(f"Pages saved/updated: {saved}")
    print(f"Blocked by robots.txt: {blocked}")
    print(f"Failed pages: {failed}")
    print(f"Pages currently stored: {stats['pages']}")
    print(f"Searchable chunks: {stats['chunks']}")
    print(f"Database last update: {stats['latest_update']}")


if __name__ == "__main__":
    crawl_website()
