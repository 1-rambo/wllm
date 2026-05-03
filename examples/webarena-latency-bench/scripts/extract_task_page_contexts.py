#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from webarena_verified.types.config import WebArenaVerifiedConfig
from webarena_verified.types.task import WebArenaSite
from webarena_verified.environments import MAGENTO_ADMIN_AUTO_LOGIN_HEADER


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def extract_visible_text(html: str, max_chars: int) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg", "canvas", "footer"]):
      tag.decompose()
    texts: list[str] = []
    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    if title:
        texts.append(f"Page Title: {normalize_whitespace(title)}")
    headings = [normalize_whitespace(node.get_text(" ", strip=True)) for node in soup.find_all(["h1", "h2", "h3"])[:12]]
    if headings:
        texts.append("Headings: " + " | ".join(h for h in headings if h))
    body_text = normalize_whitespace(soup.get_text(" ", strip=True))
    if body_text:
        texts.append("Visible Text: " + body_text)
    merged = "\n".join(part for part in texts if part)
    return merged[:max_chars]


def extract_same_origin_links(base_url: str, html: str, max_links: int) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    base_origin = urlparse(base_url)
    links: list[str] = []
    seen: set[str] = set()
    for anchor in soup.find_all("a", href=True):
        href = (anchor.get("href") or "").strip()
        if not href or href.startswith("#") or href.startswith("javascript:") or href.startswith("mailto:"):
            continue
        abs_url = urljoin(base_url, href)
        parsed = urlparse(abs_url)
        if parsed.scheme not in {"http", "https"}:
            continue
        if parsed.netloc != base_origin.netloc:
            continue
        normalized = parsed._replace(fragment="", query="").geturl().rstrip("/")
        if normalized in seen:
            continue
        seen.add(normalized)
        links.append(normalized)
        if len(links) >= max_links:
            break
    return links


def site_enum(site_name: str) -> WebArenaSite:
    return WebArenaSite(site_name)


def login_shopping_admin(session: requests.Session, base_url: str, username: str, password: str, timeout_s: int) -> None:
    login_url = base_url.rstrip("/")
    resp = session.get(login_url, timeout=timeout_s, allow_redirects=True)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    form = soup.find("form", {"id": "login-form"})
    if form is None:
        return
    form_key_input = form.find("input", {"name": "form_key"})
    form_key = form_key_input.get("value", "") if form_key_input else ""
    payload = {
        "form_key": form_key,
        "login[username]": username,
        "login[password]": password,
    }
    post_resp = session.post(login_url, data=payload, timeout=timeout_s, allow_redirects=True)
    post_resp.raise_for_status()


def build_session_for_site(config: WebArenaVerifiedConfig, site: WebArenaSite) -> requests.Session:
    session = requests.Session()
    env = config.get_environment(site)
    if env is None:
        raise ValueError(f"Missing environment config for site={site.value}")

    credentials = env.credentials or {}
    if site == WebArenaSite.SHOPPING_ADMIN and env.use_header_login and credentials.get("username"):
        session.headers[MAGENTO_ADMIN_AUTO_LOGIN_HEADER] = credentials["username"]

    if credentials.get("username") and credentials.get("password") and site != WebArenaSite.SHOPPING_ADMIN:
        session.auth = (credentials["username"], credentials["password"])

    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (compatible; WebArenaContextExtractor/1.0)",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }
    )
    return session


def render_urls(config: WebArenaVerifiedConfig, task: dict[str, Any]) -> list[str]:
    sites = [site_enum(s) for s in task["sites"]]
    rendered = config.render_url(task["startUrls"], sites, strict=False)
    return list(rendered) if isinstance(rendered, list) else [rendered]


def fetch_page_text(session: requests.Session, url: str, timeout_s: int, max_chars: int) -> str:
    resp = session.get(url, timeout=timeout_s, allow_redirects=True)
    resp.raise_for_status()
    content_type = resp.headers.get("content-type", "")
    if "html" not in content_type and not resp.text.lstrip().startswith("<"):
        return resp.text[:max_chars]
    return extract_visible_text(resp.text, max_chars=max_chars)


def crawl_page_context(
    session: requests.Session,
    start_url: str,
    timeout_s: int,
    max_chars_total: int,
    crawl_max_pages: int,
    crawl_max_links_per_page: int,
) -> str:
    queue: list[str] = [start_url]
    visited: set[str] = set()
    collected_parts: list[str] = []

    while queue and len(visited) < crawl_max_pages:
        url = queue.pop(0)
        normalized = url.rstrip("/")
        if normalized in visited:
            continue
        visited.add(normalized)
        try:
            resp = session.get(url, timeout=timeout_s, allow_redirects=True)
            resp.raise_for_status()
        except Exception as exc:  # noqa: BLE001
            collected_parts.append(f"URL: {url}\nFetch Error: {exc}")
            continue

        content_type = resp.headers.get("content-type", "")
        if "html" in content_type or resp.text.lstrip().startswith("<"):
            text = extract_visible_text(resp.text, max_chars=max_chars_total)
            links = extract_same_origin_links(resp.url, resp.text, max_links=crawl_max_links_per_page)
            for link in links:
                if link.rstrip("/") not in visited and link not in queue:
                    queue.append(link)
        else:
            text = resp.text[:max_chars_total]

        collected_parts.append(f"URL: {resp.url}\n{text}".strip())
        merged = "\n\n".join(collected_parts)
        if len(merged) >= max_chars_total:
            return merged[:max_chars_total]

    return "\n\n".join(collected_parts)[:max_chars_total]


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract real WebArena page contexts for tasks.")
    parser.add_argument("--dataset", required=True, help="Path to retrieve_info_subset.json")
    parser.add_argument("--config", required=True, help="Path to WebArena config.json with environments")
    parser.add_argument("--output", required=True, help="Path to write task_page_contexts.json")
    parser.add_argument("--max-chars", type=int, default=12000, help="Max chars per page context")
    parser.add_argument("--timeout-s", type=int, default=20, help="HTTP timeout per page")
    parser.add_argument("--limit", type=int, default=0, help="Optional limit on number of tasks")
    parser.add_argument("--crawl-max-pages", type=int, default=6, help="Max same-origin pages to crawl per start-url group")
    parser.add_argument("--crawl-max-links-per-page", type=int, default=12, help="Max same-origin links to enqueue from each crawled page")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    config_path = Path(args.config)
    output_path = Path(args.output)

    tasks = json.loads(dataset_path.read_text(encoding="utf-8"))
    if args.limit and args.limit > 0:
        tasks = tasks[: args.limit]

    config = WebArenaVerifiedConfig.from_file(config_path)
    sessions: dict[str, requests.Session] = {}
    fetched_context_by_url_key: dict[str, str] = {}
    rendered_urls_by_key: dict[str, list[str]] = {}
    task_contexts: dict[str, dict[str, Any]] = {}

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for task in tasks:
        rendered = render_urls(config, task)
        key = " | ".join(rendered)
        task["_renderedStartUrls"] = rendered
        task["_pageContextKey"] = key
        grouped[key].append(task)

    for key, group_tasks in grouped.items():
        first = group_tasks[0]
        site = site_enum(first["site"])
        session = sessions.get(site.value)
        if session is None:
            session = build_session_for_site(config, site)
            env = config.get_environment(site)
            credentials = (env.credentials if env else None) or {}
            if (
                site == WebArenaSite.SHOPPING_ADMIN
                and credentials.get("username")
                and credentials.get("password")
                and not (env.use_header_login if env else False)
            ):
                login_shopping_admin(
                    session,
                    env.active_url or first["_renderedStartUrls"][0],
                    credentials["username"],
                    credentials["password"],
                    args.timeout_s,
                )
            sessions[site.value] = session

        page_parts: list[str] = []
        per_url_budget = max(1000, args.max_chars // max(1, len(first["_renderedStartUrls"])))
        for url in first["_renderedStartUrls"]:
            try:
                page_text = crawl_page_context(
                    session,
                    url,
                    timeout_s=args.timeout_s,
                    max_chars_total=per_url_budget,
                    crawl_max_pages=args.crawl_max_pages,
                    crawl_max_links_per_page=args.crawl_max_links_per_page,
                )
            except Exception as exc:  # noqa: BLE001
                page_text = f"[FetchError] url={url} error={exc}"
            page_parts.append(f"URL: {url}\n{page_text}")

        merged = "\n\n".join(page_parts)[: args.max_chars]
        fetched_context_by_url_key[key] = merged
        rendered_urls_by_key[key] = list(first["_renderedStartUrls"])

    for task in tasks:
        key = task["_pageContextKey"]
        task_contexts[task["id"]] = {
            "renderedStartUrls": rendered_urls_by_key[key],
            "pageContextKey": key,
            "pageContext": fetched_context_by_url_key[key],
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(task_contexts, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(task_contexts)} task page contexts to {output_path}")


if __name__ == "__main__":
    main()
