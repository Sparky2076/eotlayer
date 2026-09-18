"""Local HTML/link and browser checks. Requires beautifulsoup4 and playwright."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.parse import unquote, urlsplit

from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
HTML = ROOT / "docs/_build/html"


def check_links():
    pages = {path: BeautifulSoup(path.read_text(encoding="utf-8"), "html.parser")
             for path in HTML.rglob("*.html")}
    failures = []
    count = 0
    for page, soup in pages.items():
        for link in soup.select("a[href]"):
            parts = urlsplit(link["href"])
            if parts.scheme or parts.netloc:
                continue
            target = (page.parent / unquote(parts.path)).resolve() if parts.path else page
            if target.is_dir():
                target /= "index.html"
            if not target.exists():
                failures.append(f"{page.relative_to(HTML)} -> {link['href']}")
            elif parts.fragment and target in pages:
                fragment = unquote(parts.fragment)
                if not pages[target].find(id=fragment):
                    failures.append(f"Missing anchor: {page.name} -> {link['href']}")
            count += 1
    if failures:
        raise AssertionError("\n".join(failures))
    return {"html_pages": len(pages), "internal_links_checked": count}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8768")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report = check_links()
    print("Internal HTML links and anchors passed", flush=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1440, "height": 1000}, device_scale_factor=1)
        page = context.new_page()
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        response = page.goto(args.url + "/index.html", wait_until="networkidle")
        assert response.status == 200
        assert page.title() == "Welcome to EOTLayer — EOTLayer Documentation"
        assert page.locator(".bd-sidebar-primary").is_visible()
        assert page.locator(".bd-sidebar-secondary").is_visible()
        assert page.locator(".copybtn").count() >= 1
        page.screenshot(path=str(args.output / "homepage-desktop.png"), full_page=False)
        page.screenshot(path=str(args.output / "homepage-full.png"), full_page=True)
        page.goto(args.url + "/api/layer.html", wait_until="networkidle")
        assert page.locator("#eotlayer\\.EOTLayer").count() == 1
        assert "epsilon_scaling_start" in page.locator("article").inner_text()
        page.screenshot(path=str(args.output / "api-desktop.png"), full_page=False)
        page.goto(args.url + "/key_concepts/optimal_transport.html", wait_until="networkidle")
        page.wait_for_selector("mjx-container", timeout=30000)
        report["mathjax_rendered"] = page.locator("mjx-container").count()
        page.screenshot(path=str(args.output / "concepts-desktop.png"), full_page=False)
        page.goto(args.url + "/index.html", wait_until="networkidle")
        # The theme cycles light/dark/auto; the initial mode may follow the OS.
        for _ in range(3):
            if page.evaluate("document.documentElement.dataset.theme === 'dark'"):
                break
            page.locator("button.theme-switch-button").click()
        assert page.evaluate("document.documentElement.dataset.theme === 'dark'")
        page.screenshot(path=str(args.output / "homepage-dark.png"), full_page=False)
        page.goto(args.url + "/search.html?q=envelope", wait_until="networkidle")
        page.wait_for_selector("#search-results li", timeout=20000)
        report["search_results"] = page.locator("#search-results li").count()
        assert "envelope" in page.locator("#search-results").inner_text().lower()
        print("Desktop, API, MathJax, theme switch, and search passed", flush=True)
        mobile = browser.new_context(viewport={"width": 390, "height": 844}, device_scale_factor=1, is_mobile=True, has_touch=True)
        mobile_page = mobile.new_page()
        mobile_page.goto(args.url + "/index.html", wait_until="networkidle")
        overflow = mobile_page.evaluate("document.documentElement.scrollWidth > window.innerWidth + 1")
        assert not overflow, "Mobile homepage has horizontal overflow"
        mobile_page.screenshot(path=str(args.output / "homepage-mobile.png"), full_page=True)
        menu = mobile_page.locator("button.primary-toggle").first
        menu.click()
        assert mobile_page.locator("#pst-primary-sidebar-checkbox").is_checked()
        mobile_page.locator(".bd-sidebar-primary a[href='installation.html']").first.click()
        mobile_page.wait_for_url("**/installation.html")
        report["mobile_navigation"] = "passed"
        report["browser_errors"] = errors
        assert not errors, errors
        browser.close()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
