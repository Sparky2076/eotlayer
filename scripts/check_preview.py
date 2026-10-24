"""Local HTML/link and browser checks. Requires beautifulsoup4 and playwright."""

from __future__ import annotations

import argparse
import ast
import copy
import json
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit

from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
HTML = ROOT / "docs/_build/html"
CAPTIONS = ["Documentation", "Examples", "Developer Documentation", "API Documentation"]


def check_links():
    pages = {path: BeautifulSoup(path.read_text(encoding="utf-8"), "html.parser")
             for path in HTML.rglob("*.html")}
    assert pages and (HTML / "index.html") in pages, f"Build the HTML documentation first: {HTML}"
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


def source_signature():
    """Audit the real constructor without importing torch or a solver backend."""
    tree = ast.parse((ROOT / "src/eotlayer/layer.py").read_text(encoding="utf-8"))
    layer = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "EOTLayer")
    constructor = next(node for node in layer.body
                       if isinstance(node, ast.FunctionDef) and node.name == "__init__")
    arguments = copy.deepcopy(constructor.args)
    arguments.args = arguments.args[1:]
    for argument in (*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs):
        argument.annotation = None
    signature = f"EOTLayer({ast.unparse(arguments)})"
    contract = json.loads((ROOT / "project-contract.json").read_text(encoding="utf-8"))
    assert signature == contract["product"]["signature"], "Constructor and project contract differ"
    return signature


def compact(value):
    return re.sub(r"\s|\u200b|\u00ad", "", value)


def check_navigation(page, path):
    captions = [text.strip() for text in page.locator(".wy-menu-vertical .caption-text").all_text_contents()]
    assert captions == CAPTIONS, f"Unexpected navigation groups: {captions}"
    current = page.locator(".wy-menu-vertical a.current[href]").evaluate_all(
        "links => links.map(link => new URL(link.href).pathname)")
    if path == "index.html":
        home = page.locator(".wy-side-nav-search a[href]").evaluate_all(
            "links => links.map(link => new URL(link.href).pathname)")
        assert any(value.endswith("/index.html") for value in home), "Sidebar logo must return home"
    else:
        assert any(value.endswith("/" + path) for value in current), f"No current-page link for {path}"
    return captions


def check_mobile_width(page, name):
    widths = page.evaluate("""() => ({
        viewport: innerWidth, document: document.documentElement.scrollWidth,
        body: document.body.scrollWidth
    })""")
    assert widths["viewport"] == 390, f"Mobile viewport metadata is incorrect: {widths}"
    assert widths["document"] <= widths["viewport"] + 1, f"{name} overflows: {widths}"
    assert widths["body"] <= widths["viewport"] + 1, f"{name} body overflows: {widths}"
    return widths


def check_code_scroll(page):
    """A long code line scrolls inside its block, without widening the page."""
    return page.locator(".highlight, .highlight pre, .literal-block-wrapper").evaluate_all("""blocks => {
        for (const block of blocks) {
            const style = getComputedStyle(block);
            if (block.scrollWidth > block.clientWidth + 2 && ['auto', 'scroll'].includes(style.overflowX)) {
                const initial = block.scrollLeft;
                block.scrollLeft = Math.min(50, block.scrollWidth - block.clientWidth);
                const moved = block.scrollLeft > initial;
                block.scrollLeft = initial;
                if (moved) return {tag: block.tagName, class: block.className,
                    clientWidth: block.clientWidth, scrollWidth: block.scrollWidth};
            }
        }
        return null;
    }""")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8768")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report = check_links()
    report["source_constructor"] = source_signature()
    print("Internal HTML links and anchors passed", flush=True)
    print("Starting Playwright driver", flush=True)
    with sync_playwright() as playwright:
        print("Launching Chromium", flush=True)
        browser = playwright.chromium.launch(headless=True, timeout=15000)
        print("Chromium ready", flush=True)
        context = browser.new_context(viewport={"width": 1440, "height": 1000}, device_scale_factor=1,
                                      permissions=["clipboard-read", "clipboard-write"])
        errors = []

        def watch(page):
            page.on("pageerror", lambda error: errors.append({"url": page.url, "type": "pageerror", "message": str(error)}))
            page.on("console", lambda message: errors.append({"url": page.url, "type": "console", "message": message.text})
                    if message.type == "error" else None)

        page = context.new_page()
        watch(page)
        response = page.goto(args.url + "/index.html", wait_until="networkidle")
        assert response.status == 200
        assert page.title() == "EOTLayer — EOTLayer Documentation"
        assert page.locator(".wy-nav-side").is_visible()
        assert page.locator(".wy-side-nav-search input[name='q']").is_visible()
        logo = page.locator(".wy-side-nav-search img.logo")
        assert logo.is_visible() and logo.evaluate("image => image.complete && image.naturalWidth > 0")
        assert "eotlayer-logo.svg" in logo.get_attribute("src")
        report["navigation_groups"] = check_navigation(page, "index.html")
        report["desktop_style"] = page.evaluate("""() => ({
            searchBackground: getComputedStyle(document.querySelector('.wy-side-nav-search')).backgroundColor,
            sidebarBackground: getComputedStyle(document.querySelector('.wy-nav-side')).backgroundColor,
            fontFamily: getComputedStyle(document.body).fontFamily,
            contentWidth: document.querySelector('.wy-nav-content').getBoundingClientRect().width
        })""")
        assert report["desktop_style"]["searchBackground"] in {"rgb(255, 255, 255)", "rgba(255, 255, 255, 1)"}
        sidebar_rgb = [int(value) for value in re.findall(r"\d+", report["desktop_style"]["sidebarBackground"])[:3]]
        assert max(sidebar_rgb) < 100, "Sidebar must use the intended dark RTD background"
        assert "Lato" in report["desktop_style"]["fontFamily"]
        assert report["desktop_style"]["contentWidth"] >= 850
        assert page.locator(".copybtn").count() >= 1
        button = page.locator(".copybtn").first
        button.locator("..").hover()
        button.click()
        copied = page.evaluate("navigator.clipboard.readText()")
        assert "pip install -e ." in copied, f"Copy button did not copy the install command: {copied!r}"
        report["code_copy"] = "passed"
        page.screenshot(path=str(args.output / "homepage-desktop.png"), full_page=False)
        page.screenshot(path=str(args.output / "homepage-full.png"), full_page=True)
        page.goto(args.url + "/api/layer.html", wait_until="networkidle")
        signature = page.locator("#eotlayer\\.EOTLayer")
        assert signature.count() == 1
        assert compact(report["source_constructor"]) in compact(signature.inner_text()), signature.inner_text()
        assert "epsilon_scaling_start" in page.locator(".rst-content").inner_text()
        check_navigation(page, "api/layer.html")
        report["rendered_constructor_matches_source"] = True
        page.screenshot(path=str(args.output / "api-desktop.png"), full_page=False)
        page.goto(args.url + "/key_concepts/optimal_transport.html", wait_until="networkidle")
        page.wait_for_selector("mjx-container", timeout=30000)
        report["mathjax_rendered"] = page.locator("mjx-container").count()
        page.screenshot(path=str(args.output / "concepts-desktop.png"), full_page=False)
        page.goto(args.url + "/search.html?q=envelope", wait_until="networkidle")
        page.wait_for_selector("#search-results li", timeout=20000)
        report["search_results"] = page.locator("#search-results li").count()
        assert "envelope" in page.locator("#search-results").inner_text().lower()
        page.screenshot(path=str(args.output / "search-desktop.png"), full_page=False)
        print("Desktop, navigation, source API, copy, MathJax, and search passed", flush=True)
        mobile = browser.new_context(viewport={"width": 390, "height": 844}, device_scale_factor=1, is_mobile=True, has_touch=True)
        mobile_page = mobile.new_page()
        watch(mobile_page)
        mobile_page.goto(args.url + "/index.html", wait_until="networkidle")
        report["mobile_home_width"] = check_mobile_width(mobile_page, "Mobile homepage")
        report["mobile_code_scroll"] = check_code_scroll(mobile_page)
        assert report["mobile_code_scroll"], "Long code lines must scroll inside their own block"
        mobile_page.screenshot(path=str(args.output / "homepage-mobile.png"), full_page=True)
        menu = mobile_page.locator("[data-toggle='wy-nav-top']").first
        menu.click()
        mobile_page.wait_for_function("document.querySelector('.wy-nav-side').getBoundingClientRect().left >= 0")
        check_navigation(mobile_page, "index.html")
        mobile_page.screenshot(path=str(args.output / "navigation-mobile.png"), full_page=False)
        mobile_page.locator(".wy-menu-vertical a[href='installation.html']").first.click()
        mobile_page.wait_for_url("**/installation.html")
        mobile_page.wait_for_load_state("networkidle")
        check_navigation(mobile_page, "installation.html")
        report["mobile_installation_width"] = check_mobile_width(mobile_page, "Mobile installation")
        mobile_page.locator("[data-toggle='wy-nav-top']").first.click()
        mobile_page.wait_for_function("document.querySelector('.wy-nav-side').getBoundingClientRect().left >= 0")
        # RTD expands the current section. Enter the visible API parent before
        # selecting its child, just as a reader does on a narrow screen.
        mobile_page.locator(".wy-menu-vertical a[href='api/index.html']").first.click()
        mobile_page.wait_for_url("**/api/index.html")
        mobile_page.wait_for_load_state("networkidle")
        check_navigation(mobile_page, "api/index.html")
        mobile_page.locator("[data-toggle='wy-nav-top']").first.click()
        mobile_page.wait_for_function("document.querySelector('.wy-nav-side').getBoundingClientRect().left >= 0")
        mobile_page.locator(".wy-menu-vertical a[href='layer.html']").first.click()
        mobile_page.wait_for_url("**/api/layer.html")
        mobile_page.wait_for_load_state("networkidle")
        report["mobile_api_width"] = check_mobile_width(mobile_page, "Mobile API")
        check_navigation(mobile_page, "api/layer.html")
        mobile_page.screenshot(path=str(args.output / "api-mobile.png"), full_page=True)
        report["mobile_navigation"] = "passed"
        report["mobile_page_widths"] = {}
        report["mobile_width_failures"] = []
        for html_page in sorted(HTML.rglob("*.html")):
            relative = html_page.relative_to(HTML).as_posix()
            mobile_page.goto(args.url + "/" + relative, wait_until="networkidle")
            mobile_page.evaluate("async () => { if (window.MathJax?.startup?.promise) await window.MathJax.startup.promise; }")
            widths = mobile_page.evaluate("""() => ({viewport: innerWidth,
                document: document.documentElement.scrollWidth, body: document.body.scrollWidth})""")
            report["mobile_page_widths"][relative] = widths
            if widths["viewport"] != 390 or max(widths["document"], widths["body"]) > 391:
                report["mobile_width_failures"].append({"page": relative, **widths})
        report["browser_errors"] = errors
        browser.close()
    (args.output / "qa-report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    assert not report["browser_errors"], report["browser_errors"]
    assert not report["mobile_width_failures"], report["mobile_width_failures"]
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
