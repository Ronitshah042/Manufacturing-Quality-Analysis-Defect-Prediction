"""
Open the deployed Streamlit app in a real browser, wake it if it is asleep,
and fail if it doesn't show its title. Run by .github/workflows/keep-awake.yml.

Usage:
    pip install playwright && playwright install chromium
    python scripts/keep_awake.py [APP_URL]
"""

import os
import sys

from playwright.sync_api import sync_playwright

APP_URL = "https://manufacturing-quality-analysis-defect-prediction-mkbvjywxtqkhq.streamlit.app"
APP_TITLE = "Manufacturing Quality & Predictive Maintenance"
WAKE_BUTTON = "Yes, get this app back up"


def main(url: str) -> int:
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=os.environ.get("CHROME_PATH") or None)
        page = browser.new_page()
        page.goto(url, wait_until="domcontentloaded", timeout=120_000)

        wake = page.get_by_role("button", name=WAKE_BUTTON)
        try:
            wake.wait_for(timeout=15_000)
            print("App was asleep; waking it.")
            wake.click()
        except Exception:
            print("App was already awake.")

        # On Community Cloud the app runs inside an iframe; check both the page and its frames.
        for _ in range(36):  # up to 6 minutes for a cold start
            for frame in page.frames:
                try:
                    if frame.get_by_text(APP_TITLE).first.is_visible():
                        print(f"OK: '{APP_TITLE}' is visible at {url}")
                        browser.close()
                        return 0
                except Exception:
                    pass
            page.wait_for_timeout(10_000)

        print(f"FAILED: '{APP_TITLE}' did not appear at {url}", file=sys.stderr)
        browser.close()
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else APP_URL))
