"""Log in to job sites once, so find_jobs / fetch_job can read pages that need a login.

    python login.py                 # opens BOSS直聘, 猎聘 and 牛客
    python login.py https://...     # or any site you want

A normal Chrome window opens. Log in yourself (phone code, QR code...), then come back to this
terminal and press Enter. Only the browser's cookies are saved, in data/browser/ (the password is
never seen or stored). Delete that folder to log out. Close the Job Agent app while doing this.
"""
import sys

from jobagent import config

SITES = ["https://www.zhipin.com/web/user/?ka=header-login", "https://www.liepin.com/",
         "https://www.nowcoder.com/login"]


def main():
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        sys.exit("Playwright is not installed: python -m pip install playwright, "
                 "then python -m playwright install chromium")
    urls = sys.argv[1:] or SITES
    with sync_playwright() as p:
        ctx = p.chromium.launch_persistent_context(config.BROWSER_PROFILE, headless=False,
                                                   viewport={"width": 1200, "height": 800})
        page = ctx.pages[0] if ctx.pages else ctx.new_page()
        page.goto(urls[0])
        for url in urls[1:]:
            ctx.new_page().goto(url)
        input("Log in in the browser window (each tab), then press Enter here to save and close... ")
        ctx.close()
    print(f"Saved. The app will now open job pages with your login (folder: {config.BROWSER_PROFILE}).")


if __name__ == "__main__":
    main()
