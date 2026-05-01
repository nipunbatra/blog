"""Drive the Streamlit app and record a short demo video.

Captures pose + seg on one RGB and one thermal sample each, with an
on-screen banner explaining what's being shown.
"""

import time
from pathlib import Path

from playwright.sync_api import sync_playwright

URL = "http://localhost:18511/"
OUT = Path(__file__).parent / "demo_recording"
OUT.mkdir(exist_ok=True)

SCRIPT = [
    ("POSE", "person1.jpg",        "Pose · 308 keypoints · RGB portrait"),
    ("POSE", "thermal_fae.jpg",    "Pose · 308 keypoints · thermal IR (out-of-distribution)"),
    ("SEG",  "desk_worker.jpg",    "Body-part segmentation · 29 classes · RGB"),
    ("SEG",  "thermal_mensch.jpg", "Body-part segmentation · 29 classes · thermal IR"),
]

BANNER_JS = """
(text) => {
  let el = document.getElementById('demo-banner');
  if (!el) {
    el = document.createElement('div');
    el.id = 'demo-banner';
    el.style.cssText = `
      position: fixed; top: 14px; left: 50%; transform: translateX(-50%);
      background: rgba(196,69,54,0.95); color: white;
      font: 600 18px -apple-system,Segoe UI,Roboto,sans-serif;
      padding: 10px 22px; border-radius: 6px; z-index: 9999;
      box-shadow: 0 4px 14px rgba(0,0,0,0.25);
      pointer-events: none;
    `;
    document.body.appendChild(el);
  }
  el.textContent = text;
}
"""


def set_banner(page, text):
    page.evaluate(BANNER_JS, text)


def select_task(page, label):
    sb = page.locator("section[data-testid='stSidebar']")
    sb.locator("div[data-baseweb='select']").first.click()
    page.locator("li[role='option']", has_text=label).first.click()


def click_sample(page, fname):
    sb = page.locator("section[data-testid='stSidebar']")
    sb.get_by_role("button", name=fname, exact=True).click()


def wait_for_inference(page, max_s: float = 60.0):
    """Wait for the running indicator to clear after a click."""
    spinner = page.locator("div[data-testid='stSpinner'], div[data-testid='stStatusWidget']")
    # Give the spinner up to 4s to appear
    end = time.time() + 4
    while time.time() < end:
        if spinner.count() > 0:
            break
        page.wait_for_timeout(150)
    # Then wait up to max_s for it to disappear
    end = time.time() + max_s
    while time.time() < end:
        if spinner.count() == 0:
            return time.time() - (end - max_s)
        page.wait_for_timeout(250)
    return max_s


def main():
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        ctx = browser.new_context(
            viewport={"width": 1440, "height": 900},
            record_video_dir=str(OUT),
            record_video_size={"width": 1440, "height": 900},
        )
        page = ctx.new_page()
        page.goto(URL, wait_until="domcontentloaded", timeout=30_000)
        page.wait_for_selector("section[data-testid='stSidebar']", timeout=20_000)
        page.wait_for_timeout(2000)

        set_banner(page, "Sapiens2 try-it app · 1B models on Bhaskar (RTX A5000)")
        page.wait_for_timeout(2500)

        for task, fname, caption in SCRIPT:
            print(f"[demo] {task}  {fname}")
            set_banner(page, caption)
            page.wait_for_timeout(800)
            select_task(page, task)
            page.wait_for_timeout(600)
            click_sample(page, fname)
            t = wait_for_inference(page, max_s=90)
            print(f"  inference {t:.1f}s")
            # Linger so the result stays visible
            page.wait_for_timeout(3500)

        set_banner(page, "Try your own image — drop in /DATA/nipun.batra/sapiens2/samples/")
        page.wait_for_timeout(3500)

        ctx.close()
        browser.close()
        vids = sorted(OUT.glob("*.webm"))
        print(f"[done] video(s): {vids}")


if __name__ == "__main__":
    main()
