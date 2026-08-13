"""One-off script to visually verify the chat UI actually works: loads the page,
screenshots it empty, sends a message via the real UI (clicking a suggestion chip),
waits for a real response from the live backend, and screenshots the result.
Not part of the app or the automated test suite - a manual verification aid.
"""

from playwright.sync_api import sync_playwright

OUT_DIR = "eval/ui_screenshots"

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 900, "height": 700})

    console_errors = []
    page.on("console", lambda msg: console_errors.append(msg.text) if msg.type == "error" else None)

    page.goto("http://127.0.0.1:8000/")
    page.wait_for_selector("text=SafarIA Assistant")
    page.screenshot(path=f"{OUT_DIR}/1_empty.png")
    print("Screenshot 1 (empty state) saved")

    page.click("text=What are the cabin baggage rules?")
    # Either a real answer (.badge appears) or a graceful error bubble (.bubble.error) -
    # the OpenRouter account may be out of credits, in which case this verifies the
    # error path renders properly instead of a blank/broken response.
    page.wait_for_selector(".badge, .bubble.error", timeout=30000)
    page.wait_for_timeout(500)
    page.screenshot(path=f"{OUT_DIR}/2_after_response.png")
    print("Screenshot 2 (after response) saved")

    answer_text = page.locator(".row.bot .bubble").last.text_content()
    print(f"Answer/error shown: {answer_text}")
    if page.locator(".badge").count() > 0:
        print(f"Badge: {page.locator('.badge').last.text_content()}")
    if page.locator(".bubble.error").count() > 0:
        print("(Rendered as a graceful error bubble, not a crash)")

    print(f"Console errors: {console_errors}")

    browser.close()
