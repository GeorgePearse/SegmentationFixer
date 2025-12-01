import time
from playwright.sync_api import sync_playwright


def run():
    with sync_playwright() as p:
        print("Launching browser...")
        browser = p.chromium.launch()
        page = browser.new_page()

        print("Navigating to frontend...")
        try:
            page.goto("http://localhost:5173", timeout=10000)
        except Exception as e:
            print(f"Failed to load page: {e}")
            browser.close()
            return

        # Wait for the app to connect and receive data
        # We expect "Review Correction" to appear when a proposal is loaded
        print("Waiting for 'Review Correction' header...")
        try:
            page.wait_for_selector("text=Review Correction", timeout=20000)
            print("Found 'Review Correction' header!")
        except Exception:
            print("Timeout waiting for data. Capturing current state...")
            page.screenshot(path="debug_timeout.png")
            content = page.content()
            print("Page content:", content[:1000])  # Print first 1000 chars
            browser.close()
            return

        # Wait for image and SVG
        page.wait_for_selector("img[alt='Target']")
        page.wait_for_selector("polygon")

        print("Taking initial screenshot...")
        page.screenshot(path="screenshot_1_initial.png")

        # Get current ID
        try:
            info_text = page.locator(".info-item").nth(1).inner_text()
            print(f"Initial State: {info_text}")
        except:
            print("Could not get info text")

        # Press 'I' to Accept
        print("Pressing 'I'...")
        page.keyboard.press("i")

        # Wait for change (simple wait for demo purposes, ideally we check ID change)
        time.sleep(1)

        print("Taking second screenshot...")
        page.screenshot(path="screenshot_2_next.png")

        try:
            info_text_2 = page.locator(".info-item").nth(1).inner_text()
            print(f"Next State: {info_text_2}")
        except:
            print("Could not get second info text")

        browser.close()
        print("Test complete.")


if __name__ == "__main__":
    run()
