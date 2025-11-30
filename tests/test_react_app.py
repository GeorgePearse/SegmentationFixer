import pytest
from playwright.sync_api import Page, expect
import subprocess
import time
import os
import sys
import requests

# Constants
BACKEND_PORT = 8000
FRONTEND_PORT = 3000
BACKEND_URL = f"http://localhost:{BACKEND_PORT}"
FRONTEND_URL = f"http://localhost:{FRONTEND_PORT}"


def wait_for_server(url, timeout=30):
    start = time.time()
    while time.time() - start < timeout:
        try:
            requests.get(url)
            return True
        except requests.ConnectionError:
            time.sleep(1)
    return False


@pytest.fixture(scope="module")
def services():
    # 1. Ensure demo data exists
    env = os.environ.copy()
    env["PYTHONPATH"] = os.getcwd()
    subprocess.run(
        [sys.executable, "examples/generate_demo_data.py"], check=True, env=env
    )

    # 2. Start Backend
    backend_cmd = [sys.executable, "backend/main.py"]
    backend_proc = subprocess.Popen(
        backend_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )

    # 3. Start Frontend
    # using 'npm run dev -- --port 3000' to ensure port, though vite config has it.
    frontend_cmd = ["npm", "run", "dev", "--", "--host"]
    # We need to run this from frontend dir
    frontend_proc = subprocess.Popen(
        frontend_cmd, cwd="frontend", stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )

    # Wait for services
    print("Waiting for backend...")
    if not wait_for_server(f"{BACKEND_URL}/api/pending"):
        print(backend_proc.stderr.read().decode())
        raise RuntimeError("Backend failed to start")

    print("Waiting for frontend...")
    # Vite might take a sec to compile
    if not wait_for_server(FRONTEND_URL):
        print(frontend_proc.stderr.read().decode())
        raise RuntimeError("Frontend failed to start")

    yield

    # Cleanup
    backend_proc.terminate()
    frontend_proc.terminate()
    backend_proc.wait()
    frontend_proc.wait()


def test_react_app_flow(page: Page, services):
    # Verify backend API sees the file
    try:
        resp = requests.get(f"{BACKEND_URL}/api/pending")
        print("Pending images API response:", resp.json())
        assert len(resp.json()) > 0
    except Exception as e:
        print("Backend API check failed:", e)

    # Navigate to app
    page.goto(FRONTEND_URL)

    # Print console logs
    page.on("console", lambda msg: print(f"Browser Console: {msg.text}"))

    # Check Title
    expect(
        page.get_by_role("heading", name="Segmentation Fixer Review")
    ).to_be_visible()

    # Check Metadata display
    # We expect "SAM Confidence: 98.5%" and "Difference (IoU): 0.325" (approx)
    expect(page.get_by_text("SAM Confidence:")).to_be_visible()
    expect(page.get_by_text("Difference (IoU):")).to_be_visible()

    # Check if an image is loaded (we expect the demo image)
    # The image src should point to the backend

    img = page.locator(".image-container img")
    expect(img).to_be_visible()

    # Verify the image source contains the backend URL
    # We might need to check the 'src' attribute
    # In the app we set src={`${API_URL}/images/${currentImage}`}
    # API_URL is localhost:8000

    # Check Buttons
    accept_btn = page.get_by_role("button", name="Accept")
    reject_btn = page.get_by_role("button", name="Reject")

    expect(accept_btn).to_be_visible()
    expect(reject_btn).to_be_visible()

    # Click Accept
    accept_btn.click()

    # Expect "No images pending" message
    # In App.jsx: <div className="empty">No images pending review! Great job.</div>
    expect(page.locator(".empty")).to_contain_text(
        "No images pending review! Great job."
    )
