"""Capture full-page screenshots of every dashboard page with headless Edge over CDP.

Headless ``--screenshot`` fires before Streamlit finishes running the page script, so it
captures a blank page. This drives the browser over the DevTools protocol instead: it
navigates to each page, waits until Streamlit's running indicator has cleared and the page
text has stopped changing, grows the viewport to the full content height, then captures.

Usage: python scripts/capture_dashboard.py [base_url]  (with streamlit run app.py running)
Output: outputs/report/screenshots/NN_page.png plus manifest.json.
"""
import base64
import json
import pathlib
import subprocess
import sys
import time
import urllib.parse
import urllib.request

HERE = pathlib.Path(__file__).parent
try:
    import websocket  # websocket-client; only this script needs it
except ImportError:
    sys.exit("pip install websocket-client")

EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8501/"
ONLY = set(sys.argv[2].split(",")) if len(sys.argv) > 2 else None  # e.g. "05_simulation"
# Pages that compute nothing until buttons are pressed: file prefix -> labels, clicked in order.
CLICK_FIRST = {"05_simulation": ["Run the comparison", "Run the service-level sweep"]}
PORT = 9333
WIDTH = 1600
AS_OF = "2025-12-05"
OUT = HERE.parent / "outputs" / "report" / "screenshots"
PAGES = [
    ("01_overview", "Overview"),
    ("02_forecast", "Forecast"),
    ("03_recommendations", "Recommendations"),
    ("04_what_if", "What-if"),
    ("05_simulation", "Simulation & Evaluation"),
]

STATE_JS = """(() => {
  const main = document.querySelector('[data-testid="stMain"]') || document.querySelector('section.main') || document.body;
  const stop = [...document.querySelectorAll('button')].some(b => (b.innerText || '').trim() === 'Stop');
  return JSON.stringify({
    running: stop || !!document.querySelector('[data-testid="stStatusWidget"] [data-testid="stStatusWidgetRunningIcon"]'),
    spinners: document.querySelectorAll('[data-testid="stSpinner"]').length,
    exceptions: document.querySelectorAll('[data-testid="stException"]').length,
    headings: document.querySelectorAll('h1, h2').length,
    plots: document.querySelectorAll('.js-plotly-plot').length,
    text: (main.innerText || '').length,
    height: Math.max(main.scrollHeight || 0, document.body.scrollHeight || 0)
  });
})()"""


class CDP:
    """Minimal synchronous DevTools client: send a command, wait for its reply."""

    def __init__(self, ws_url):
        self.ws = websocket.create_connection(ws_url, timeout=60, suppress_origin=True)
        self.next_id = 0

    def call(self, method, **params):
        self.next_id += 1
        mid = self.next_id
        self.ws.send(json.dumps({"id": mid, "method": method, "params": params}))
        while True:
            msg = json.loads(self.ws.recv())
            if msg.get("id") == mid:
                if "error" in msg:
                    raise RuntimeError(f"{method}: {msg['error']}")
                return msg.get("result", {})

    def state(self):
        value = self.call("Runtime.evaluate", expression=STATE_JS, returnByValue=True)
        return json.loads(value["result"]["value"])


def wait_until_settled(cdp, timeout, stable_polls=3, interval=2.0):
    """Return the final page state once rendering has finished, or raise on timeout."""
    deadline = time.monotonic() + timeout
    last_text, stable, state = None, 0, {}
    while time.monotonic() < deadline:
        time.sleep(interval)
        state = cdp.state()
        busy = state["running"] or state["spinners"] or not state["headings"] or state["text"] < 200
        if busy:
            stable, last_text = 0, None
            continue
        stable = stable + 1 if state["text"] == last_text else 1
        last_text = state["text"]
        if stable >= stable_polls:
            return state
    raise TimeoutError(f"page did not settle within {timeout}s; last state {state}")


def main():
    OUT.mkdir(exist_ok=True)
    profile = HERE / "edge-cdp-profile"
    proc = subprocess.Popen(
        [EDGE, "--headless=new", f"--remote-debugging-port={PORT}", "--remote-allow-origins=*",
         f"--user-data-dir={profile}", "--disable-gpu", "--no-first-run", "--hide-scrollbars",
         f"--window-size={WIDTH},1100", "about:blank"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    manifest = []
    try:
        targets = None
        for _ in range(60):
            try:
                targets = json.load(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/json/list", timeout=2))
                break
            except Exception:  # noqa: BLE001 - browser still starting
                time.sleep(0.5)
        if not targets:
            raise RuntimeError("Edge DevTools endpoint never came up")
        page = next(t for t in targets if t.get("type") == "page")
        cdp = CDP(page["webSocketDebuggerUrl"])
        cdp.call("Page.enable")

        for name, label in PAGES:
            if ONLY and name not in ONLY:
                continue
            started = time.monotonic()
            query = urllib.parse.urlencode({"page": label, "as_of": AS_OF}, quote_via=urllib.parse.quote)
            url = f"{BASE}?{query}"
            cdp.call("Emulation.setDeviceMetricsOverride", width=WIDTH, height=1100, deviceScaleFactor=1, mobile=False)
            cdp.call("Page.navigate", url=url)
            state = wait_until_settled(cdp, timeout=420)
            for button in CLICK_FIRST.get(name, []):
                click_js = ("(() => { const b = [...document.querySelectorAll('button')]"
                            f".find(x => (x.innerText || '').trim() === {json.dumps(button)});"
                            " if (b) { b.click(); return true; } return false; })()")
                clicked = cdp.call("Runtime.evaluate", expression=click_js, returnByValue=True)["result"]["value"]
                if not clicked:
                    raise RuntimeError(f"{label}: button {button!r} not found")
                time.sleep(4)  # let the rerun start before judging whether it has settled
                state = wait_until_settled(cdp, timeout=900, stable_polls=4, interval=3.0)
            height = int(min(max(state["height"] + 80, 1100), 12000))
            cdp.call("Emulation.setDeviceMetricsOverride", width=WIDTH, height=height, deviceScaleFactor=1, mobile=False)
            state = wait_until_settled(cdp, timeout=120, stable_polls=2, interval=1.5)
            shot = cdp.call("Page.captureScreenshot", format="png")
            path = OUT / f"{name}.png"
            path.write_bytes(base64.b64decode(shot["data"]))
            entry = {"file": path.name, "page": label, "url": url, "seconds": round(time.monotonic() - started, 1),
                     "viewport_height": height, "bytes": path.stat().st_size, **state}
            manifest.append(entry)
            print(json.dumps(entry))
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
        manifest_path = OUT / "manifest.json"
        merged = {}
        if ONLY and manifest_path.exists():  # a partial run updates its pages and keeps the rest
            merged = {m["file"]: m for m in json.loads(manifest_path.read_text(encoding="utf-8"))}
        merged.update({m["file"]: m for m in manifest})
        manifest_path.write_text(json.dumps(sorted(merged.values(), key=lambda m: m["file"]), indent=2), encoding="utf-8")
    expected = len([p for p in PAGES if not ONLY or p[0] in ONLY])
    failed = [m for m in manifest if m["exceptions"]]
    print(f"captured {len(manifest)} of {expected} requested pages; pages showing a Streamlit exception: {[m['page'] for m in failed]}")
    return 1 if failed or len(manifest) != expected else 0


if __name__ == "__main__":
    sys.exit(main())
