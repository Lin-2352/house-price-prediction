---
name: verify
description: How to build, launch, and drive this Streamlit app for runtime verification.
---

# Verifying this repo (Streamlit app)

Surface: browser GUI (pixels), served by `streamlit run app.py`. The
`src/` package underneath is not itself a surface — always drive through
the running app, not by importing pipeline functions directly.

## Launch

```bash
export PATH="$PATH:/c/Users/Harsh Raj/AppData/Roaming/Python/Python313/Scripts"  # streamlit.exe lives here on this machine
cd /d/Desktop/Project/house_price_prediction
nohup streamlit run app.py --server.headless true --server.port 85XX > streamlit_verify.log 2>&1 &
disown
```
Use a fresh port per verification run (8501/8502/8503 have been used
already) to avoid colliding with an app instance left running from a
prior session. Wait for `Local URL:` in the log (~5-10s), then navigate
the browser tool to `http://localhost:85XX`.

Requires `artifacts/india/model_bundle.joblib` to already exist (produced
by `python -m src.pipeline --market india`) — the app only loads
persisted artifacts, it does not train anything itself.

## Known tool limitation: screenshots of Plotly/canvas content come back blank

The `computer` screenshot/zoom action reliably returns a **blank/background
image over any Plotly SVG chart or `st.dataframe` (glide-data-grid canvas)
region** — both on localhost and on the deployed Streamlit Cloud iframe.
This is a CDP `Page.captureScreenshot` compositing limitation in this
environment, not an app bug. Confirmed via: correct DOM geometry, correct
`fill`/`opacity` styles, and in-page canvas rasterization sampling showing
real non-background pixel data. Do **not** conclude "chart is broken" from
a blank screenshot alone.

To actually verify chart/table content, use `javascript_tool` instead of
screenshots:

```js
// Plotly charts: serialize the SVG, draw it into a canvas, sample pixels
const plots = document.querySelectorAll('.js-plotly-plot');
for (const p of plots) {
  const svg = p.querySelector('svg.main-svg');
  const xml = new XMLSerializer().serializeToString(svg);
  const url = URL.createObjectURL(new Blob([xml], {type: 'image/svg+xml;charset=utf-8'}));
  const img = new Image();
  img.onload = () => { /* draw to canvas, ctx.getImageData, count non-background pixels */ };
  img.src = url;
}

// st.dataframe grids render straight to <canvas> -- read pixels directly, no SVG step needed
document.querySelectorAll('[data-testid="stDataFrame"] canvas')
```
A `[data-testid="stException"]` count of 0 plus non-trivial non-background
pixel counts is solid evidence the chart/table actually rendered.

Plain text content (metrics, headings, alerts) renders fine in normal
screenshots and in `get_page_text` -- only canvas/SVG-backed widgets hit
this issue.

## Driving the app (no visible "Estimate price" button reliably found via `find`)

The button is often below the fold and the accessibility tree / `find`
tool doesn't always locate it. Click via `javascript_tool` instead:
```js
Array.from(document.querySelectorAll('button')).find(b => b.innerText.trim() === 'Estimate price').click();
```
Same pattern for radio options (`Split-conformal`, market switches) and
checkboxes (`Unknown / skip`) -- locate via `label`/`button` innerText
match and `.click()`, then `wait` ~3s for the Streamlit rerun before
reading state.

If the app is embedded in an iframe (true on the deployed
`*.streamlit.app` URL, not on localhost), everything above must be
scoped through `document.querySelector('iframe').contentDocument`.

## Useful checks after driving an action

- `document.querySelectorAll('[data-testid="stException"]').length` -- 0 expected
- `document.querySelectorAll('[data-testid="stMetricValue"]')` -- predicted price / range values
- `document.querySelectorAll('[data-testid="stAlert"]')` -- confidence banner text
- Server-side errors: `grep -iE "error|traceback|exception" streamlit_verify.log`

## What's worth probing (beyond the happy path)

- Toggle "Unknown / skip" on an important field (e.g. Area/City/Location)
  and re-estimate -- checks the missing-field confidence signal and that
  `SimpleImputer` handles the resulting NaN without crashing. Note: by
  design a single missing field alone does NOT trip the low-confidence
  flag (needs 2-of-3 signals or one extreme signal) -- don't mistake that
  for a bug.
- Switch "Range method" between CQR and split-conformal and re-estimate --
  ranges should differ (not stale/cached) since they're different
  conformal objects.
- Rapid repeated clicks on "Estimate price" -- should not throw.
- Expand the "Model performance & validation" panel and spot-check a
  number (e.g. MAE/MAPE) against `artifacts/india/metrics.json` directly
  (`cat artifacts/india/metrics.json`) -- the app panel must match exactly
  since it's the same file.

## Cleanup

```bash
kill $(ps aux | grep '[s]treamlit run' | awk '{print $2}')
```
