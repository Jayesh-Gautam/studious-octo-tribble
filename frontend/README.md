# AeroFix Console (React + shadcn/ui)

Presentation-grade web front end for AeroFix. It talks to the FastAPI service in `src/aerofix/app/api.py`.

```powershell
npm install          # once
npm run build        # writes dist/, which FastAPI serves at http://localhost:8000
npm run dev          # optional: hot-reload dev server on :5173 (proxies /api to :8000)
```

Pages: Fault advisor (with in-app FAA document viewer), Feedback log, Model performance, Dataset, About.
Deep links for demos: `/?q=FWD%20lav%20smoke%20detector%20inop&family=B737&variant=737-800&tab=documents`, `/?page=performance`.
