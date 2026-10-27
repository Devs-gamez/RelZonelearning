# Rel Zone (Python)

    pip install -r requirements.txt
    python app.py          # then open http://127.0.0.1:5000

- `app.py`            Flask server: serves the hub, `/proxy` (rewriting web proxy), `/api/chat` (Groq, streamed)
- `static/index.html` The hub UI (games, piano, AI, web, settings)
- `.env`              `GROQ_API_KEY=...` (server-side only; never sent to the browser)

Optional env vars: `HOST`, `PORT`, `GROQ_MODEL`.
To reach it from other devices on your network: `HOST=0.0.0.0 python app.py`
(the proxy is open to anyone who can reach the server, so don't expose it publicly without adding a login).
