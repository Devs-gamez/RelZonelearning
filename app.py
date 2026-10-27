"""Rel Zone: a small Flask site with a rewriting web proxy and a Groq-powered AI chat.

Run:  pip install -r requirements.txt && python app.py   ->  http://127.0.0.1:5000
"""
import ipaddress
import json
import os
import re
import secrets
import socket
import time
from collections import defaultdict, deque
from urllib.parse import parse_qs, quote, urlencode, urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from flask import Flask, Response, jsonify, request, send_from_directory, stream_with_context
from markupsafe import escape

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_env():
    path = os.path.join(BASE_DIR, ".env")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip("\"'"))


load_env()

GROQ_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_URL = os.environ.get("GROQ_URL", "https://api.groq.com/openai/v1/chat/completions")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
# If the first model is retired/unavailable, the server tries these in order.
FALLBACK_MODELS = ["openai/gpt-oss-20b", "llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
_working = {"model": None}
ALLOW_PRIVATE = os.environ.get("ALLOW_PRIVATE") == "1"  # only for local testing
MAX_HTML = 5 * 1024 * 1024
MAX_ASSET = 15 * 1024 * 1024
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")
SYS = ("You are Rel AI, a friendly, concise assistant inside a site called Rel Zone. "
       "Help with questions, homework, writing and brainstorming. Show your reasoning on math. "
       "Use plain text, no markdown tables.")

app = Flask(__name__, static_folder=None)

# ---------------------------------------------------------------- rate limit
HITS = defaultdict(deque)


def limited(bucket, limit, window):
    ip = request.remote_addr or "?"
    now = time.time()
    q = HITS[(bucket, ip)]
    while q and now - q[0] > window:
        q.popleft()
    if len(q) >= limit:
        return True
    q.append(now)
    return False


# ---------------------------------------------------------------- pages
VERSION = "4"


@app.get("/")
def index():
    r = send_from_directory(os.path.join(BASE_DIR, "static"), "index.html")
    r.headers["Cache-Control"] = "no-store"  # never serve a stale copy of the page
    return r


@app.get("/api/health")
def health():
    """Open http://127.0.0.1:5000/api/health to see exactly why the AI isn't working."""
    info = {"version": VERSION, "key_loaded": bool(GROQ_KEY), "model": GROQ_MODEL}
    if not GROQ_KEY:
        info["problem"] = "GROQ_API_KEY not found. Put it in a file named .env next to app.py."
        return jsonify(info)
    if limited("health", 10, 60):
        info["problem"] = "Too many checks, wait a minute."
        return jsonify(info), 429
    try:
        r, got = groq_call([{"role": "user", "content": "hi"}], False, 64)
        if r is not None:
            r.close()
            info["groq_status"] = 200
            info["model_in_use"] = got
            info["result"] = "OK, Groq accepted the key and the model. The AI should work."
        else:
            info["groq_status"], info["problem"] = got
    except requests.RequestException as ex:
        info["problem"] = "Server couldn't reach Groq: %s" % ex
    return jsonify(info)


# ---------------------------------------------------------------- AI
def error_detail(r):
    try:
        return (r.json().get("error") or {}).get("message", "") or r.text[:200]
    except ValueError:
        return r.text[:200]


def model_missing(status, detail):
    d = (detail or "").lower()
    return status in (400, 404) and ("model" in d) and any(
        w in d for w in ("does not exist", "decommission", "no access", "not have access", "deprecated", "not found"))


def groq_call(messages, stream, max_tokens):
    """Call Groq, falling back to other models if one has been retired. Returns (response, model)."""
    order = [m for m in [_working["model"], GROQ_MODEL] + FALLBACK_MODELS if m]
    seen, last = set(), None
    for model in order:
        if model in seen:
            continue
        seen.add(model)
        body = {"model": model, "messages": messages, "stream": stream, "max_tokens": max_tokens}
        if model.startswith("openai/gpt-oss"):
            body["reasoning_effort"] = "low"  # keep replies fast; reasoning tokens count toward max_tokens
        else:
            body["temperature"] = 0.7
        r = requests.post(GROQ_URL, json=body, stream=stream, timeout=(5, 60),
                          headers={"Authorization": "Bearer " + GROQ_KEY})
        if r.status_code == 200:
            if _working["model"] != model:
                print("Using Groq model:", model)
            _working["model"] = model
            return r, model
        detail = error_detail(r)
        r.close()
        last = (r.status_code, detail)
        if model_missing(r.status_code, detail):
            print("! Groq model %s unavailable (%s). Trying next..." % (model, detail))
            continue
        break
    return None, last


@app.post("/api/chat")
def chat():
    if not GROQ_KEY:
        return jsonify(error="no_key"), 500
    if limited("chat", 20, 60):
        return jsonify(error="rate_limited"), 429
    data = request.get_json(silent=True) or {}
    msgs = []
    for m in (data.get("messages") or [])[-12:]:
        if isinstance(m, dict) and m.get("role") in ("user", "assistant") and m.get("content"):
            msgs.append({"role": m["role"], "content": str(m["content"])[:4000]})
    if not msgs or msgs[-1]["role"] != "user":
        return jsonify(error="bad_request"), 400
    try:
        up, info = groq_call([{"role": "system", "content": SYS}] + msgs, True, 2048)
    except requests.RequestException as ex:
        print("! Couldn't reach Groq:", ex)
        return jsonify(error="upstream_unreachable", detail="Server couldn't reach Groq (network/firewall?)"), 502
    if up is None:
        status, detail = info
        print("! Groq returned %s: %s" % (status, detail))
        code = status if status in (401, 403, 429) else 502
        return jsonify(error="upstream", status=status, detail=detail), code

    def gen():
        try:
            for chunk in up.iter_content(chunk_size=None):
                if chunk:
                    yield chunk
        finally:
            up.close()

    return Response(stream_with_context(gen()), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ---------------------------------------------------------------- proxy
def purl(u):
    return "/proxy?url=" + quote(u, safe="")


def unwrap(u):
    """Skip search-engine click-tracking redirects."""
    try:
        p = urlparse(u)
        h = p.hostname or ""
        q = parse_qs(p.query)
        if h.endswith("duckduckgo.com") and p.path == "/l/" and q.get("uddg"):
            return q["uddg"][0]
        if "google." in h and p.path == "/url" and q.get("q"):
            return q["q"][0]
    except ValueError:
        pass
    return u


def is_safe(url):
    p = urlparse(url)
    if p.scheme not in ("http", "https") or not p.hostname:
        return False
    if ALLOW_PRIVATE:
        return True
    try:
        infos = socket.getaddrinfo(p.hostname, p.port or (443 if p.scheme == "https" else 80),
                                   proto=socket.IPPROTO_TCP)
    except (socket.gaierror, UnicodeError):
        return False
    return all(ipaddress.ip_address(i[4][0]).is_global for i in infos)


def fetch(method, url, data=None):
    """Fetch with manual redirects so every hop is checked (blocks SSRF to private IPs)."""
    for _ in range(6):
        if not is_safe(url):
            raise ValueError("That address isn't allowed.")
        r = requests.request(method, url, data=data, allow_redirects=False, stream=True,
                             timeout=(5, 20),
                             headers={"User-Agent": UA, "Accept-Language": "en-US,en;q=0.9",
                                      "Accept": "text/html,application/xhtml+xml,*/*;q=0.8"})
        loc = r.headers.get("Location")
        if r.status_code in (301, 302, 303, 307, 308) and loc:
            url = urljoin(url, loc)
            if r.status_code in (301, 302, 303):
                method, data = "GET", None
            r.close()
            continue
        return r, url
    raise ValueError("Too many redirects.")


def read_limited(r, limit):
    buf, size = [], 0
    for chunk in r.iter_content(65536):
        size += len(chunk)
        if size > limit:
            break
        buf.append(chunk)
    r.close()
    return b"".join(buf)


SKIP = re.compile(r"^(javascript:|mailto:|tel:|data:|blob:|about:|#)", re.I)
CSS_URL = re.compile(r"""url\(\s*(["']?)(.*?)\1\s*\)""", re.I)
CSS_IMPORT = re.compile(r"""@import\s+(["'])(.*?)\1""", re.I)


def rw(u, base):
    u = (u or "").strip()
    if not u or SKIP.match(u):
        return u
    return purl(unwrap(urljoin(base, u)))


def rw_css(css, base):
    css = CSS_URL.sub(lambda m: "url(%s%s%s)" % (m.group(1), rw(m.group(2), base), m.group(1)), css)
    return CSS_IMPORT.sub(lambda m: "@import %s%s%s" % (m.group(1), rw(m.group(2), base), m.group(1)), css)


def rw_srcset(v, base):
    out = []
    for part in re.split(r",\s+", v.strip()):
        bits = part.strip().split()
        if bits:
            bits[0] = rw(bits[0].rstrip(","), base)
            out.append(" ".join(bits))
    return ", ".join(out)


def reporter(url, nonce):
    """Tells the Rel Zone shell which page the frame is showing (for the address bar)."""
    js = json.dumps(url).replace("</", "<\\/")
    return ("<script nonce=\"%s\">(function(){try{parent.postMessage({rz:1,url:%s,title:document.title},'*')}"
            "catch(e){}})();</script>" % (nonce, js))


URL_ATTRS = {"a": ["href"], "area": ["href"], "link": ["href"], "img": ["src", "data-src"],
             "source": ["src"], "video": ["src", "poster"], "audio": ["src"], "track": ["src"],
             "embed": ["src"], "iframe": ["src"], "object": ["data"], "input": ["src"]}
DROP_REL = {"preload", "prefetch", "preconnect", "dns-prefetch", "modulepreload", "prerender"}


def rw_html(raw, charset, page_url, nonce):
    soup = BeautifulSoup(raw, "html.parser", from_encoding=charset)
    base = page_url
    b = soup.find("base", href=True)
    if b:
        base = urljoin(page_url, b["href"])
    for t in soup.find_all("base"):
        t.decompose()
    for t in soup.find_all("noscript"):
        t.unwrap()
    for t in soup.find_all("meta"):
        he = (t.get("http-equiv") or "").lower()
        if he in ("refresh", "content-security-policy"):
            t.decompose()
    for t in soup.find_all(True):
        for a in list(t.attrs):
            if a.startswith("on") or a in ("integrity", "crossorigin", "nonce", "target"):
                del t.attrs[a]
        name = t.name
        if name == "link":
            rel = t.get("rel") or []
            if isinstance(rel, str):
                rel = rel.split()
            if DROP_REL & {x.lower() for x in rel}:
                t.decompose()
                continue
        for a in URL_ATTRS.get(name, []):
            if t.get(a):
                t[a] = rw(t[a], base)
        if name == "img" and not t.get("src") and t.get("data-src"):
            t["src"] = t["data-src"]
        for a in ("srcset", "data-srcset"):
            if t.get(a):
                t[a] = rw_srcset(t[a], base)
        if t.get("style"):
            t["style"] = rw_css(t["style"], base)
    for t in soup.find_all("style"):
        if t.string:
            t.string = rw_css(t.string, base)
    for f in soup.find_all("form"):
        action = urljoin(base, f.get("action") or page_url)
        method = "post" if (f.get("method") or "get").lower() == "post" else "get"
        f.attrs.pop("enctype", None)
        f["action"] = "/proxy"
        f["method"] = method
        f.insert(0, soup.new_tag("input", attrs={"type": "hidden", "name": "__u", "value": action}))
    head = soup.find("head")
    meta = soup.new_tag("meta", attrs={"charset": "utf-8"})
    if head:
        head.insert(0, meta)
    else:
        soup.insert(0, meta)
    out = soup.decode()
    return out + reporter(page_url, nonce)


def error_page(msg, status, url=""):
    nonce = secrets.token_urlsafe(12)
    html = ("<!doctype html><meta charset=\"utf-8\"><body style=\"font-family:system-ui,sans-serif;"
            "color:#556;padding:28px;line-height:1.5\"><h3>Couldn't load that page</h3><p>%s</p>%s</body>"
            % (escape(msg), reporter(url, nonce) if url else ""))
    return Response(html, status=status, content_type="text/html; charset=utf-8",
                    headers={"Content-Security-Policy": "sandbox allow-scripts; script-src 'nonce-%s'" % nonce})


@app.route("/proxy", methods=["GET", "POST"])
def proxy():
    if limited("proxy", 300, 60):
        return error_page("Too many requests. Wait a moment and try again.", 429)
    method, data = "GET", None
    src = request.args if "__u" in request.args else (request.form if "__u" in request.form else None)
    if src is not None:  # a form submitted from a proxied page
        target = src["__u"]
        fields = [(k, v) for k, v in src.items(multi=True) if k != "__u"]
        if request.method == "POST":
            method, data = "POST", fields
        else:
            target = urlparse(target)._replace(query=urlencode(fields)).geturl()
    else:
        target = request.args.get("url", "")
    target = unwrap(target.strip())
    if target and "://" not in target:
        target = "https://" + target
    if not target:
        return error_page("No address given.", 400)
    try:
        r, final = fetch(method, target, data)
    except ValueError as e:
        return error_page(str(e), 400, target)
    except requests.RequestException:
        return error_page("Couldn't reach that site. It may be down or blocking proxies.", 502, target)

    ct = r.headers.get("Content-Type", "application/octet-stream")
    mime = ct.split(";")[0].strip().lower()
    cs = re.search(r"charset=([\w-]+)", ct, re.I)
    charset = cs.group(1) if cs else None
    nonce = secrets.token_urlsafe(12)
    headers = {"Access-Control-Allow-Origin": "*",
               # Opaque origin + only our nonce'd script may run, so proxied pages can't touch the hub.
               "Content-Security-Policy": "sandbox allow-scripts allow-forms; script-src 'nonce-%s'" % nonce}
    status = r.status_code if 200 <= r.status_code < 600 else 200

    if mime in ("text/html", "application/xhtml+xml"):
        out = rw_html(read_limited(r, MAX_HTML), charset, final, nonce)
        headers["Cache-Control"] = "no-store"
        return Response(out, status=status, content_type="text/html; charset=utf-8", headers=headers)
    if mime == "text/css":
        css = read_limited(r, MAX_HTML).decode(charset or "utf-8", errors="replace")
        headers["Cache-Control"] = "public, max-age=300"
        return Response(rw_css(css, final), status=status, content_type="text/css; charset=utf-8", headers=headers)

    headers["Cache-Control"] = "public, max-age=300"

    def gen():
        size = 0
        try:
            for chunk in r.iter_content(65536):
                size += len(chunk)
                if size > MAX_ASSET:
                    break
                yield chunk
        finally:
            r.close()

    return Response(stream_with_context(gen()), status=status, content_type=ct, headers=headers)


if __name__ == "__main__":
    if not GROQ_KEY:
        print("! GROQ_API_KEY isn't set, so Rel AI won't work. Put it in .env")
    else:
        print("Rel Zone v%s | Groq key loaded (%s...%s) | check http://127.0.0.1:%s/api/health" % (VERSION, GROQ_KEY[:4], GROQ_KEY[-4:], os.environ.get("PORT", "5000")))
    app.run(host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", "5000")), threaded=True)
