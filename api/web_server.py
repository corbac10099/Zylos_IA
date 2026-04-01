"""
api/web_server.py — Serveur web HTTP pour ZYLOS AI
====================================================
Interface web locale permettant d'utiliser ZYLOS AI via navigateur.
Tourne sur http://localhost:8080

Usage :
    python api/web_server.py
    # ou depuis main.py avec --web
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils.logger import get_logger
log = get_logger(__name__)

# HTML de l'interface - injecté directement
_HTML_UI = None  # chargé depuis web_ui.html si présent, sinon embarqué


class ZylosHandler(BaseHTTPRequestHandler):
    """Gestionnaire de requêtes HTTP pour ZYLOS AI."""

    def log_message(self, format, *args):
        # Silencieux sauf erreurs
        if args and len(args) >= 2 and str(args[1]).startswith(('4', '5')):
            log.warning("HTTP %s %s %s", *args)

    def do_GET(self):
        parsed = urlparse(self.path)
        path   = parsed.path

        if path == "/" or path == "/index.html":
            self._serve_ui()
        elif path == "/api/status":
            self._api_status()
        elif path == "/api/metrics":
            self._api_metrics()
        elif path == "/api/history":
            self._api_history()
        else:
            self._send_json({"error": "Not found"}, 404)

    def do_POST(self):
        parsed = urlparse(self.path)
        path   = parsed.path

        length  = int(self.headers.get("Content-Length", 0))
        body    = self.rfile.read(length) if length else b""

        try:
            data = json.loads(body) if body else {}
        except json.JSONDecodeError:
            data = {}

        if path == "/api/chat":
            self._api_chat(data)
        elif path == "/api/reset":
            self._api_reset()
        else:
            self._send_json({"error": "Not found"}, 404)

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors_headers()
        self.end_headers()

    def _cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _send_json(self, data: Any, status: int = 200):
        body = json.dumps(data, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self._cors_headers()
        self.end_headers()
        self.wfile.write(body)

    def _serve_ui(self):
        html = _get_html_ui()
        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self._cors_headers()
        self.end_headers()
        self.wfile.write(body)

    def _api_status(self):
        try:
            from core.model import model
            from core.backend import backend_info
            from config import RWKV as RWKV_CFG, MISTRAL
            self._send_json({
                "ready":       model.is_ready,
                "model_size":  RWKV_CFG.default_size,
                "backend":     backend_info.name.upper(),
                "device":      backend_info.device_name,
                "vram_mb":     backend_info.vram_mb,
                "quantization": backend_info.quantization_level(),
                "mistral":     MISTRAL.is_configured(),
                "version":     "2.0",
            })
        except Exception as e:
            self._send_json({"error": str(e), "ready": False})

    def _api_metrics(self):
        try:
            from utils.metrics import metrics
            snap = metrics.snapshot()
            self._send_json(snap)
        except Exception as e:
            self._send_json({"error": str(e)})

    def _api_history(self):
        try:
            from modules.brain import brain
            self._send_json({"history": brain.get_history()})
        except Exception as e:
            self._send_json({"error": str(e), "history": []})

    def _api_chat(self, data: dict):
        message = data.get("message", "").strip()
        if not message:
            self._send_json({"error": "Message vide"}, 400)
            return

        try:
            from modules.brain import brain
            t0 = time.perf_counter()
            resp = brain.chat(message, use_rag=data.get("use_rag", True))
            elapsed = time.perf_counter() - t0
            self._send_json({
                "text":       resp.text,
                "latency_ms": round(elapsed * 1000, 1),
                "tokens_out": resp.tokens_out,
                "rag_hit":    resp.rag_hit,
                "error":      resp.error,
                "reasoning":  resp.reasoning,
            })
        except Exception as e:
            log.error("Erreur API chat : %s", e, exc_info=True)
            self._send_json({"error": str(e), "text": ""})

    def _api_reset(self):
        try:
            from modules.brain import brain
            brain.reset_history()
            self._send_json({"ok": True})
        except Exception as e:
            self._send_json({"error": str(e)})


def _get_html_ui() -> str:
    """Charge l'UI HTML depuis fichier ou retourne l'UI embarquée."""
    ui_path = _ROOT / "api" / "web_ui.html"
    if ui_path.exists():
        return ui_path.read_text(encoding="utf-8")
    return _EMBEDDED_HTML


def run_server(host: str = "0.0.0.0", port: int = 8080) -> None:
    """Lance le serveur HTTP ZYLOS."""
    server = HTTPServer((host, port), ZylosHandler)
    log.info("🌐 Interface web ZYLOS disponible sur http://localhost:%d", port)
    print(f"\n🌐  Interface web : http://localhost:{port}\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()


# ══════════════════════════════════════════════════════════════════════
# UI HTML EMBARQUÉE (fallback si web_ui.html absent)
# ══════════════════════════════════════════════════════════════════════
_EMBEDDED_HTML = """<!DOCTYPE html>
<html lang="fr">
<head><meta charset="UTF-8"><title>ZYLOS AI</title>
<style>body{font-family:monospace;background:#0a0a0a;color:#e0e0e0;margin:0;padding:20px;}
.chat-box{max-width:800px;margin:0 auto;}
h1{color:#00ff88;font-size:1.5em;}
#messages{height:60vh;overflow-y:auto;border:1px solid #333;padding:10px;margin:10px 0;}
.msg-user{color:#88ccff;text-align:right;margin:5px 0;}
.msg-bot{color:#e0e0e0;margin:5px 0;}
input{width:80%;background:#1a1a1a;color:#e0e0e0;border:1px solid #444;padding:8px;}
button{background:#00ff88;color:#000;border:none;padding:8px 16px;cursor:pointer;}
</style></head>
<body>
<div class="chat-box">
<h1>⚡ ZYLOS AI</h1>
<div id="status">Vérification...</div>
<div id="messages"></div>
<input id="inp" placeholder="Votre message..." onkeydown="if(event.key==='Enter')send()">
<button onclick="send()">Envoyer</button>
<button onclick="reset()" style="background:#ff4444;color:#fff;margin-left:5px">Reset</button>
</div>
<script>
async function checkStatus(){
  const r=await fetch('/api/status');
  const d=await r.json();
  document.getElementById('status').innerHTML=
    d.ready?'✅ Modèle prêt — '+d.backend+' '+d.model_size:'⏳ Chargement...';
  if(!d.ready) setTimeout(checkStatus,2000);
}
async function send(){
  const inp=document.getElementById('inp');
  const msg=inp.value.trim();
  if(!msg)return;
  inp.value='';
  addMsg(msg,'user');
  const r=await fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({message:msg})});
  const d=await r.json();
  addMsg(d.error||d.text,'bot');
}
async function reset(){
  await fetch('/api/reset',{method:'POST'});
  document.getElementById('messages').innerHTML='';
}
function addMsg(text,who){
  const div=document.createElement('div');
  div.className='msg-'+who;
  div.textContent=(who==='user'?'Vous : ':' Zylos : ')+text;
  document.getElementById('messages').appendChild(div);
  document.getElementById('messages').scrollTop=99999;
}
checkStatus();
</script>
</body></html>"""


if __name__ == "__main__":
    import os
    os.environ.setdefault("ZYLOS_LOG_LEVEL", "INFO")

    # Init basique
    from config import PATHS
    PATHS.create_all()
    from utils.metrics import metrics
    metrics.init()

    # Charger modèle en background
    def _load():
        from core.model import model
        model.load()

    t = threading.Thread(target=_load, daemon=True)
    t.start()

    run_server()