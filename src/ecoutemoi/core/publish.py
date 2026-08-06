"""Sorties texte externes : OBS Studio (obs-websocket) et page web locale.

Deux chemins indépendants, alimentés par le MÊME texte que la fenêtre de sortie :

- `ObsTextPublisher` — client **obs-websocket 5.x** : pousse le texte dans une
  source « Texte (GDI+ / FreeType 2) » via `SetInputSettings`. C'est la voie
  officielle et elle évite complètement la capture de fenêtre + chroma key : OBS
  compose le texte lui-même, donc net, redimensionnable, sans liseré.
- `WebSubtitleServer` — petit serveur HTTP servant une page de sous-titres et un
  flux **SSE**. Ouvrable dans un navigateur (second écran, régie, retour
  orateur) ou en « Source navigateur » OBS, y compris depuis une autre machine.

Aucune dépendance ajoutée. Le client WebSocket est écrit ici (≈120 lignes de
stdlib) : l'app n'a qu'UN usage WebSocket, côté client, en texte, sur la boucle
locale — une bibliothèque complète (asyncio, TLS, permessage-deflate) coûterait
plus en surface de bundle PyInstaller qu'elle n'apporterait. Le sens
serveur→navigateur passe en SSE, qui se reconnecte tout seul et tient en
quelques lignes de `http.server`.
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import json
import logging
import os
import re
import socket
import struct
import sys
import threading
import time
from dataclasses import dataclass

log = logging.getLogger(__name__)

_WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"  # RFC 6455
_MAX_FRAME = 8 << 20  # 8 Mio : au-delà, c'est un pair qui déraille
OBS_MIN_INTERVAL_S = 0.10  # 10 mises à jour/s max : inutile de marteler OBS
OBS_RECONNECT_MAX_S = 15.0
_HEX_RE = re.compile(r"^#[0-9A-Fa-f]{6}$")


# ----------------------------------------------------------------- WebSocket
class _WebSocket:
    """Client WebSocket RFC 6455 minimal, texte, socket bloquant."""

    def __init__(self, host: str, port: int, timeout: float = 5.0, path: str = "/"):
        self._sock = socket.create_connection((host, port), timeout=timeout)
        self._sock.settimeout(timeout)
        self._buf = b""
        try:
            self._handshake(host, port, path)
        except BaseException:
            self.close()
            raise

    # -- établissement
    def _handshake(self, host: str, port: int, path: str) -> None:
        key = base64.b64encode(os.urandom(16)).decode()
        self._sock.sendall(
            (
                f"GET {path} HTTP/1.1\r\n"
                f"Host: {host}:{port}\r\n"
                "Upgrade: websocket\r\n"
                "Connection: Upgrade\r\n"
                f"Sec-WebSocket-Key: {key}\r\n"
                "Sec-WebSocket-Version: 13\r\n"
                "\r\n"
            ).encode()
        )
        head = self._read_until(b"\r\n\r\n").decode("latin-1")
        status = head.split("\r\n", 1)[0]
        if " 101" not in status:
            raise ConnectionError(f"passage en WebSocket refusé ({status.strip()})")
        expected = base64.b64encode(hashlib.sha1((key + _WS_GUID).encode()).digest()).decode()
        for line in head.split("\r\n"):
            name, _, value = line.partition(":")
            if name.strip().lower() == "sec-websocket-accept":
                if value.strip() != expected:
                    raise ConnectionError("Sec-WebSocket-Accept invalide (ce n'est pas un serveur WebSocket)")
                return
        raise ConnectionError("réponse sans Sec-WebSocket-Accept")

    # -- lecture bas niveau
    def _read_until(self, delimiter: bytes) -> bytes:
        while delimiter not in self._buf:
            chunk = self._sock.recv(4096)
            if not chunk:
                raise ConnectionError("connexion fermée pendant la négociation")
            self._buf += chunk
        head, _, self._buf = self._buf.partition(delimiter)
        return head + delimiter

    def _recv_exact(self, n: int) -> bytes:
        while len(self._buf) < n:
            chunk = self._sock.recv(max(4096, n - len(self._buf)))
            if not chunk:
                raise ConnectionError("connexion fermée par le pair")
            self._buf += chunk
        out, self._buf = self._buf[:n], self._buf[n:]
        return out

    # -- trames
    @staticmethod
    def _apply_mask(payload: bytes, mask: bytes) -> bytes:
        return bytes(b ^ mask[i % 4] for i, b in enumerate(payload))

    def _send_frame(self, opcode: int, payload: bytes) -> None:
        header = bytearray([0x80 | opcode])  # FIN + opcode
        n = len(payload)
        if n < 126:
            header.append(0x80 | n)  # les trames client sont TOUJOURS masquées
        elif n < 65536:
            header.append(0x80 | 126)
            header += struct.pack(">H", n)
        else:
            header.append(0x80 | 127)
            header += struct.pack(">Q", n)
        mask = os.urandom(4)
        header += mask
        self._sock.sendall(bytes(header) + self._apply_mask(payload, mask))

    def _recv_frame(self) -> tuple[bool, int, bytes]:
        b0, b1 = self._recv_exact(2)
        fin, opcode = bool(b0 & 0x80), b0 & 0x0F
        masked, n = bool(b1 & 0x80), b1 & 0x7F
        if n == 126:
            (n,) = struct.unpack(">H", self._recv_exact(2))
        elif n == 127:
            (n,) = struct.unpack(">Q", self._recv_exact(8))
        if n > _MAX_FRAME:
            raise ConnectionError(f"trame de {n} octets refusée (> {_MAX_FRAME})")
        mask = self._recv_exact(4) if masked else b""
        payload = self._recv_exact(n) if n else b""
        return fin, opcode, self._apply_mask(payload, mask) if mask else payload

    # -- API
    def send_text(self, text: str) -> None:
        self._send_frame(0x1, text.encode())

    def recv_text(self) -> str:
        """Prochain message applicatif ; ping/pong traités au passage."""
        chunks: list[bytes] = []
        while True:
            fin, opcode, payload = self._recv_frame()
            if opcode == 0x8:
                raise ConnectionError("fermeture demandée par le serveur")
            if opcode == 0x9:  # ping -> pong
                self._send_frame(0xA, payload)
                continue
            if opcode == 0xA:  # pong non sollicité
                continue
            chunks.append(payload)
            if fin:
                return b"".join(chunks).decode("utf-8", errors="replace")

    def close(self) -> None:
        with contextlib.suppress(OSError, ConnectionError):
            self._send_frame(0x8, struct.pack(">H", 1001))  # going away
        with contextlib.suppress(OSError):
            self._sock.close()


# ------------------------------------------------------------------ OBS
class ObsError(RuntimeError):
    """Requête refusée par OBS (source absente, mauvais type, droits…)."""


def obs_auth(password: str, salt: str, challenge: str) -> str:
    """Réponse au défi obs-websocket 5.x (deux SHA-256 encadrés de base64)."""
    secret = base64.b64encode(hashlib.sha256((password + salt).encode()).digest()).decode()
    return base64.b64encode(hashlib.sha256((secret + challenge).encode()).digest()).decode()


class ObsClient:
    """Session obs-websocket 5.x : Hello → Identify → Identified, puis requêtes."""

    def __init__(self, host: str, port: int, password: str = "", timeout: float = 5.0):
        self._ws = _WebSocket(host, port, timeout=timeout)
        self._req_id = 0
        try:
            self._identify(password)
        except BaseException:
            self._ws.close()
            raise

    def _recv_op(self, op: int, tries: int = 8) -> dict:
        for _ in range(tries):
            msg = json.loads(self._ws.recv_text())
            if msg.get("op") == op:
                return msg.get("d") or {}
        raise ConnectionError(f"aucun message op={op} reçu d'OBS")

    def _identify(self, password: str) -> None:
        hello = self._recv_op(0)
        payload: dict = {"rpcVersion": 1, "eventSubscriptions": 0}  # aucun événement voulu
        auth = hello.get("authentication")
        if auth:
            if not password:
                raise ConnectionError(
                    "OBS demande un mot de passe (Outils → Paramètres du serveur WebSocket)"
                )
            payload["authentication"] = obs_auth(password, auth["salt"], auth["challenge"])
        self._ws.send_text(json.dumps({"op": 1, "d": payload}))
        self._recv_op(2)

    def request(self, request_type: str, request_data: dict | None = None) -> dict:
        self._req_id += 1
        rid = str(self._req_id)
        self._ws.send_text(
            json.dumps(
                {
                    "op": 6,
                    "d": {
                        "requestType": request_type,
                        "requestId": rid,
                        "requestData": request_data or {},
                    },
                }
            )
        )
        while True:
            msg = json.loads(self._ws.recv_text())
            if msg.get("op") != 7:
                continue
            data = msg.get("d") or {}
            if data.get("requestId") != rid:
                continue
            status = data.get("requestStatus") or {}
            if not status.get("result"):
                raise ObsError(
                    f"{request_type} refusé par OBS : "
                    f"{status.get('comment') or status.get('code') or 'raison inconnue'}"
                )
            return data.get("responseData") or {}

    # -- requêtes utilisées
    def version(self) -> str:
        d = self.request("GetVersion")
        return f"OBS {d.get('obsVersion', '?')} · obs-websocket {d.get('obsWebSocketVersion', '?')}"

    def text_inputs(self) -> list[str]:
        """Noms des sources de type texte (GDI+ / FreeType 2)."""
        inputs = self.request("GetInputList").get("inputs") or []
        return [str(i.get("inputName")) for i in inputs if "text" in str(i.get("inputKind", "")).lower()]

    def set_text(self, source: str, text: str) -> None:
        self.request(
            "SetInputSettings",
            {"inputName": source, "inputSettings": {"text": text}, "overlay": True},
        )

    def close(self) -> None:
        self._ws.close()


def obs_probe(host: str, port: int, password: str = "", timeout: float = 5.0) -> tuple[str, list[str]]:
    """Test de connexion pour la GUI : (version, sources texte). Lève en cas d'échec."""
    client = ObsClient(host, port, password, timeout=timeout)
    try:
        return client.version(), client.text_inputs()
    finally:
        client.close()


class ObsTextPublisher:
    """Pousse le dernier texte connu vers une source Texte d'OBS, en tâche de fond.

    Tolérant par construction : OBS peut être lancé après, fermé pendant, ou
    redémarré — le fil se reconnecte avec un délai croissant sans jamais bloquer
    l'appelant. `set_text()` ne fait que déposer la dernière valeur (les
    intermédiaires sont écrasées : seul l'état courant compte).
    """

    def __init__(self, host: str, port: int, password: str, source: str, on_status=None):
        self.host, self.port, self.password, self.source = host, port, password, source
        self._on_status = on_status
        self.status = "OBS : non connecté"
        self._pending = ""
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="obs-publish")

    def start(self) -> None:
        self._thread.start()

    def set_text(self, text: str) -> None:
        with self._lock:
            if text == self._pending:
                return
            self._pending = text
        self._wake.set()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread.is_alive():
            self._thread.join(3.0)

    def _set_status(self, status: str) -> None:
        if status != self.status:
            self.status = status
            log.info("%s", status)
            if self._on_status is not None:
                with contextlib.suppress(Exception):
                    self._on_status(status)

    def _run(self) -> None:
        delay = 1.0
        while not self._stop.is_set():
            try:
                client = ObsClient(self.host, self.port, self.password)
            except Exception as exc:
                self._set_status(f"OBS injoignable : {exc}")
                if self._stop.wait(delay):
                    return
                delay = min(delay * 2.0, OBS_RECONNECT_MAX_S)
                continue
            delay = 1.0
            try:
                self._set_status(f"OBS connecté ({client.version()}) → « {self.source} »")
                self._pump(client)
            except Exception as exc:
                self._set_status(f"OBS déconnecté : {exc}")
            finally:
                with contextlib.suppress(Exception):
                    client.close()

    def _pump(self, client: ObsClient) -> None:
        sent: str | None = None
        while not self._stop.is_set():
            self._wake.wait(timeout=1.0)
            self._wake.clear()
            with self._lock:
                text = self._pending
            if text == sent:
                continue
            client.set_text(self.source, text)
            sent = text
            time.sleep(OBS_MIN_INTERVAL_S)


# ------------------------------------------------------------- page web
@dataclass(frozen=True)
class WebStyle:
    """Apparence de la page web, alignée sur celle de la fenêtre de sortie."""

    font_family: str = ""
    font_size: int = 34
    text_color: str = "#FFFFFF"
    outline_color: str = "#000000"
    bg_color: str = "#00FF00"
    outline_width: int = 3
    align: str = "center"
    margin_h: int = 24
    margin_v: int = 12
    transparent: bool = False
    max_chars_per_line: int = 42  # 0 => pleine largeur
    dual: bool = False  # deux lignes de sous-titres empilées

    @classmethod
    def from_settings(cls, s) -> WebStyle:
        return cls(
            font_family=s.font_family,
            font_size=s.font_size,
            text_color=s.text_color,
            outline_color=s.outline_color,
            bg_color=s.bg_color,
            outline_width=max(0, s.outline_width),
            align=s.align,
            margin_h=s.margin_h,
            margin_v=s.margin_v,
            transparent=s.overlay_transparent,
            max_chars_per_line=max(0, s.max_chars_per_line),
            dual=bool(s.dual_enabled),
        )


def local_ip_addresses() -> list[str]:
    """Adresses IP de la machine, hors boucle locale, pour l'accès LAN.

    Utilise psutil (déjà une dépendance) plutôt qu'une résolution du nom d'hôte :
    sur un poste avec plusieurs interfaces (Wi-Fi + Ethernet + docker0), la
    résolution DNS ne rend qu'une adresse, souvent la mauvaise. L'opérateur a
    besoin de voir TOUTES les adresses joignables pour choisir la bonne.
    """
    import psutil

    found: list[str] = []
    try:
        interfaces = psutil.net_if_addrs()
        stats = psutil.net_if_stats()
    except Exception as exc:  # plateforme exotique : pas de liste, pas de drame
        log.debug("Énumération des interfaces impossible : %s", exc)
        return found
    for name, addresses in interfaces.items():
        info = stats.get(name)
        if info is not None and not info.isup:
            continue
        for addr in addresses:
            if addr.family != socket.AF_INET:
                continue  # IPv4 seulement : c'est ce qu'on tape dans un navigateur
            ip = addr.address
            if ip.startswith("127.") or ip in found:
                continue
            found.append(ip)
    return found


def _css_color(value: str, fallback: str) -> str:
    return value if _HEX_RE.match(value or "") else fallback


def render_page(style: WebStyle, transparent: bool | None = None) -> str:
    """Page de sous-titres autonome (aucune ressource externe).

    Le contour passe par `-webkit-text-stroke` + `paint-order: stroke fill` :
    contrairement à un `text-shadow` multiple, le trait ne mange pas le glyphe et
    reste net à toute taille. Le moteur des « Sources navigateur » OBS est du
    Chromium, la propriété y est donc acquise.
    """
    see_through = style.transparent if transparent is None else transparent
    background = "transparent" if see_through else _css_color(style.bg_color, "#00FF00")
    family = json.dumps(style.font_family) + "," if style.font_family else ""
    align = style.align if style.align in ("left", "center", "right") else "center"
    flex_align = {"left": "flex-start", "center": "center", "right": "flex-end"}[align]
    # `ch` = largeur du chiffre « 0 » : l'unité CSS qui approche le mieux la norme
    # « 42 caractères par ligne », et la seule qui suive la police choisie.
    max_width_rule = f"max-width: {style.max_chars_per_line}ch;" if style.max_chars_per_line else ""
    return f"""<!doctype html>
<html lang="fr"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Écoute Moi — sous-titres</title>
<style>
  html, body {{ margin: 0; height: 100%; background: {background}; overflow: hidden; }}
  #stack {{
    position: absolute; left: {style.margin_h}px; right: {style.margin_h}px;
    bottom: {style.margin_v}px;
    display: flex; flex-direction: column; align-items: {flex_align}; gap: .25em;
  }}
  .line {{
    text-align: {align};
    font-family: {family} system-ui, "Segoe UI", Roboto, sans-serif;
    font-size: {max(8, style.font_size)}px; font-weight: 600; line-height: 1.25;
    color: {_css_color(style.text_color, "#FFFFFF")};
    -webkit-text-stroke: {style.outline_width}px {_css_color(style.outline_color, "#000000")};
    paint-order: stroke fill;
    white-space: pre-wrap; overflow-wrap: anywhere;
    {max_width_rule}
  }}
  .line:empty {{ display: none; }}
  #secondary {{ opacity: .92; }}
  #offline {{
    position: absolute; top: 8px; right: 10px; font: 12px system-ui;
    color: #ff6b6b; background: rgba(0,0,0,.55); padding: 2px 6px; border-radius: 3px;
    display: none;
  }}
</style></head><body>
<div id="stack">
  <div class="line" id="primary"></div>
  <div class="line" id="secondary"></div>
</div>
<div id="offline">hors ligne</div>
<script>
  const primary = document.getElementById('primary');
  const secondary = document.getElementById('secondary');
  const offline = document.getElementById('offline');
  let source = null;
  function connect() {{
    source = new EventSource('/events');
    source.onopen = () => {{ offline.style.display = 'none'; }};
    source.onmessage = (event) => {{
      const data = JSON.parse(event.data);
      // Rétro-compatible : une chaîne seule reste le sous-titre principal.
      if (typeof data === 'string') {{
        primary.textContent = data;
        secondary.textContent = '';
      }} else {{
        primary.textContent = data.primary || '';
        secondary.textContent = data.secondary || '';
      }}
    }};
    source.onerror = () => {{
      offline.style.display = 'block';
      source.close();
      setTimeout(connect, 1000);  // le serveur redémarre entre deux sessions
    }};
  }}
  connect();
</script></body></html>
"""


class WebSubtitleServer:
    """Serveur HTTP local : `/` la page, `/events` le flux SSE, `/text` le brut."""

    def __init__(self, port: int, style: WebStyle, bind_lan: bool = False):
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        self.port = port
        self.style = style
        self.bind_lan = bool(bind_lan)
        self.host = "0.0.0.0" if bind_lan else "127.0.0.1"
        self._text = ""
        self._secondary = ""
        self._version = 0
        self._cond = threading.Condition()
        self._stopping = False
        outer = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"
            server_version = "EcouteMoi"

            def log_message(self, fmt: str, *args) -> None:  # pas de bruit sur stderr
                log.debug("web: " + fmt, *args)

            def do_GET(self) -> None:
                path = self.path.split("?", 1)[0]
                if path in ("/", "/index.html"):
                    query = self.path.split("?", 1)[1] if "?" in self.path else ""
                    transparent = True if "bg=transparent" in query else None
                    outer._send_bytes(self, 200, "text/html; charset=utf-8",
                                      render_page(outer.style, transparent).encode())  # fmt: skip
                elif path == "/text":
                    outer._send_bytes(self, 200, "text/plain; charset=utf-8", outer.text().encode())
                elif path == "/events":
                    outer._serve_events(self)
                else:
                    outer._send_bytes(self, 404, "text/plain; charset=utf-8", b"introuvable\n")

        class Server(ThreadingHTTPServer):
            daemon_threads = True  # un flux SSE bloqué ne retient pas la sortie
            # Windows : SO_REUSEADDR y a une sémantique DIFFÉRENTE d'Unix — il
            # autorise deux serveurs à se lier au même port, et le trafic part
            # vers l'un ou l'autre. Un port déjà occupé passait donc inaperçu
            # (« démarré » sans rien servir) au lieu d'être signalé à l'opérateur.
            allow_reuse_address = sys.platform != "win32"

        self._server = Server((self.host, port), Handler)
        self.port = self._server.server_address[1]  # port 0 => celui choisi par l'OS
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True, name="web-subtitles")

    # -- état partagé
    def text(self) -> str:
        """Texte brut servi sur `/text` : les deux sous-titres, un par ligne."""
        with self._cond:
            return "\n".join(part for part in (self._text, self._secondary) if part)

    def set_text(self, text: str, secondary: str = "") -> None:
        with self._cond:
            if (text, secondary) == (self._text, self._secondary):
                return
            self._text, self._secondary = text, secondary
            self._version += 1
            self._cond.notify_all()

    def _wait_change(self, seen: int, timeout: float) -> tuple[dict, int]:
        with self._cond:
            if self._version == seen and not self._stopping:
                self._cond.wait(timeout)
            return {"primary": self._text, "secondary": self._secondary}, self._version

    # -- HTTP
    @staticmethod
    def _send_bytes(handler, code: int, content_type: str, body: bytes) -> None:
        handler.send_response(code)
        handler.send_header("Content-Type", content_type)
        handler.send_header("Content-Length", str(len(body)))
        handler.send_header("Cache-Control", "no-store")
        handler.end_headers()
        with contextlib.suppress(OSError):
            handler.wfile.write(body)

    def _serve_events(self, handler) -> None:
        handler.send_response(200)
        handler.send_header("Content-Type", "text/event-stream; charset=utf-8")
        handler.send_header("Cache-Control", "no-store")
        handler.send_header("Connection", "close")
        handler.end_headers()
        handler.close_connection = True  # flux sans longueur : pas de keep-alive
        seen = -1
        while not self._stopping:
            payload, version = self._wait_change(seen, timeout=10.0)
            if version == seen:
                chunk = ": ping\n\n"  # commentaire SSE : garde le flux ouvert
            else:
                seen = version
                # JSON plutôt que le texte brut : `data:` interdit les sauts de ligne.
                chunk = f"data: {json.dumps(payload)}\n\n"
            try:
                handler.wfile.write(chunk.encode())
                handler.wfile.flush()
            except OSError:
                return  # onglet fermé

    # -- cycle de vie
    def start(self) -> None:
        self._thread.start()
        log.info("Page de sous-titres : %s", self.url())

    def url(self) -> str:
        shown = "127.0.0.1" if self.host in ("0.0.0.0", "::") else self.host
        return f"http://{shown}:{self.port}/"

    def urls(self) -> list[str]:
        """Toutes les URL joignables : la boucle locale, puis le LAN si ouvert."""
        addresses = [self.url()]
        if self.bind_lan:
            addresses += [f"http://{ip}:{self.port}/" for ip in local_ip_addresses()]
        return addresses

    def stop(self) -> None:
        self._stopping = True
        with self._cond:
            self._cond.notify_all()  # réveille les flux SSE pour qu'ils sortent
        # `shutdown()` ATTEND la sortie de `serve_forever()` : appelé sur un
        # serveur jamais démarré, il bloque pour toujours. Ça arrive vraiment —
        # socket liée puis `start()` qui échoue, ou arrêt immédiat après create.
        if self._thread.is_alive():
            with contextlib.suppress(Exception):
                self._server.shutdown()
            self._thread.join(timeout=5.0)
        with contextlib.suppress(Exception):
            self._server.server_close()


# --------------------------------------------------------------- façade
class TextPublishers:
    """Sorties externes vues comme une seule cible. `apply()` est idempotent.

    L'appelant (la GUI) pousse le texte à chaque mise à jour sans se demander qui
    est actif : ce qui est désactivé ne fait rien.
    """

    def __init__(self, on_status=None):
        self._on_status = on_status
        self.obs: ObsTextPublisher | None = None
        self.obs_dual: ObsTextPublisher | None = None
        self.web: WebSubtitleServer | None = None
        self._obs_key: tuple | None = None
        self._web_key: tuple | None = None
        self.web_error: str | None = None
        self._text = ""
        self._secondary = ""

    # -- configuration
    def apply(self, settings) -> None:
        """(Re)démarre OBS et/ou la page web selon `settings`. Sans effet si rien n'a changé."""
        dual = bool(settings.dual_enabled)
        obs_key = (
            bool(settings.obs_ws_enabled),
            settings.obs_ws_host,
            int(settings.obs_ws_port),
            settings.obs_ws_password,
            settings.obs_ws_source,
            settings.obs_ws_source_dual if dual else None,
        )
        if obs_key != self._obs_key:
            self._obs_key = obs_key
            for publisher in (self.obs, self.obs_dual):
                if publisher is not None:
                    publisher.stop()
            self.obs = self.obs_dual = None
            if settings.obs_ws_enabled:
                self.obs = ObsTextPublisher(
                    settings.obs_ws_host,
                    int(settings.obs_ws_port),
                    settings.obs_ws_password,
                    settings.obs_ws_source,
                    on_status=self._on_status,
                )
                self.obs.start()
                self.obs.set_text(self._text)
                if dual and settings.obs_ws_source_dual.strip():
                    # Deuxième connexion plutôt qu'une seule multiplexée : chaque
                    # source a son propre fil, donc une source absente dans OBS ne
                    # bloque pas l'autre sous-titre.
                    self.obs_dual = ObsTextPublisher(
                        settings.obs_ws_host,
                        int(settings.obs_ws_port),
                        settings.obs_ws_password,
                        settings.obs_ws_source_dual,
                        on_status=self._on_status,
                    )
                    self.obs_dual.start()
                    self.obs_dual.set_text(self._secondary)

        # L'apparence ne fait PAS partie de la clé : elle est relue à chaque
        # requête, donc un changement de police ou de couleur ne coûte qu'un
        # rechargement de la page au lieu d'un redémarrage du serveur (et
        # `apply()` est appelé à chaque cran de molette sur la taille de police).
        style = WebStyle.from_settings(settings)
        web_key = (bool(settings.web_enabled), int(settings.web_port), bool(settings.web_bind_lan))
        if web_key != self._web_key:
            self._web_key = web_key
            if self.web is not None:
                self.web.stop()
                self.web = None
            self.web_error = None
            if settings.web_enabled:
                try:
                    self.web = WebSubtitleServer(
                        int(settings.web_port), style, bind_lan=bool(settings.web_bind_lan)
                    )
                    self.web.start()
                    self.web.set_text(self._text, self._secondary)
                except OSError as exc:
                    self.web_error = f"port {settings.web_port} indisponible : {exc}"
                    log.warning("Page web non démarrée — %s", self.web_error)
                    if self._on_status is not None:
                        with contextlib.suppress(Exception):
                            self._on_status(f"Page web : {self.web_error}")
        if self.web is not None:
            self.web.style = style  # à chaud : visible au prochain chargement

    # -- diffusion
    def set_text(self, text: str, secondary: str = "") -> None:
        self._text, self._secondary = text, secondary
        if self.obs is not None:
            self.obs.set_text(text)
        if self.obs_dual is not None:
            self.obs_dual.set_text(secondary)
        if self.web is not None:
            self.web.set_text(text, secondary)

    def web_url(self) -> str | None:
        return self.web.url() if self.web is not None else None

    def web_urls(self) -> list[str]:
        return self.web.urls() if self.web is not None else []

    def active(self) -> list[str]:
        names = []
        if self.obs is not None:
            names.append("OBS" + (" ×2" if self.obs_dual is not None else ""))
        if self.web is not None:
            names.append("web")
        return names

    def stop(self) -> None:
        for publisher in (self.obs, self.obs_dual):
            if publisher is not None:
                publisher.stop()
        self.obs = self.obs_dual = None
        if self.web is not None:
            self.web.stop()
            self.web = None
        self._obs_key = self._web_key = None


__all__ = [
    "ObsClient",
    "ObsError",
    "ObsTextPublisher",
    "TextPublishers",
    "WebStyle",
    "WebSubtitleServer",
    "obs_auth",
    "obs_probe",
    "render_page",
]
