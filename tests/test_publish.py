"""Sorties externes : protocole obs-websocket, client WebSocket, page web/SSE.

Le serveur WebSocket factice ci-dessous parle assez de RFC 6455 pour valider la
négociation, le masquage et le dialogue Hello/Identify/Request d'obs-websocket 5
— sans OBS ni dépendance réseau.
"""

from __future__ import annotations

import base64
import hashlib
import json
import socket
import struct
import threading
import urllib.error
import urllib.request

import pytest

from ecoutemoi.config import Settings
from ecoutemoi.core.publish import (
    ObsClient,
    ObsError,
    TextPublishers,
    WebStyle,
    WebSubtitleServer,
    obs_auth,
    render_page,
)

_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


# ------------------------------------------------------------- serveur factice
class FakeObs(threading.Thread):
    """Serveur obs-websocket minimal : négocie, identifie, répond aux requêtes."""

    def __init__(self, password: str = "", inputs: list[dict] | None = None):
        super().__init__(daemon=True, name="fake-obs")
        self.password = password
        self.inputs = inputs if inputs is not None else []
        self.listener = socket.socket()
        self.listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.listener.bind(("127.0.0.1", 0))
        self.listener.listen(1)
        self.port = self.listener.getsockname()[1]
        self.received: list[dict] = []
        self.identified = threading.Event()
        self.error: str | None = None

    # -- trames
    @staticmethod
    def _mask(payload: bytes, mask: bytes) -> bytes:
        return bytes(b ^ mask[i % 4] for i, b in enumerate(payload))

    def _send(self, sock: socket.socket, obj: dict) -> None:
        payload = json.dumps(obj).encode()
        header = bytearray([0x81])  # FIN + texte ; le serveur ne masque jamais
        n = len(payload)
        if n < 126:
            header.append(n)
        else:
            header.append(126)
            header += struct.pack(">H", n)
        sock.sendall(bytes(header) + payload)

    def _recv(self, sock: socket.socket, buf: bytearray) -> dict | None:
        """Prochain message texte, ou None sur trame de fermeture."""

        def take(count: int) -> bytes:
            while len(buf) < count:
                chunk = sock.recv(4096)
                if not chunk:
                    raise ConnectionError("client parti")
                buf.extend(chunk)
            out = bytes(buf[:count])
            del buf[:count]
            return out

        b0, b1 = take(2)
        assert b1 & 0x80, "une trame client DOIT être masquée (RFC 6455)"
        n = b1 & 0x7F
        if n == 126:
            (n,) = struct.unpack(">H", take(2))
        mask = take(4)
        payload = self._mask(take(n), mask)
        opcode = b0 & 0x0F
        if opcode == 0x8:
            return None
        assert opcode == 0x1, f"texte attendu, opcode {opcode:#x}"
        return json.loads(payload)

    def run(self) -> None:
        try:
            sock, _ = self.listener.accept()
            with sock:
                buf = bytearray()
                self._handshake(sock, buf)
                hello = {"op": 0, "d": {"rpcVersion": 1}}
                if self.password:
                    hello["d"]["authentication"] = {"challenge": "chal", "salt": "sel"}
                self._send(sock, hello)
                identify = self._recv(sock, buf)
                assert identify is not None and identify["op"] == 1
                if self.password:
                    expected = obs_auth(self.password, "sel", "chal")
                    assert identify["d"]["authentication"] == expected, "mauvaise réponse au défi"
                self._send(sock, {"op": 2, "d": {"negotiatedRpcVersion": 1}})
                self.identified.set()
                while True:
                    msg = self._recv(sock, buf)
                    if msg is None:  # fermeture propre du client
                        return
                    self.received.append(msg)
                    self._send(sock, {"op": 7, "d": self._answer(msg["d"])})
        except (ConnectionError, OSError, AssertionError) as exc:
            self.error = str(exc)
        finally:
            self.listener.close()

    def _handshake(self, sock: socket.socket, buf: bytearray) -> None:
        while b"\r\n\r\n" not in buf:
            buf.extend(sock.recv(4096))
        head, _, rest = bytes(buf).partition(b"\r\n\r\n")
        buf.clear()
        buf.extend(rest)
        key = ""
        for line in head.decode().split("\r\n"):
            name, _, value = line.partition(":")
            if name.strip().lower() == "sec-websocket-key":
                key = value.strip()
        assert key, "Sec-WebSocket-Key absente"
        accept = base64.b64encode(hashlib.sha1((key + _GUID).encode()).digest()).decode()
        sock.sendall(
            (
                "HTTP/1.1 101 Switching Protocols\r\n"
                "Upgrade: websocket\r\nConnection: Upgrade\r\n"
                f"Sec-WebSocket-Accept: {accept}\r\n\r\n"
            ).encode()
        )

    def _answer(self, d: dict) -> dict:
        rtype, rid = d["requestType"], d["requestId"]
        base = {"requestType": rtype, "requestId": rid}
        if rtype == "GetVersion":
            return {**base, "requestStatus": {"result": True, "code": 100},
                    "responseData": {"obsVersion": "31.0.0", "obsWebSocketVersion": "5.5.0"}}  # fmt: skip
        if rtype == "GetInputList":
            return {**base, "requestStatus": {"result": True, "code": 100},
                    "responseData": {"inputs": self.inputs}}  # fmt: skip
        if rtype == "SetInputSettings":
            name = d["requestData"]["inputName"]
            if name == "absente":
                return {**base, "requestStatus": {"result": False, "code": 600,
                                                  "comment": "source inconnue"}}  # fmt: skip
            return {**base, "requestStatus": {"result": True, "code": 100}}
        return {**base, "requestStatus": {"result": False, "code": 204, "comment": "non géré"}}


# ------------------------------------------------------------------- OBS
def test_obs_auth_matches_reference_vector():
    """Vecteur du protocole obs-websocket 5 : base64(sha256(base64(sha256(pw+salt))+challenge))."""
    secret = base64.b64encode(hashlib.sha256(b"motdepasse" + b"sel").digest()).decode()
    expected = base64.b64encode(hashlib.sha256((secret + "chal").encode()).digest()).decode()
    assert obs_auth("motdepasse", "sel", "chal") == expected


def test_obs_client_identifies_and_sets_text():
    server = FakeObs(inputs=[{"inputName": "Sous-titres", "inputKind": "text_gdiplus_v3"},
                             {"inputName": "Micro", "inputKind": "wasapi_input_capture"}])  # fmt: skip
    server.start()
    client = ObsClient("127.0.0.1", server.port)
    try:
        assert "31.0.0" in client.version()
        assert client.text_inputs() == ["Sous-titres"]  # les non-texte sont écartés
        client.set_text("Sous-titres", "bonjour à tous")
    finally:
        client.close()
    sent = [m["d"] for m in server.received if m["d"]["requestType"] == "SetInputSettings"]
    assert sent[-1]["requestData"]["inputSettings"] == {"text": "bonjour à tous"}
    assert server.error is None


def test_obs_client_authenticates_when_challenged():
    server = FakeObs(password="motdepasse")
    server.start()
    client = ObsClient("127.0.0.1", server.port, "motdepasse")
    client.close()
    assert server.identified.wait(2.0)
    assert server.error is None


def test_obs_client_without_password_when_required():
    server = FakeObs(password="motdepasse")
    server.start()
    with pytest.raises(ConnectionError, match="mot de passe"):
        ObsClient("127.0.0.1", server.port)


def test_obs_error_on_missing_source():
    server = FakeObs()
    server.start()
    client = ObsClient("127.0.0.1", server.port)
    try:
        with pytest.raises(ObsError, match="source inconnue"):
            client.set_text("absente", "coucou")
    finally:
        client.close()


def test_obs_client_refuses_plain_http_server():
    """Un serveur qui ne parle pas WebSocket doit échouer clairement, pas pendre."""
    style = WebStyle()
    server = WebSubtitleServer(0, style)
    server.start()
    try:
        with pytest.raises(ConnectionError):
            ObsClient("127.0.0.1", server._server.server_address[1], timeout=2.0)
    finally:
        server.stop()


# ------------------------------------------------------------------- page web
def _get(url: str, timeout: float = 5.0) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.read().decode()


def test_render_page_is_self_contained():
    page = render_page(WebStyle(font_size=48, text_color="#FFEE00", bg_color="#00FF00"))
    assert "EventSource" in page and "48px" in page and "#FFEE00" in page
    assert "background: #00FF00" in page
    assert "http://" not in page and "https://" not in page  # aucune ressource externe


def test_render_page_transparent_override():
    page = render_page(WebStyle(bg_color="#00FF00"), transparent=True)
    assert "background: transparent" in page


def test_render_page_rejects_injected_color():
    """Une couleur bricolée à la main dans settings.json ne doit pas fuir en CSS."""
    page = render_page(WebStyle(text_color="red; } body { display:none", bg_color="nope"))
    assert "display:none" not in page
    assert "color: #FFFFFF" in page and "background: #00FF00" in page


def test_web_server_serves_page_and_text():
    server = WebSubtitleServer(0, WebStyle())
    server.start()
    port = server._server.server_address[1]
    try:
        server.set_text("première ligne\nseconde ligne")
        assert "EventSource" in _get(f"http://127.0.0.1:{port}/")
        assert _get(f"http://127.0.0.1:{port}/text") == "première ligne\nseconde ligne"
        with pytest.raises(urllib.error.HTTPError) as err:
            _get(f"http://127.0.0.1:{port}/nexistepas")
        assert err.value.code == 404
    finally:
        server.stop()


def _sse_event(stream) -> dict:
    payload = json.loads(stream.readline().decode().removeprefix("data: ").strip())
    stream.readline()  # ligne vide de fin d'événement
    return payload


def test_web_server_sse_pushes_updates():
    server = WebSubtitleServer(0, WebStyle())
    server.start()
    port = server.port
    try:
        server.set_text("avant")
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/events", timeout=5) as stream:
            assert _sse_event(stream) == {"primary": "avant", "secondary": ""}
            server.set_text("après\nsur deux lignes")
            # JSON : un saut de ligne ne peut pas casser le cadrage SSE
            assert _sse_event(stream) == {
                "primary": "après\nsur deux lignes",
                "secondary": "",
            }
            server.set_text("principal", "second sous-titre")
            assert _sse_event(stream) == {
                "primary": "principal",
                "secondary": "second sous-titre",
            }
    finally:
        server.stop()


def test_web_text_endpoint_joins_both_subtitles():
    server = WebSubtitleServer(0, WebStyle())
    server.start()
    try:
        server.set_text("ligne une", "line two")
        assert _get(f"http://127.0.0.1:{server.port}/text") == "ligne une\nline two"
        server.set_text("seule")
        assert _get(f"http://127.0.0.1:{server.port}/text") == "seule"
    finally:
        server.stop()


def test_render_page_handles_both_lines_and_char_limit():
    page = render_page(WebStyle(max_chars_per_line=42))
    assert 'id="primary"' in page and 'id="secondary"' in page
    assert "max-width: 42ch;" in page
    assert "max-width" not in render_page(WebStyle(max_chars_per_line=0))


def test_local_ip_addresses_excludes_loopback():
    from ecoutemoi.core.publish import local_ip_addresses

    for ip in local_ip_addresses():
        assert not ip.startswith("127.")
        assert ip.count(".") == 3  # IPv4 uniquement : c'est ce qu'on tape au clavier


def test_web_urls_lists_lan_addresses_only_when_open():
    from ecoutemoi.core.publish import local_ip_addresses

    closed = WebSubtitleServer(0, WebStyle(), bind_lan=False)
    try:
        assert closed.urls() == [closed.url()]
    finally:
        closed.stop()
    opened = WebSubtitleServer(0, WebStyle(), bind_lan=True)
    try:
        assert len(opened.urls()) == 1 + len(local_ip_addresses())
        assert opened.urls()[0].startswith("http://127.0.0.1:")
    finally:
        opened.stop()


# -------------------------------------------------------------------- façade
def test_publishers_disabled_by_default_and_idempotent():
    pubs = TextPublishers()
    try:
        pubs.apply(Settings())
        assert pubs.active() == [] and pubs.web_url() is None
        pubs.set_text("ne va nulle part")  # ne doit rien lever
    finally:
        pubs.stop()


def test_publishers_start_and_stop_web():
    pubs = TextPublishers()
    try:
        pubs.apply(Settings(web_enabled=True, web_port=0))
        assert pubs.active() == ["web"]
        pubs.set_text("texte diffusé")
        port = pubs.web._server.server_address[1]
        assert _get(f"http://127.0.0.1:{port}/text") == "texte diffusé"
        pubs.apply(Settings(web_enabled=False))  # même façade, sortie coupée
        assert pubs.active() == []
    finally:
        pubs.stop()


def test_publishers_report_busy_port():
    """Port déjà pris (autre instance de l'app) : erreur exploitable, jamais
    d'exception jusqu'à la GUI, et surtout jamais un « démarré » silencieux.

    Sous Windows, SO_REUSEADDR laisserait DEUX serveurs se lier au même port :
    le conflit doit remonter là aussi, d'où allow_reuse_address désactivé sur
    cette plateforme (voir WebSubtitleServer).
    """
    first = WebSubtitleServer(0, WebStyle())
    first.start()
    pubs = TextPublishers()
    try:
        pubs.apply(Settings(web_enabled=True, web_port=first.port))
        assert pubs.web is None
        assert pubs.web_error and str(first.port) in pubs.web_error
    finally:
        pubs.stop()
        first.stop()
