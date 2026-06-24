"""Browser-backed sign-in and chat-token capture.

Playwright support for the pure-HTTP :class:`copilot.client.Copilot`: it does NOT
chat. Its sole job is to establish and refresh the signed-in session that the
HTTP driver runs on — interactive Microsoft/Google login plus headless capture of
the Copilot chat token.

``BrowserCopilot`` launches a **persistent** Playwright Chromium profile so that
Cloudflare clearance and any sign-in survive restarts. Two responsibilities:

  * :meth:`login` — opens a visible window for interactive sign-in, then warms up
    one chat turn to mint the token and snapshots ``session/token.json``.
  * :meth:`acquire_chat_token` — headless: returns the chat token, warming up a
    turn to mint/capture it when the MSAL cache can't be read directly.

Why a warm-up + WebSocket capture (not a localStorage read): federated *Google*
logins store the MSAL token cache **encrypted** and only mint the
``ChatAI.ReadWrite`` token on the first chat turn. So the token can't be read
from storage; instead we let the page open its own ``wss://.../c/api/chat``
socket and read ``accessToken`` (and ``X-UserIdentityType``) straight off that
URL — see :meth:`_install_ws_listener`. Microsoft logins expose a readable token
and skip the warm-up entirely.

All actual chatting lives in :mod:`copilot.driver` (pure HTTP). Recapture token
shapes with ``tests/diagnostic.py`` if Microsoft changes them.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional
from urllib.parse import parse_qs, urlparse

from playwright.sync_api import sync_playwright, Error as PlaywrightError

from .auth import DEFAULT_AUTH_FILE, DEFAULT_PROFILE_DIR

COPILOT_URL = "https://copilot.microsoft.com/"

# --- in-page JavaScript -----------------------------------------------------

# Discover the Copilot chat MSAL access token from localStorage. The cache holds
# several tokens for different scopes; the chat WebSocket only accepts the one
# scoped 'ChatAI.ReadWrite' — a wrong-audience token (e.g. the Graph
# User.Read/Files.Read token) makes the WS upgrade 401. We therefore PREFER the
# ChatAI token and only fall back to the first token found if none matches.
# Returns null for anonymous sessions (anonymous chat may still work via cookies).
_FIND_TOKEN_JS = """
() => {
  try {
    let fallback = null;
    for (let i = 0; i < localStorage.length; i++) {
      const k = localStorage.key(i);
      const v = localStorage.getItem(k);
      if (v && v.indexOf('"credentialType":"AccessToken"') !== -1) {
        try {
          const o = JSON.parse(v);
          if (o && o.secret) {
            // Match the chat scope (e.g. '<resource>/ChatAI.ReadWrite'); take the
            // first non-matching token only as a last-resort fallback.
            if (o.target && o.target.indexOf('ChatAI') !== -1) return o.secret;
            if (!fallback) fallback = o.secret;
          }
        } catch (e) {}
      }
    }
    return fallback;
  } catch (e) {}
  return null;
}
"""

# True once the user is signed in, *before* the chat token is minted. MSAL writes
# an `msal.*.account.keys` index (a non-empty list of cached accounts) the moment
# sign-in completes — and, crucially, this index is NOT encrypted even when the
# token cache itself is, so it is a reliable sign-in signal for every account
# type (Microsoft *and* federated Google). We deliberately do not key off the
# ChatAI access token here: for Google logins MSAL stores the token cache
# *encrypted* ({id,nonce,data,...}) and only mints the chat token on the first
# chat turn, so waiting for it during login would never succeed (see login()).
_SIGNED_IN_JS = """
() => {
  try {
    for (let i = 0; i < localStorage.length; i++) {
      const k = localStorage.key(i);
      if (k && k.indexOf('account.keys') !== -1) {
        try {
          const a = JSON.parse(localStorage.getItem(k) || 'null');
          if (Array.isArray(a) ? a.length > 0 : (a && Object.keys(a).length > 0))
            return true;
        } catch (e) {}
      }
    }
  } catch (e) {}
  return false;
}
"""


class BrowserCopilot:
    """Drives Microsoft Copilot through a real Playwright browser.

    Parameters
    ----------
    profile_dir:
        Directory for the persistent Chromium profile (cookies, Cloudflare
        clearance, sign-in). Reused across runs.
    headless:
        Run without a visible window. Use ``False`` (or :meth:`login`) for the
        first interactive sign-in, then ``True`` afterwards.
    """

    label = "Microsoft Copilot (browser)"
    default_model = "Copilot"

    def __init__(
        self,
        profile_dir: str = DEFAULT_PROFILE_DIR,
        headless: bool = True,
        nav_timeout: int = 60,
        proxy: Optional[str] = None,
    ):
        self.profile_dir = str(Path(profile_dir).resolve())
        self.headless = headless
        self.nav_timeout = nav_timeout
        # Copilot consumer chat is geo-restricted. If you are outside a supported
        # region, route the browser through a proxy/VPN in a supported region,
        # e.g. proxy="http://user:pass@host:port" or "socks5://host:port".
        self.proxy = proxy

        self._pw = None
        self._context = None
        self._page = None
        self._login_log_fh = None
        # Chat token captured live off the page's own chat WebSocket. This is the
        # only way to recover the token for sessions whose MSAL cache is encrypted
        # (e.g. federated Google logins), where _FIND_TOKEN_JS cannot read it.
        self._captured_chat_token: Optional[str] = None
        self._captured_identity_type: Optional[str] = None
        self._ws_listener_installed = False

    # -- lifecycle ----------------------------------------------------------

    def start(self, headless: Optional[bool] = None) -> "BrowserCopilot":
        """Launch the persistent browser context and open Copilot."""
        if self._context is not None:
            return self
        if headless is not None:
            self.headless = headless
        try:
            self._pw = sync_playwright().start()
            launch_kwargs = dict(
                headless=self.headless,
                args=["--disable-blink-features=AutomationControlled"],
            )
            if self.proxy:
                launch_kwargs["proxy"] = self._parse_proxy(self.proxy)
            self._context = self._pw.chromium.launch_persistent_context(
                self.profile_dir,
                **launch_kwargs,
            )
            self._page = self._context.pages[0] if self._context.pages else self._context.new_page()
            self._page.set_default_timeout(self.nav_timeout * 1000)
            self._page.goto(COPILOT_URL, wait_until="domcontentloaded")
            # Give Cloudflare a moment to clear on first paint. We deliberately do
            # NOT wait for "networkidle": Copilot's SPA keeps telemetry/heartbeat
            # connections open indefinitely, so the network never goes idle and the
            # wait would always time out. A short fixed settle is enough.
            self._page.wait_for_timeout(2000)
        except PlaywrightError as exc:
            self.close()
            raise ConnectionError(f"Failed to start browser: {exc}") from exc
        return self

    @staticmethod
    def _parse_proxy(proxy: str) -> dict:
        """Turn a ``scheme://user:pass@host:port`` string into Playwright form."""
        from urllib.parse import urlparse

        u = urlparse(proxy)
        server = f"{u.scheme}://{u.hostname}:{u.port}" if u.port else f"{u.scheme}://{u.hostname}"
        cfg = {"server": server}
        if u.username:
            cfg["username"] = u.username
        if u.password:
            cfg["password"] = u.password
        return cfg

    def region_blocked(self) -> bool:
        """True if Copilot is showing the 'Not available in your region' notice."""
        if self._page is None:
            return False
        try:
            text = self._page.evaluate("() => document.body ? document.body.innerText : ''")
        except PlaywrightError:
            return False
        return "available in your region" in (text or "").lower()

    def close(self) -> None:
        for attr, closer in (
            ("_context", lambda c: c.close()),
            ("_pw", lambda p: p.stop()),
            ("_login_log_fh", lambda f: f.close()),
        ):
            obj = getattr(self, attr, None)
            if obj is not None:
                try:
                    closer(obj)
                except Exception:
                    pass
                setattr(self, attr, None)
        self._page = None

    def __enter__(self) -> "BrowserCopilot":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.close()

    # -- auth ---------------------------------------------------------------

    def login(self, path: str = DEFAULT_AUTH_FILE, timeout: int = 300) -> dict:
        """Open a visible window for interactive Microsoft/Google sign-in.

        Auto-detects success — a cached account appearing in the page (the moment
        sign-in completes, see :data:`_SIGNED_IN_JS`) — then **warms up** the
        session with one throwaway chat turn to mint the Copilot chat token and
        captures it off the page's own chat WebSocket. This warm-up is what makes
        federated *Google* logins work: their MSAL cache is encrypted and the chat
        token is only minted on the first turn, so the old "wait for the token in
        localStorage" approach timed out (~5 min) and saved a null token.
        Microsoft accounts already have a readable token, so the warm-up returns
        instantly and their flow is unchanged.

        No key-press needed; the browser closes itself. Every step is appended to
        ``<session>/login.log``. ``timeout`` bounds the wait. The session persists
        in ``profile_dir`` for headless reuse.
        """
        self.close()
        self.start(headless=False)
        self._install_ws_listener()

        log = self._open_login_log(Path(path).resolve().parent / "login.log")
        log(f"login started; browser open at {COPILOT_URL}")
        self._mirror_page_events(log)

        print(
            "\nA browser window is open at copilot.microsoft.com.\n"
            "Sign in (and pass any 'verify you're human' check).\n"
            "It finishes by itself once sign-in is detected — no need to press Enter.\n"
        )

        # Wait for sign-in (a cached account), not for the chat token: the token
        # may not exist until the first turn. Bail early on window close/timeout.
        detected = False
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._window_closed():
                log("browser window closed before sign-in was detected")
                break
            if self.signed_in():
                log("sign-in detected (account cached)")
                detected = True
                break
            try:
                self._page.wait_for_timeout(1500)
            except PlaywrightError:
                break

        token = None
        if detected:
            print("Signed in — finishing setup (sending a warm-up message)...")
            log("warming up to mint/capture the chat token")
            try:
                token = self.acquire_chat_token(timeout=max(30, int(deadline - time.time())))
            except PlaywrightError as exc:
                log(f"warm-up error: {exc}")
            log(f"chat token captured: {'yes' if token else 'no'}"
                f" (identity={self._captured_identity_type})")
        else:
            log(f"not signed in within {timeout}s; snapshotting current state")
            print("Sign-in not detected; saving whatever session state exists.")

        # Snapshot for the headless curl_cffi path.
        auth: dict = {}
        try:
            auth = self.export_auth(path=path, stamp=time.time())
            log(f"auth snapshot saved to {path} (access_token={'yes' if auth.get('access_token') else 'no'}"
                f", identity={auth.get('identity_type')})")
            print(f"Auth snapshot saved to {path}")
        except Exception as exc:
            log(f"could not snapshot auth: {exc}")
            print(f"(could not snapshot auth: {exc})")

        log("closing browser")
        self.close()
        print(f"Session saved to {self.profile_dir}")
        return auth

    def _open_login_log(self, log_path: Path):
        """Return a best-effort timestamped append-logger to ``log_path``.

        The handle is parked on the context so :meth:`close` can release it; if the
        file can't be opened, the returned logger is a silent no-op.
        """
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            self._login_log_fh = log_path.open("a", encoding="utf-8")
        except OSError:
            self._login_log_fh = None

        def log(message: str) -> None:
            fh = self._login_log_fh
            if fh is None:
                return
            try:
                fh.write(f"{datetime.now(timezone.utc).isoformat()}\t{message}\n")
                fh.flush()
            except Exception:
                pass

        return log

    def _mirror_page_events(self, log) -> None:
        """Stream main-frame navigations and console errors into the login log."""
        try:
            self._page.on(
                "framenavigated",
                lambda fr: fr == self._page.main_frame and log(f"navigated: {fr.url}"),
            )
            self._page.on(
                "console",
                lambda m: m.type == "error" and log(f"console.error: {m.text}"),
            )
        except PlaywrightError:
            pass

    def _window_closed(self) -> bool:
        """True if the page/context is gone (e.g. the user closed the window)."""
        try:
            return self._page is None or self._page.is_closed()
        except Exception:
            return True

    def access_token(self) -> Optional[str]:
        """Return the Copilot chat token, or ``None`` if not available.

        Prefers a token captured live off the page's own chat WebSocket (the only
        source that works when the MSAL cache is encrypted, e.g. Google logins),
        and otherwise falls back to reading the unencrypted MSAL cache via
        ``_FIND_TOKEN_JS`` (Microsoft logins). Call :meth:`acquire_chat_token`
        first to ensure one of these is populated.
        """
        if self._captured_chat_token:
            return self._captured_chat_token
        self._ensure_started()
        try:
            return self._page.evaluate(_FIND_TOKEN_JS)
        except PlaywrightError:
            return None

    def signed_in(self) -> bool:
        """True once a Microsoft/Google account is cached (sign-in complete)."""
        self._ensure_started()
        try:
            return bool(self._page.evaluate(_SIGNED_IN_JS))
        except PlaywrightError:
            return False

    def _install_ws_listener(self) -> None:
        """Capture the chat token off the page's own chat WebSocket.

        The page opens ``wss://.../c/api/chat?...&accessToken=<token>`` (plus, for
        federated logins, ``&X-UserIdentityType=google``) when it sends a turn.
        Reading the token here is encryption-proof: the page has already decrypted
        it. parse_qs URL-decodes the value, so we store the raw token (the drivers
        re-quote it when building their own socket URL)."""
        if self._ws_listener_installed or self._page is None:
            return

        def on_ws(ws):
            try:
                url = ws.url
                if "/c/api/chat" not in url or "accessToken=" not in url:
                    return
                q = parse_qs(urlparse(url).query)
                tok = (q.get("accessToken") or [None])[0]
                if tok:
                    self._captured_chat_token = tok
                    self._captured_identity_type = (q.get("X-UserIdentityType") or [None])[0]
            except Exception:
                pass

        try:
            self._page.on("websocket", on_ws)
            self._ws_listener_installed = True
        except PlaywrightError:
            pass

    def _send_warmup(self, text: str = "hi") -> bool:
        """Send one message through the page composer to mint the chat token.

        Returns True if a send was attempted. Federated (Google) sessions only
        mint the ChatAI token on the first chat turn, so we trigger one here and
        let :meth:`_install_ws_listener` capture the token off the resulting
        socket."""
        for sel in ("textarea", "div[contenteditable='true']", "[role='textbox']"):
            try:
                self._page.wait_for_selector(sel, state="visible", timeout=8000)
            except PlaywrightError:
                continue
            try:
                self._page.click(sel)
                self._page.keyboard.type(text, delay=15)
                self._page.keyboard.press("Enter")
                return True
            except PlaywrightError:
                continue
        return False

    def acquire_chat_token(
        self, timeout: int = 60, warmup: bool = True, signin_grace: int = 8
    ) -> Optional[str]:
        """Return a usable chat token, minting it via a warm-up turn if needed.

        Fast path: a token already readable (captured, or unencrypted MSAL cache)
        is returned immediately — this is the common Microsoft case. Otherwise, if
        ``warmup`` and the user is signed in, send one throwaway message and
        capture the token off the chat WebSocket (the encrypted-cache / Google
        case). Returns ``None`` if no token could be obtained within ``timeout``.

        ``signin_grace`` bounds how long we wait for an *existing* sign-in to
        register before giving up. A headless refresh can't perform interactive
        sign-in, so on a not-signed-in profile we bail after this short grace
        instead of blocking the full ``timeout`` — that wait is what made the
        no-session path feel hung before it fell through to a visible login.
        Sign-in normally registers within ~1-2s of page load (an already-signed-in
        profile passes the grace immediately).
        """
        self._ensure_started()
        self._install_ws_listener()

        tok = self.access_token()
        if tok or not warmup:
            return tok

        deadline = time.time() + timeout
        signin_deadline = time.time() + min(signin_grace, timeout)
        while time.time() < signin_deadline and not self.signed_in():
            if self._window_closed():
                return None
            self._page.wait_for_timeout(500)
        if not self.signed_in():
            return None

        if not self._send_warmup():
            return self.access_token()

        while time.time() < deadline:
            if self._captured_chat_token:
                return self._captured_chat_token
            if self._window_closed():
                break
            self._page.wait_for_timeout(500)
        return self.access_token()

    def cookies(self) -> Dict[str, str]:
        """Return the signed-in Microsoft cookies as a name->value dict."""
        self._ensure_started()
        try:
            raw = self._context.cookies()
        except PlaywrightError:
            return {}
        return {c["name"]: c["value"] for c in raw if "microsoft.com" in c.get("domain", "")}

    def export_auth(self, path: str = DEFAULT_AUTH_FILE, stamp: Optional[float] = None) -> dict:
        """Snapshot the signed-in cookies + access token to ``path`` as JSON.

        ``stamp`` is the epoch seconds to record as ``saved_at`` (pass
        ``time.time()`` from the caller). Returns the auth dict.
        """
        auth = {
            "cookies": self.cookies(),
            "access_token": self.access_token(),
            # Federated logins (Google) ride an extra &X-UserIdentityType= on the
            # chat socket; the drivers replay it. None for Microsoft accounts.
            "identity_type": self._captured_identity_type,
            "saved_at": stamp if stamp is not None else 0,
        }
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(auth, indent=2), encoding="utf-8")
        return auth

    # -- internals ----------------------------------------------------------

    def _ensure_started(self) -> None:
        if self._context is None or self._page is None:
            self.start()
