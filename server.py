"""Local web host for Godhood Trials' independent learning population."""
from __future__ import annotations

import argparse
from collections import deque
import json
import mimetypes
import os
from pathlib import Path
import socket
import sys
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlsplit

from sim.ecology import VERBS as ECOLOGY_VERBS
from sim.social import TONES, LEGACY_TONES
from sim.world import DIRECTIONS, ITEMS, RECIPES, World

BASE_RATE = 4
SPEEDS = (1, 5, 20, "max")
VERBS = {"tap", "tone", "make_seeds", "plant", "revive", "move", "rest", "wait", "eat", "drink", "gather", "pickup", "give", "drop", "build", "toggle_door", "dismantle"}
VERBS.update(ECOLOGY_VERBS)

# Explicit publication assets only; operational logs and saved lives are private.
PUBLIC_PROGRESS = {
    'docs/progress/' + name for name in (
        'index.html', 'progress.css', 'progress.js', 'results.js', 'results.json',
        'replay.js', 'README.md', 'preview.jpg', 'replay-preview.jpg', 'world-preview.jpg',
    )
} | {
    'README.md', 'CHANGELOG.md', 'LICENSE', 'web/icon.svg',
    'docs/PROGRESS.md', 'docs/PLAYING.md', 'docs/MILESTONES.md',
    'docs/PRACTICE-AMOUNT.md', 'docs/VOLUNTARY-COORDINATION.md',
    'docs/CULLING-TRIAL.md', 'docs/VALUE-INTERFERENCE.md', 'docs/PROGRESS-VALIDATION.md',
}


class Runtime:
    def __init__(self, root: Path, *, seed=1729, public_demo=False, heartbeat_timeout=3.0, autoload=True):
        self.root = Path(root).resolve()
        self.runs = self.root / "runs"
        self.runs.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.world = World(seed)
        self.public_demo = public_demo
        self.speed = 1
        self.paused = True
        self.pause_reason = "Waiting for a viewer"
        self.manual_pause = False
        self.last_heartbeat = 0.0
        self.heartbeat_timeout = heartbeat_timeout
        self.pending_action = None
        self.last_message = ""
        self.message_serial = 0
        self.saved_at = None
        self.achieved_speed = 0.0
        self.stop_event = threading.Event()
        self.thread = None
        self._samples = deque()
        if autoload:
            candidate = self.runs / "autosave.json"
            if not candidate.exists():
                candidate = self.runs / "world.json"
            if candidate.exists():
                try:
                    self.world = World.load(candidate)
                    self.saved_at = datetime.fromtimestamp(candidate.stat().st_mtime, timezone.utc).isoformat()
                except (OSError, ValueError, KeyError, TypeError, IndexError):
                    self._message("The previous save could not be read; a fresh world is ready.")

    def _message(self, message):
        self.last_message = message
        self.message_serial += 1

    def _save(self, filename):
        self.world.save(self.runs / filename)
        self.saved_at = datetime.now(timezone.utc).isoformat()

    def state_bytes(self, terrain=False, perspective="player"):
        # World.state contains references to resources/terrain. Serialize while locked.
        with self.lock:
            self.last_heartbeat = time.monotonic()
            if not self.manual_pause:
                self.paused = False
                self.pause_reason = ""
            state = self.world.state(terrain=terrain)
            if perspective not in self.world.residents:
                perspective = "player"
            viewer = self.world.residents[perspective]
            state["perception"] = {"resident": perspective, "origin": [viewer.x, viewer.y],
                                   "observation": self.world.observe(perspective)}
            state.update(paused=self.paused, speed=self.speed, achievedSpeed=self.achieved_speed,
                         publicDemo=self.public_demo, lastMessage=self.last_message,
                         messageSerial=self.message_serial, savedAt=self.saved_at,
                         pauseReason=self.pause_reason, queuedAction=self.pending_action is not None,
                         hasSave=(self.runs / "world.json").is_file())
            return json.dumps(state, separators=(",", ":"), allow_nan=False).encode("utf-8")

    def _tick(self):
        command = self.pending_action
        self.pending_action = None
        results = self.world.step({"player": command} if command else None)
        if command:
            self._message(results["player"][1])
        self._samples.append(time.monotonic())

    def control(self, data):
        with self.lock:
            if self.public_demo:
                return False, "This shared demo is read-only."
            command = data.get("command")
            if command == "pause":
                self.manual_pause = self.paused = True
                self.pause_reason = "Paused by you"
                self.achieved_speed = 0.0
                message = "World paused."
            elif command == "resume":
                self.manual_pause = self.paused = False
                self.pause_reason = ""
                self.last_heartbeat = time.monotonic()
                message = "World resumed."
            elif command == "step":
                self.manual_pause = self.paused = True
                self.pause_reason = "Paused by you"
                self.achieved_speed = 0.0
                had_action = self.pending_action is not None
                self._tick()
                message = self.last_message if had_action else "Advanced one tick."
            elif command == "speed":
                value = data.get("value")
                if type(value) not in (int, str) or value not in SPEEDS:
                    return False, "Choose 1, 5, 20, or max speed."
                if value != 1 and self.speed == 1:
                    self._save("before-fast-forward.json")
                self.speed = value
                message = "Maximum speed selected." if value == "max" else f"Speed set to {value}×."
            elif command == "save":
                self._save("world.json")
                message = "World saved."
            elif command == "load":
                path = self.runs / "world.json"
                if not path.is_file():
                    return False, "Save a world first."
                loaded = World.load(path)
                self._save("before-load.json")
                self.world = loaded
                self.pending_action = None
                self.speed = 1
                self.manual_pause = self.paused = True
                self.pause_reason = "Loaded; paused by you"
                self.achieved_speed = 0.0
                self._samples.clear()
                message = "Saved world loaded and paused."
            else:
                return False, "Unknown control."
            self._message(message)
            return True, message

    def action(self, data):
        with self.lock:
            if self.public_demo:
                return False, "This shared demo is read-only."
            if set(data) - {"verb", "direction", "item", "target", "kind", "x", "y", "tone"}:
                return False, "Unknown action field."
            verb = data.get("verb")
            if not isinstance(verb, str) or verb not in VERBS:
                return False, "Unknown action."
            if "tone" in data and (verb != "tone" or not isinstance(data["tone"], str) or data["tone"] not in TONES + tuple(LEGACY_TONES)):
                return False, "Choose one tone from 0 to 9."
            if verb == "move" and data.get("direction") not in DIRECTIONS:
                return False, "Choose north, east, south, or west."
            if "direction" in data and data["direction"] not in DIRECTIONS:
                return False, "Unknown direction."
            if "item" in data and data["item"] not in ITEMS:
                return False, "Unknown item."
            if "target" in data and data["target"] not in self.world.residents:
                return False, "Unknown resident."
            if "kind" in data and data["kind"] not in RECIPES:
                return False, "Unknown construction piece."
            if "x" in data or "y" in data:
                if any(type(data.get(k)) is not int for k in ("x", "y")) or not self.world.inside(data["x"], data["y"]):
                    return False, "Choose a tile in the valley."
                if verb not in ("plant", "build", "toggle_door", "dismantle", "water_crop"):
                    return False, "This action does not target a tile."
            if self.pending_action is not None:
                return False, "One action is already queued; let the next tick finish."
            self.pending_action = dict(data)
            self.speed = 1
            message = "Action queued. Step or resume to perform it." if self.paused else "Action queued for the next tick."
            return True, message

    def start(self):
        self.thread = threading.Thread(target=self._run, name="world-clock", daemon=True)
        self.thread.start()

    def _run(self):
        next_tick = time.monotonic()
        last_save = next_tick
        while not self.stop_event.is_set():
            now = time.monotonic()
            with self.lock:
                if not self.manual_pause and now - self.last_heartbeat > self.heartbeat_timeout:
                    self.paused = True
                    self.pause_reason = "No viewer; time is paused"
                if self.paused:
                    next_tick = now
                    self.achieved_speed = 0.0
                    self._samples.clear()
                    delay = .025
                elif self.speed == "max":
                    # Small batches release the lock regularly, even on a fast machine.
                    for _ in range(4):
                        if self.stop_event.is_set() or self.paused:
                            break
                        self._tick()
                    delay = .001
                else:
                    interval = 1 / (BASE_RATE * self.speed)
                    if now >= next_tick:
                        self._tick()
                        next_tick = max(next_tick + interval, now + interval / 4)
                    delay = min(.025, max(.001, next_tick - time.monotonic()))
                cutoff = now - 2.0
                while self._samples and self._samples[0] < cutoff:
                    self._samples.popleft()
                if not self.paused and len(self._samples) >= 2:
                    elapsed = self._samples[-1] - self._samples[0]
                    self.achieved_speed = (len(self._samples) - 1) / max(.001, elapsed) / BASE_RATE
                if now - last_save >= 5:
                    try:
                        self._save("autosave.json")
                    except OSError:
                        self._message("Autosave failed. Check available disk space and permissions.")
                    last_save = now
            self.stop_event.wait(delay)

    def close(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=10)
        with self.lock:
            self._save("autosave.json")


class WorldServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, runtime):
        self.runtime = runtime
        self.allowed_hosts = {"localhost", "127.0.0.1", "::1"}
        if runtime.public_demo:
            try:
                self.allowed_hosts.update(socket.gethostbyname_ex(socket.gethostname())[2])
            except OSError:
                pass
        super().__init__(address, Handler)
        if address[0] != "0.0.0.0":
            self.allowed_hosts.add(address[0].lower())


class Handler(BaseHTTPRequestHandler):
    server_version = "GodhoodTrials/1"

    def _authorized(self, mutation=False):
        host = self.headers.get("Host", "")
        try:
            authority = urlsplit("//" + host)
            if authority.hostname not in self.server.allowed_hosts or authority.port != self.server.server_port or authority.username or authority.password or authority.path or authority.query or authority.fragment:
                return False
        except ValueError:
            return False
        origin = self.headers.get("Origin")
        if origin is not None and origin != "http://" + host:
            return False
        return not mutation or origin is not None

    def _send(self, status, body, content_type="application/json; charset=utf-8", *, head=False, progress=False):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        styles = "'self' 'unsafe-inline'" if progress else "'self'"
        self.send_header("Content-Security-Policy", f"default-src 'self'; script-src 'self'; style-src {styles}; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")
        self.end_headers()
        if not head:
            self.wfile.write(body)

    def _json(self, status, data):
        self._send(status, json.dumps(data, allow_nan=False).encode("utf-8"))

    def do_HEAD(self):
        self._get(head=True)

    def do_GET(self):
        self._get()

    def _get(self, head=False):
        if not self._authorized():
            self._json(403, {"ok": False, "message": "Local host or origin required."})
            return
        parsed = urlsplit(self.path)
        if parsed.path == "/health":
            population = getattr(self.server.runtime,'population',None)
            body = json.dumps({"ok": True, "service": "godhood-trials", "pid": os.getpid(),
                               "publicDemo": self.server.runtime.public_demo,
                               "controllerMode": 'independent-learning' if population else 'scripted-reference',
                               "tick": self.server.runtime.world.tick,
                               "paused": self.server.runtime.paused,
                               "manualPause": self.server.runtime.manual_pause,
                               "models": {rid: getattr(b,'backend','one-step') for rid,b in population.brains.items()} if population else {},
                               "learners": len(population.brains) if population else 0}).encode("utf-8")
            self._send(200, body, head=head)
            return
        if parsed.path == "/api/state":
            query = parse_qs(parsed.query)
            body = self.server.runtime.state_bytes(terrain=query.get("terrain") == ["1"],
                                                  perspective=query.get("perspective", ["player"])[0])
            self._send(200, body, head=head)
            return
        path = unquote(parsed.path)
        if "\\" in path or "\x00" in path or ".." in path.split("/"):
            self._json(404, {"ok": False, "message": "File not found."})
            return
        root = self.server.runtime.root.resolve()
        web = (root / "web").resolve()
        relative = path.lstrip("/")
        progress = relative in PUBLIC_PROGRESS
        candidate = (root / relative if progress else web / ("index.html" if path == "/" else relative)).resolve()
        boundary = root if progress else web
        if not candidate.is_relative_to(boundary) or not candidate.is_file():
            self._json(404, {"ok": False, "message": "File not found."})
            return
        try:
            body = candidate.read_bytes()
        except OSError:
            self._json(404, {"ok": False, "message": "File not found."})
            return
        content_type = mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
        if content_type.startswith("text/") or candidate.suffix in (".js", ".svg"):
            content_type += "; charset=utf-8"
        self._send(200, body, content_type, head=head, progress=progress)

    def do_POST(self):
        if self.server.runtime.public_demo:
            self._json(403, {"ok": False, "message": "This shared demo is read-only."})
            return
        if not self._authorized(mutation=True):
            self._json(403, {"ok": False, "message": "Same-origin local requests required."})
            return
        if self.headers.get("Content-Type", "").split(";")[0].strip().lower() != "application/json":
            self._json(415, {"ok": False, "message": "Send application/json."})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 16384 or self.headers.get("Transfer-Encoding"):
                raise ValueError("Invalid body length")
            data = json.loads(self.rfile.read(length))
            if not isinstance(data, dict):
                raise ValueError("Object required")
            route = urlsplit(self.path).path
            if route == "/api/control":
                ok, message = self.server.runtime.control(data)
            elif route == "/api/action":
                ok, message = self.server.runtime.action(data)
            else:
                self._json(404, {"ok": False, "message": "Unknown endpoint."})
                return
        except (ValueError, TypeError, KeyError, UnicodeError, OSError, IndexError):
            self._json(400, {"ok": False, "message": "Invalid request or unreadable save."})
            return
        self._json(200 if ok else 400, {"ok": ok, "message": message})

    def log_message(self, format, *args):
        # Avoid a line per UI poll; unexpected request errors remain visible.
        if len(args) > 1 and str(args[1]) not in ("200", "304"):
            super().log_message(format, *args)

    def setup(self):
        super().setup()
        self.connection.settimeout(5)


def create_runtime(root, *, seed=1729, public_demo=False):
    """The playable entry point requires learning; it never substitutes scripts."""
    from experiments.living_population import LearningRuntime
    return LearningRuntime(root, seed=seed, backend='recurrent-ppo',
                           migrate_world=True, public_demo=public_demo)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8788)
    parser.add_argument("--host", default="127.0.0.1", choices=("127.0.0.1", "localhost", "0.0.0.0"))
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--public-demo", action="store_true", help="Allow LAN viewing; all HTTP mutations disabled")
    args = parser.parse_args()
    if args.host == "0.0.0.0" and not args.public_demo:
        parser.error("Binding all interfaces requires --public-demo")
    root = Path(__file__).resolve().parent
    # Reuse this module when launched as a script, so the learner imports one host.
    sys.modules.setdefault('server',sys.modules[__name__])
    from sim.checkpoint_lease import checkpoint_lease
    with checkpoint_lease(root / 'runs'):
        serve(root, args)


def serve(root, args):
    runtime = create_runtime(root, seed=args.seed, public_demo=args.public_demo)
    server = WorldServer((args.host, args.port), runtime)
    runtime.start()
    stop_path = runtime.runs / f"stop-{os.getpid()}.request"

    def watch_shutdown():
        while not runtime.stop_event.wait(.2):
            if stop_path.is_file():
                # A local launcher writes this only after checking process ownership.
                server.shutdown()
                return

    watcher = threading.Thread(target=watch_shutdown, name="launcher-stop", daemon=True)
    watcher.start()
    print(f"Godhood Trials: http://127.0.0.1:{server.server_port}/", flush=True)
    try:
        server.serve_forever(poll_interval=.1)
    except KeyboardInterrupt:
        pass
    finally:
        runtime.close()
        server.server_close()
        watcher.join(timeout=1)
        if stop_path.is_file():
            stop_path.unlink()


if __name__ == "__main__":
    main()
