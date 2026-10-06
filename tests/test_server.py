"""Isolated localhost tests. No external network or third-party packages."""
import http.client
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest

from server import Runtime, WorldServer


class ServerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "web").mkdir()
        (self.root / "web" / "index.html").write_text("<h1>Valley</h1>", encoding="utf-8")
        (self.root / "private.txt").write_text("not public", encoding="utf-8")
        self.runtime = Runtime(self.root, autoload=False)
        self.server = WorldServer(("127.0.0.1", 0), self.runtime)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.port = self.server.server_port

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.runtime.close()
        self.temp.cleanup()

    def request(self, path, data=None, headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=3)
        supplied = dict(headers or {})
        if data is not None:
            supplied.setdefault("Origin", f"http://127.0.0.1:{self.port}")
            supplied.setdefault("Content-Type", "application/json")
        connection.request("POST" if data is not None else "GET", path, json.dumps(data) if data is not None else None, supplied)
        response = connection.getresponse()
        status, body = response.status, response.read()
        connection.close()
        try:
            body = json.loads(body)
        except (ValueError, UnicodeError):
            pass
        return status, body

    def test_state_heartbeat_and_manual_pause(self):
        self.assertTrue(self.runtime.paused)
        status, state = self.request("/api/state?terrain=1")
        self.assertEqual(status, 200)
        self.assertFalse(state["paused"])
        self.assertEqual(len(state["terrain"]), 64)
        self.assertIn("messageSerial", state)
        self.request("/api/control", {"command": "pause"})
        self.assertTrue(self.request("/api/state")[1]["paused"])
        self.assertEqual(self.runtime.world.tick, 0)

    def test_action_queues_without_tick_and_step_consumes_once(self):
        self.request("/api/control", {"command": "pause"})
        status, reply = self.request("/api/action", {"verb": "drink"})
        self.assertEqual(status, 200)
        self.assertTrue(reply["ok"])
        self.assertEqual(self.runtime.world.tick, 0)
        self.assertIsNotNone(self.runtime.pending_action)
        self.assertEqual(self.request("/api/action", {"verb": "rest"})[0], 400)
        self.request("/api/control", {"command": "step"})
        state = self.request("/api/state")[1]
        self.assertEqual(state["tick"], 1)
        self.assertFalse(state["queuedAction"])
        self.assertEqual(state["lastMessage"], "Stand beside the river to drink.")
        self.assertTrue(state["paused"])

    def test_manual_save_load_and_fast_forward_backups(self):
        self.request("/api/control", {"command": "pause"})
        self.request("/api/control", {"command": "save"})
        self.assertTrue(self.request("/api/state")[1]["hasSave"])
        self.request("/api/control", {"command": "step"})
        self.request("/api/control", {"command": "speed", "value": 20})
        self.assertEqual(json.loads((self.root / "runs/before-fast-forward.json").read_text())["tick"], 1)
        self.request("/api/control", {"command": "load"})
        self.assertEqual(self.runtime.world.tick, 0)
        self.assertTrue(self.runtime.paused)
        self.assertEqual(self.runtime.speed, 1)
        self.assertEqual(json.loads((self.root / "runs/before-load.json").read_text())["tick"], 1)
        self.assertEqual(self.request("/api/control", {"command": "speed", "value": True})[0], 400)

    def test_origin_host_content_type_and_action_validation(self):
        self.assertEqual(self.request("/api/state", headers={"Host": f"untrusted.example:{self.port}"})[0], 403)
        self.assertEqual(self.request("/api/control", {"command": "pause"}, {"Origin": "https://untrusted.example"})[0], 403)
        self.assertEqual(self.request("/api/control", {"command": "pause"}, {"Content-Type": "text/plain"})[0], 415)
        self.assertEqual(self.request("/api/action", {"verb": "move", "direction": "up"})[0], 400)
        self.assertEqual(self.request("/api/action", {"verb": "gather", "rid": "r0"})[0], 400)
        self.assertEqual(self.request("/api/action", {"verb": "give", "target": []})[0], 400)
        self.assertEqual(self.runtime.world.tick, 0)

    def test_static_files_and_traversal(self):
        self.assertEqual(self.request("/")[0], 200)
        for path in ("/../private.txt", "/%2e%2e/private.txt", "/..%5cprivate.txt", "/runs/world.json"):
            self.assertEqual(self.request(path)[0], 404, path)
        self.assertEqual(self.request("/health")[1]["service"], "godhood-trials")

    def test_progress_assets_do_not_expose_private_documents_or_wake_world(self):
        progress = self.root / 'docs/progress'
        progress.mkdir(parents=True)
        (progress / 'index.html').write_text('<h1>Recorded evidence</h1>')
        (progress / 'results.json').write_text('{"recorded":true}')
        (self.root / 'docs/POPULATION-WATCH-LOG.md').write_text('private log')
        (self.root / 'docs/unlisted.md').write_text('not a publication asset')
        self.assertEqual(self.request('/docs/progress/index.html')[0], 200)
        self.assertEqual(self.request('/docs/progress/results.json')[1], {'recorded': True})
        for path in ('/docs/POPULATION-WATCH-LOG.md', '/docs/unlisted.md',
                     '/docs/progress/../POPULATION-WATCH-LOG.md', '/runs/population.pt'):
            self.assertEqual(self.request(path)[0], 404, path)
        self.assertTrue(self.runtime.paused)
        self.assertEqual(self.runtime.world.tick, 0)

    def test_perception_selects_one_residents_own_observation(self):
        status, state = self.request("/api/state?perspective=r3")
        self.assertEqual(status, 200)
        self.assertEqual(state["perception"]["resident"], "r3")
        self.assertEqual(state["perception"]["observation"], self.runtime.world.observe("r3"))
        self.assertNotIn("origin", state["perception"]["observation"])

    def test_construction_action_reaches_world_and_rejects_malformed_target(self):
        self.request("/api/control", {"command": "pause"})
        a = self.runtime.world.residents["player"]
        a.inventory["wood"] = 2
        self.assertEqual(self.request("/api/action", {"verb": "build", "kind": "wall", "x": a.x+1, "y": a.y})[0], 200)
        self.request("/api/control", {"command": "step"})
        self.assertEqual(self.runtime.world.structures[f"{a.x+1},{a.y}"]["kind"], "wall")
        for invalid in ({"x": [], "y": 1}, {"x": True, "y": 1}, {"x": 5}, {"x": 100, "y": 100}):
            self.assertEqual(self.request("/api/action", {"verb": "build", **invalid})[0], 400)

    def test_public_demo_rejects_every_mutation(self):
        self.runtime.public_demo = True
        for path, data in (("/api/control", {"command": "step"}), ("/api/action", {"verb": "rest"})):
            self.assertEqual(self.request(path, data)[0], 403)
        state = self.request("/api/state")[1]
        self.assertTrue(state["publicDemo"])
        self.assertEqual(state["speed"], 1)
        self.assertEqual(state["tick"], 0)

    def test_heartbeat_timeout_pauses_physics(self):
        self.runtime.heartbeat_timeout = .15
        self.runtime.start()
        self.request("/api/state")
        time.sleep(.3)
        with self.runtime.lock:
            self.assertTrue(self.runtime.paused)
            tick = self.runtime.world.tick
        time.sleep(.1)
        self.assertEqual(self.runtime.world.tick, tick)
        self.assertFalse(self.request("/api/state")[1]["paused"])

    def test_action_restores_one_x_and_shutdown_autosaves(self):
        self.request("/api/control", {"command": "speed", "value": "max"})
        self.request("/api/action", {"verb": "rest"})
        self.assertEqual(self.runtime.speed, 1)
        self.runtime.close()
        self.assertEqual(json.loads((self.root / "runs/autosave.json").read_text())["tick"], 0)


if __name__ == "__main__":
    unittest.main()
