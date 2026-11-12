import threading

from idcard.store import Store


def test_add_list_clear(tmp_path):
    s = Store(tmp_path / "db" / "x.db")
    s.add(1, 0.2, "a.jpg", time="2026-01-01 10:00:00")
    s.add(2, 0.1, "b.jpg", time="2026-01-01 10:01:00")
    rows = s.list()
    assert [r["track_id"] for r in rows] == [2, 1]  # newest first
    assert sorted(s.clear()) == ["a.jpg", "b.jpg"]
    assert s.list() == []


def test_csv(tmp_path):
    s = Store(tmp_path / "x.db")
    s.add(7, 0.25, "a.jpg", time="2026-01-01 10:00:00")
    lines = s.to_csv().strip().splitlines()
    assert lines[0] == "time,track_id,score,snapshot"
    assert lines[1] == "2026-01-01 10:00:00,7,0.250,a.jpg"


def test_threads(tmp_path):
    s = Store(tmp_path / "x.db")
    threads = [threading.Thread(target=lambda i=i: [s.add(i, 0.1, f"{i}.jpg") for _ in range(20)])
               for i in range(4)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert len(s.list()) == 80
