"""Fan-out walk used by agents.scripts.check_tree_fanout."""

from agents.scripts.check_tree_fanout import _iter_fanouts


def test_iter_fanouts_max_width():
    tree = {
        "name": "root",
        "children": [
            {"name": "c1", "children": []},
            {"name": "c2", "children": []},
            {"name": "c3", "children": []},
        ],
    }
    pairs = list(_iter_fanouts(tree, []))
    assert max(n for n, _ in pairs) == 3


def test_iter_fanouts_deep_chain():
    tree = {
        "name": "a",
        "children": [{"name": "b", "children": [{"name": "c", "children": []}]}],
    }
    pairs = list(_iter_fanouts(tree, []))
    fanouts = [n for n, _ in pairs]
    paths = [tuple(p) for _, p in pairs]
    assert max(fanouts) == 1
    assert ("a", "b", "c") in paths
