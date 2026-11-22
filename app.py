"""
AI-Based Knowledge Learning Graph for Data Structures & Algorithms
===================================================================
Run:   python app.py        (then the browser opens at http://127.0.0.1:5000)

Architecture (3 small layers, each easy to explain in a viva):
    1. KnowledgeGraph  -> loads graph.json, validates it is a DAG, computes levels
    2. ProgressStore   -> reads/writes progress.json (mastery per topic)
    3. Recommender     -> rule-based "AI": picks the next best topic, adapts to quiz scores
    + Flask routes     -> thin REST API consumed by templates/index.html (Vis.js)
"""

import json
import os
import threading
import webbrowser
from collections import deque

from flask import Flask, jsonify, render_template, request

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
GRAPH_FILE = os.path.join(DATA_DIR, "graph.json")
QUIZ_FILE = os.path.join(DATA_DIR, "quizzes.json")
PROGRESS_FILE = os.path.join(DATA_DIR, "progress.json")

MASTERY_THRESHOLD = 0.70   # mastery needed to count a topic as "Mastered"
WEAK_THRESHOLD = 0.50      # last quiz score below this = student is struggling
REVIEW_THRESHOLD = 0.85    # mastered topics below this are suggested for revision
OLD_WEIGHT, NEW_WEIGHT = 0.4, 0.6   # mastery = 0.4*old + 0.6*latest score


# --------------------------------------------------------------------------
# Layer 1: Knowledge graph (structure only, no user data)
# --------------------------------------------------------------------------
class KnowledgeGraph:
    """Directed acyclic graph: an edge A -> B means 'learn A before B'."""

    def __init__(self, path):
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        self.nodes = {n["id"]: n for n in raw["nodes"]}
        self.prereqs = {nid: list(n["prereqs"]) for nid, n in self.nodes.items()}
        self.children = {nid: [] for nid in self.nodes}
        for nid, pres in self.prereqs.items():
            for p in pres:
                if p not in self.nodes:
                    raise ValueError(f"Unknown prerequisite '{p}' in node '{nid}'")
                self.children[p].append(nid)
        self.topo_order = self._topological_sort()
        self.levels = self._compute_levels()
        self._descendants = {nid: self._reachable(nid, self.children) for nid in self.nodes}
        self._ancestors = {nid: self._reachable(nid, self.prereqs) for nid in self.nodes}

    def _topological_sort(self):
        """Kahn's algorithm. Also proves the graph has no cycles."""
        indeg = {nid: len(p) for nid, p in self.prereqs.items()}
        queue = deque(nid for nid, d in indeg.items() if d == 0)
        order = []
        while queue:
            cur = queue.popleft()
            order.append(cur)
            for child in self.children[cur]:
                indeg[child] -= 1
                if indeg[child] == 0:
                    queue.append(child)
        if len(order) != len(self.nodes):
            raise ValueError("graph.json contains a cycle; prerequisites must form a DAG")
        return order

    def _compute_levels(self):
        """Level = longest prerequisite chain above a node (used for the layout)."""
        levels = {}
        for nid in self.topo_order:
            pres = self.prereqs[nid]
            levels[nid] = 0 if not pres else 1 + max(levels[p] for p in pres)
        return levels

    @staticmethod
    def _reachable(start, adjacency):
        """All nodes reachable from `start` by DFS (excluding start)."""
        seen, stack = set(), list(adjacency[start])
        while stack:
            cur = stack.pop()
            if cur not in seen:
                seen.add(cur)
                stack.extend(adjacency[cur])
        return seen

    def descendants(self, nid):
        return self._descendants[nid]

    def ancestors(self, nid):
        return self._ancestors[nid]


# --------------------------------------------------------------------------
# Layer 2: Progress storage (JSON file)
# --------------------------------------------------------------------------
class ProgressStore:
    """Stores {topic_id: {mastery, attempts, last_score, history}} in progress.json."""

    def __init__(self, path, node_ids):
        self.path = path
        self.node_ids = list(node_ids)
        self.data = self._load()

    @staticmethod
    def _blank():
        return {"mastery": 0.0, "attempts": 0, "last_score": None, "history": []}

    def _load(self):
        data = {}
        if os.path.exists(self.path):
            try:
                with open(self.path, encoding="utf-8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError):
                data = {}
        for nid in self.node_ids:          # make sure every topic has an entry
            data.setdefault(nid, self._blank())
        return data

    def save(self):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2)

    def get(self, nid):
        return self.data[nid]

    def record_quiz(self, nid, score):
        """Update mastery with a weighted moving average of quiz scores."""
        p = self.data[nid]
        if p["attempts"] == 0:
            p["mastery"] = score
        else:
            p["mastery"] = OLD_WEIGHT * p["mastery"] + NEW_WEIGHT * score
        p["mastery"] = round(p["mastery"], 2)
        p["attempts"] += 1
        p["last_score"] = round(score, 2)
        p["history"] = (p["history"] + [round(score, 2)])[-5:]
        self.save()

    def reset(self):
        self.data = {nid: self._blank() for nid in self.node_ids}
        self.save()


# --------------------------------------------------------------------------
# Layer 3: Learning engine (statuses + recommendations + paths)
# --------------------------------------------------------------------------
class LearningEngine:
    def __init__(self, graph, store):
        self.g = graph
        self.store = store

    # ---- status of a node: mastered / in_progress / locked ----
    def is_mastered(self, nid):
        p = self.store.get(nid)
        return p["attempts"] > 0 and p["mastery"] >= MASTERY_THRESHOLD

    def statuses(self):
        result = {}
        for nid in self.g.topo_order:
            if self.is_mastered(nid):
                result[nid] = "mastered"
            elif all(self.is_mastered(p) for p in self.g.prereqs[nid]):
                result[nid] = "in_progress"   # unlocked, not yet mastered
            else:
                result[nid] = "locked"
        return result

    # ---- AI recommendation (transparent scoring rules) ----
    def recommend(self, limit=3):
        status = self.statuses()
        best = {}   # topic id -> recommendation dict (keep the highest score)

        def offer(nid, kind, score, reason):
            if nid not in best or score > best[nid]["score"]:
                best[nid] = {"id": nid, "label": self.g.nodes[nid]["label"],
                             "type": kind, "score": round(score, 1), "reason": reason}

        for nid in self.g.topo_order:
            node, prog = self.g.nodes[nid], self.store.get(nid)

            if status[nid] == "in_progress":
                unlocks = len(self.g.descendants(nid))
                base = 10 + 1.5 * unlocks - 1.0 * node["difficulty"]

                if prog["attempts"] == 0:
                    offer(nid, "learn", base,
                          f"Prerequisites done. Mastering it opens up {unlocks} more topic(s).")
                elif prog["last_score"] < WEAK_THRESHOLD:
                    # Adaptive rule: struggling -> revise the weakest prerequisite first
                    pres = self.g.prereqs[nid]
                    weakest = min(pres, key=lambda p: self.store.get(p)["mastery"]) if pres else None
                    if weakest and self.store.get(weakest)["mastery"] < 0.9:
                        pct = int(self.store.get(weakest)["mastery"] * 100)
                        offer(weakest, "revise", 100,
                              f"You scored {int(prog['last_score'] * 100)}% on {node['label']}. "
                              f"Your weakest prerequisite is this one ({pct}% mastery).")
                    else:
                        offer(nid, "retry", 60,
                              f"You scored {int(prog['last_score'] * 100)}%. Review the notes and retry.")
                else:
                    offer(nid, "retry", base + 5,
                          f"Close: {int(prog['mastery'] * 100)}% mastery, "
                          f"need {int(MASTERY_THRESHOLD * 100)}%. One more attempt should do it.")

            elif status[nid] == "mastered" and prog["mastery"] < REVIEW_THRESHOLD:
                # Spaced-review style rule: shaky mastery on a foundation topic
                offer(nid, "revise", 8 + (REVIEW_THRESHOLD - prog["mastery"]) * 20,
                      f"Mastery is {int(prog['mastery'] * 100)}%. A quick revision keeps your foundation solid.")

        ranked = sorted(best.values(), key=lambda r: r["score"], reverse=True)
        return ranked[:limit]

    # ---- personalised learning path to a target topic ----
    def path_to(self, target):
        needed = (self.g.ancestors(target) | {target})
        todo = [n for n in needed if not self.is_mastered(n)]
        todo.sort(key=lambda n: self.g.topo_order.index(n))
        return [{"id": n, "label": self.g.nodes[n]["label"],
                 "hours": self.g.nodes[n]["hours"]} for n in todo]

    def stats(self):
        st = self.statuses()
        total = len(st)
        mastered = sum(1 for s in st.values() if s == "mastered")
        return {"total": total, "mastered": mastered,
                "in_progress": sum(1 for s in st.values() if s == "in_progress"),
                "locked": sum(1 for s in st.values() if s == "locked"),
                "percent": round(100 * mastered / total)}


# --------------------------------------------------------------------------
# Flask app + REST API
# --------------------------------------------------------------------------
app = Flask(__name__)

graph = KnowledgeGraph(GRAPH_FILE)
with open(QUIZ_FILE, encoding="utf-8") as _f:
    QUIZZES = json.load(_f)
store = ProgressStore(PROGRESS_FILE, graph.nodes.keys())
engine = LearningEngine(graph, store)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/graph")
def api_graph():
    status = engine.statuses()
    nodes = []
    for nid in graph.topo_order:
        n, p = graph.nodes[nid], store.get(nid)
        nodes.append({
            "id": nid, "label": n["label"], "level": graph.levels[nid],
            "difficulty": n["difficulty"], "hours": n["hours"],
            "description": n["description"], "prereqs": n["prereqs"],
            "status": status[nid], "mastery": p["mastery"],
            "attempts": p["attempts"], "last_score": p["last_score"],
            "has_quiz": nid in QUIZZES,
        })
    edges = [{"from": p, "to": nid} for nid in graph.nodes for p in graph.prereqs[nid]]
    return jsonify({"nodes": nodes, "edges": edges, "stats": engine.stats()})


@app.route("/api/recommendations")
def api_recommendations():
    recs = engine.recommend(limit=3)
    message = None
    if not recs and engine.stats()["mastered"] == engine.stats()["total"]:
        message = "You have mastered every topic. Excellent work!"
    return jsonify({"recommendations": recs, "message": message})


@app.route("/api/path/<nid>")
def api_path(nid):
    if nid not in graph.nodes:
        return jsonify({"error": "unknown topic"}), 404
    path = engine.path_to(nid)
    return jsonify({"target": nid, "steps": path,
                    "total_hours": sum(s["hours"] for s in path)})


@app.route("/api/quiz/<nid>")
def api_quiz(nid):
    if nid not in graph.nodes or nid not in QUIZZES:
        return jsonify({"error": "no quiz for this topic"}), 404
    if engine.statuses()[nid] == "locked":
        return jsonify({"error": "Topic is locked. Master its prerequisites first."}), 403
    questions = [{"q": q["q"], "options": q["options"]} for q in QUIZZES[nid]]
    return jsonify({"id": nid, "label": graph.nodes[nid]["label"], "questions": questions})


@app.route("/api/quiz/<nid>", methods=["POST"])
def api_submit_quiz(nid):
    if nid not in graph.nodes or nid not in QUIZZES:
        return jsonify({"error": "no quiz for this topic"}), 404
    before = engine.statuses()
    if before[nid] == "locked":
        return jsonify({"error": "Topic is locked."}), 403

    answers = (request.get_json(silent=True) or {}).get("answers", [])
    questions = QUIZZES[nid]
    if len(answers) != len(questions):
        return jsonify({"error": "Please answer every question."}), 400

    review, correct = [], 0
    for q, given in zip(questions, answers):
        ok = (given == q["answer"])
        correct += ok
        review.append({"correct": ok, "right_answer": q["options"][q["answer"]],
                       "explain": q["explain"]})
    score = correct / len(questions)
    store.record_quiz(nid, score)

    after = engine.statuses()
    newly_unlocked = [graph.nodes[n]["label"] for n in graph.nodes
                      if before[n] == "locked" and after[n] != "locked"]
    return jsonify({"score": score, "correct": correct, "total": len(questions),
                    "mastery": store.get(nid)["mastery"], "status": after[nid],
                    "newly_unlocked": newly_unlocked, "review": review})


@app.route("/api/reset", methods=["POST"])
def api_reset():
    store.reset()
    return jsonify({"ok": True})


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
if __name__ == "__main__":
    url = "http://127.0.0.1:5000"
    print(f"\n  DSA Knowledge Graph running at {url}\n  Press CTRL+C to stop.\n")
    threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    app.run(host="127.0.0.1", port=5000, debug=False)
