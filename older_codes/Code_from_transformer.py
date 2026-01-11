"""
PSEUDOCODE (Python-style) — Talent Sourcing + Adaptive Re-Ranking (Star Feedback)

Goal:
- Given a role query (keywords) and candidate records, rank candidates by a fitness score.
- Filter out irrelevant candidates early.
- When recruiter stars a candidate, re-rank so similar candidates move up.

Data fields available:
- id (int)
- job_title (str)
- location (str)
- connections (str like "120" or "500+")
Output:
- fit ∈ [0, 1] per candidate, plus a ranked list.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import math


# ----------------------------
# Data Structures
# ----------------------------

@dataclass
class Candidate:
    id: int
    job_title: str
    location: str
    connections_raw: str  # e.g. "500+" or "120"


@dataclass
class RankedCandidate:
    candidate: Candidate
    fit: float            # base probability [0,1]
    score: float          # final ranking score (with feedback)
    debug: Dict[str, float]


# ----------------------------
# Utilities
# ----------------------------

def parse_connections(connections_raw: str) -> int:
    """Convert '500+' -> 500, '120' -> 120, empty/invalid -> 0."""
    if not connections_raw:
        return 0
    s = connections_raw.strip()
    if s.endswith("+"):
        s = s[:-1]
    try:
        return max(0, int(s))
    except ValueError:
        return 0


def log1p(x: int) -> float:
    """Stable log transform for numeric features."""
    return math.log(1.0 + max(0, x))


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def sigmoid(x: float) -> float:
    """Sigmoid to map logits -> probability."""
    # Note: this is pseudocode; real code should handle overflow carefully.
    return 1.0 / (1.0 + math.exp(-x))


# ----------------------------
# Text Builder
# ----------------------------

def build_candidate_text(c: Candidate) -> str:
    """
    Build a single text blob for embedding. In real life you would add
    skills/summary/experience etc. if available.
    """
    return f"Job title: {c.job_title}. Location: {c.location}. Connections: {c.connections_raw}."


# ----------------------------
# Embedding + Similarity (Abstract Interfaces)
# ----------------------------

class EmbeddingModel:
    """
    Abstract embedding model.
    In production: SBERT-like model that returns vectors (list[float] or numpy array).
    """
    def encode(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Cosine similarity between two vectors. Assumes same length."""
    dot = 0.0
    na = 0.0
    nb = 0.0
    for ai, bi in zip(a, b):
        dot += ai * bi
        na += ai * ai
        nb += bi * bi
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


def mean_vector(vectors: List[List[float]]) -> List[float]:
    """Average a list of vectors."""
    if not vectors:
        return []
    d = len(vectors[0])
    out = [0.0] * d
    for v in vectors:
        for i in range(d):
            out[i] += v[i]
    n = float(len(vectors))
    return [x / n for x in out]


# ----------------------------
# Fitness Scoring Head (Simple Logistic Regression / MLP placeholder)
# ----------------------------

class FitnessScorer:
    """
    Pseudocode fitness scorer.

    In reality:
    - Train this on historical labeled data (good fit vs not fit).
    - Features can include: cosine(query, candidate), log_connections, location match, etc.
    """
    def __init__(self):
        # Example weights for demonstration; you'd learn these from data.
        self.w_sim = 4.0
        self.w_conn = 0.2
        self.bias = -1.0

    def predict_fit(self, sim_qc: float, connections_int: int, location_match: float = 0.0) -> float:
        x_conn = log1p(connections_int)
        # A minimal logit model:
        logit = (self.w_sim * sim_qc) + (self.w_conn * x_conn) + (0.0 * location_match) + self.bias
        return clamp01(sigmoid(logit))


# ----------------------------
# Filtering
# ----------------------------

def passes_initial_filter(sim_qc: float, tau: float) -> bool:
    """
    Filter out clearly irrelevant candidates based on similarity threshold.
    tau is role-agnostic threshold tuned to keep high recall.
    """
    return sim_qc >= tau


# ----------------------------
# Main Ranking Engine
# ----------------------------

class TalentRanker:
    """
    Implements:
    - Initial ranking using role query similarity + numeric features
    - Feedback-driven re-ranking using starred candidate prototype
    """
    def __init__(
        self,
        embedder: EmbeddingModel,
        fitness_scorer: FitnessScorer,
        filter_tau: float = 0.35,   # example threshold; tune on validation set
        alpha: float = 0.7,         # weight for base fit
        beta: float = 0.3           # weight for prototype similarity
    ):
        self.embedder = embedder
        self.fitness_scorer = fitness_scorer
        self.filter_tau = filter_tau
        self.alpha = alpha
        self.beta = beta

        # Store starred embeddings per role query (or per "search session")
        self.starred_embs: List[List[float]] = []

    def reset_feedback(self) -> None:
        """Call when starting a new role/search session."""
        self.starred_embs = []

    def star_candidate(self, candidate_embedding: List[float]) -> None:
        """Recruiter stars a candidate; we store its embedding for prototype building."""
        self.starred_embs.append(candidate_embedding)

    def _prototype(self) -> Optional[List[float]]:
        """Mean embedding of starred candidates."""
        if not self.starred_embs:
            return None
        return mean_vector(self.starred_embs)

    def rank(self, role_query: str, candidates: List[Candidate]) -> List[RankedCandidate]:
        """
        Produce ranked candidates with:
        - Base fit probability
        - Feedback-aware final score (if stars exist)
        """

        # 1) Build texts
        role_text = role_query.strip()
        cand_texts = [build_candidate_text(c) for c in candidates]

        # 2) Embed role + candidates
        q_emb = self.embedder.encode([role_text])[0]      # [d]
        c_embs = self.embedder.encode(cand_texts)         # [N, d]

        # 3) Compute similarities and filter
        proto = self._prototype()                         # [d] or None
        ranked: List[RankedCandidate] = []

        for c, c_emb in zip(candidates, c_embs):
            sim_qc = cosine_similarity(q_emb, c_emb)

            # (A) Filter out irrelevant candidates early
            if not passes_initial_filter(sim_qc, tau=self.filter_tau):
                continue

            # (B) Compute base fit probability
            conn_int = parse_connections(c.connections_raw)
            base_fit = self.fitness_scorer.predict_fit(sim_qc=sim_qc, connections_int=conn_int)

            # (C) If we have starred candidates, add prototype similarity for adaptive reranking
            sim_pc = cosine_similarity(proto, c_emb) if proto is not None else 0.0

            # Final score used for sorting:
            # - base_fit is already a probability
            # - sim_pc is in [-1, 1], so we map it to [0, 1] by (sim_pc+1)/2 for stability
            sim_pc_01 = (sim_pc + 1.0) / 2.0

            final_score = (self.alpha * base_fit) + (self.beta * sim_pc_01)

            ranked.append(
                RankedCandidate(
                    candidate=c,
                    fit=base_fit,
                    score=final_score,
                    debug={
                        "sim_qc": sim_qc,
                        "sim_pc": sim_pc,
                        "sim_pc_01": sim_pc_01,
                        "connections": float(conn_int),
                    }
                )
            )

        # 4) Sort descending by final score
        ranked.sort(key=lambda r: r.score, reverse=True)
        return ranked


# ----------------------------
# Training Loop (High-level Pseudocode)
# ----------------------------

def train_fitness_scorer_pseudocode(training_examples: List[Dict]) -> FitnessScorer:
    """
    Pseudocode for training the scorer on labeled data.

    Each example contains:
    - role_query
    - candidate fields
    - label (0/1) OR fit score in [0,1]
    """
    # In reality you would:
    # 1) embed role + candidate
    # 2) compute sim_qc
    # 3) build feature vector: [sim_qc, log_connections, location_match, ...]
    # 4) train logistic regression or small MLP with BCE / MSE
    scorer = FitnessScorer()

    # Placeholder: show *where* training happens
    # for epoch in range(E):
    #   for ex in training_examples:
    #       features = build_features(ex)
    #       pred = scorer(features)
    #       loss = loss_fn(pred, label)
    #       update weights
    return scorer


# ----------------------------
# Example Usage Flow (Role session)
# ----------------------------

def example_flow(role_query: str, candidates: List[Candidate], embedder: EmbeddingModel):
    scorer = FitnessScorer()
    ranker = TalentRanker(embedder=embedder, fitness_scorer=scorer)

    # 1) Initial ranking (no stars)
    ranker.reset_feedback()
    ranked_0 = ranker.rank(role_query, candidates)

    # 2) Recruiter stars candidate #7 (index 6) from the currently ranked list
    if len(ranked_0) >= 7:
        starred = ranked_0[6].candidate

        # Need the embedding of the starred candidate (re-embed or cache embeddings in production)
        starred_text = build_candidate_text(starred)
        starred_emb = embedder.encode([starred_text])[0]

        ranker.star_candidate(starred_emb)

    # 3) Re-rank with feedback
    ranked_1 = ranker.rank(role_query, candidates)

    return ranked_0, ranked_1
