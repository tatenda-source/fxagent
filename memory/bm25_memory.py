"""BM25-based memory store for trading reflections.

Uses lexical matching (BM25) instead of embeddings — no API cost, fully offline.
Stores past trade outcomes and lessons learned, retrieves relevant ones for
future decisions on the same or similar instruments.

Inspired by TradingAgents' pragmatic approach to agent memory.
"""

import json
import os
import logging
from typing import List, Dict, Optional, Any
from pathlib import Path

logger = logging.getLogger(__name__)


class TradingMemory:
    """Persistent memory store for trade reflections and lessons learned.

    Uses BM25 for retrieval when rank_bm25 is available, falls back to
    keyword matching otherwise. Stores memories as JSON on disk.
    """

    def __init__(self, memory_dir: str = None):
        if memory_dir is None:
            memory_dir = str(Path(__file__).parent / "trade_memories")
        self.memory_dir = memory_dir
        os.makedirs(self.memory_dir, exist_ok=True)
        self._memories: List[Dict[str, Any]] = []
        self._load_all()

    def _load_all(self):
        """Load all memories from disk."""
        self._memories = []
        memory_file = os.path.join(self.memory_dir, "memories.json")
        if os.path.exists(memory_file):
            try:
                with open(memory_file, "r") as f:
                    self._memories = json.load(f)
                logger.info(f"Loaded {len(self._memories)} memories from disk")
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to load memories: {e}")
                self._memories = []

    def _save_all(self):
        """Persist all memories to disk."""
        memory_file = os.path.join(self.memory_dir, "memories.json")
        try:
            with open(memory_file, "w") as f:
                json.dump(self._memories, f, indent=2, default=str)
        except IOError as e:
            logger.error(f"Failed to save memories: {e}")

    def add_memory(self, pair: str, situation: str, outcome: str,
                   lessons: str, metadata: Optional[Dict] = None):
        """Store a new trade reflection.

        Args:
            pair: The instrument (e.g., "GBPUSD=X")
            situation: Description of the trade setup and market conditions
            outcome: What actually happened (TP_HIT, SL_HIT, pnl)
            lessons: What was learned from this trade
            metadata: Optional extra data (regime, confidence, etc.)
        """
        memory = {
            "pair": pair,
            "situation": situation,
            "outcome": outcome,
            "lessons": lessons,
            "metadata": metadata or {},
            "timestamp": str(metadata.get("timestamp", "")) if metadata else "",
        }
        self._memories.append(memory)
        self._save_all()
        logger.info(f"Stored memory for {pair}: {lessons[:80]}...")

    def retrieve(self, query: str, pair: Optional[str] = None,
                 top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve relevant memories using BM25 or keyword matching.

        Args:
            query: Search query (e.g., current market situation description)
            pair: Optional filter to same instrument
            top_k: Number of results to return
        """
        candidates = self._memories
        if pair:
            candidates = [m for m in candidates if m["pair"] == pair]

        if not candidates:
            return []

        # Try BM25 first
        try:
            return self._bm25_retrieve(query, candidates, top_k)
        except ImportError:
            return self._keyword_retrieve(query, candidates, top_k)

    def _bm25_retrieve(self, query: str, candidates: List[Dict],
                       top_k: int) -> List[Dict]:
        """BM25 retrieval using rank_bm25 library."""
        from rank_bm25 import BM25Okapi

        corpus = [
            f"{m['situation']} {m['outcome']} {m['lessons']}"
            for m in candidates
        ]
        tokenized_corpus = [doc.lower().split() for doc in corpus]
        tokenized_query = query.lower().split()

        bm25 = BM25Okapi(tokenized_corpus)
        scores = bm25.get_scores(tokenized_query)

        # Sort by score descending
        scored = sorted(
            zip(candidates, scores),
            key=lambda x: x[1],
            reverse=True,
        )

        return [m for m, s in scored[:top_k] if s > 0]

    def _keyword_retrieve(self, query: str, candidates: List[Dict],
                          top_k: int) -> List[Dict]:
        """Fallback: simple keyword matching."""
        query_words = set(query.lower().split())

        scored = []
        for m in candidates:
            text = f"{m['situation']} {m['outcome']} {m['lessons']}".lower()
            text_words = set(text.split())
            overlap = len(query_words & text_words)
            if overlap > 0:
                scored.append((m, overlap))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [m for m, _ in scored[:top_k]]

    def get_lessons_for_pair(self, pair: str, limit: int = 5) -> List[str]:
        """Get recent lessons for a specific pair."""
        pair_memories = [m for m in self._memories if m["pair"] == pair]
        pair_memories.sort(key=lambda m: m.get("timestamp", ""), reverse=True)
        return [m["lessons"] for m in pair_memories[:limit]]

    def get_all_lessons(self, limit: int = 20) -> List[str]:
        """Get most recent lessons across all pairs."""
        sorted_memories = sorted(
            self._memories,
            key=lambda m: m.get("timestamp", ""),
            reverse=True,
        )
        return [m["lessons"] for m in sorted_memories[:limit]]

    def clear_pair(self, pair: str):
        """Remove all memories for a specific pair."""
        self._memories = [m for m in self._memories if m["pair"] != pair]
        self._save_all()

    def __len__(self):
        return len(self._memories)
