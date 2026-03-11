"""
evaluate.py — Round-Robin Checkpoint Tournament
=================================================
Scans checkpoints/ for every saved model and runs a round-robin tournament
to determine the Top-5.

How it works:
    - The leaderboard always holds exactly TOP_K agents.
    - When a new checkpoint is found, it joins the pool (leaderboard + challenger)
      making a group of TOP_K + 1 agents.
    - Every agent in that group plays EVAL_HANDS hands as the "focal" player
      while the others act as opponents.
    - The agent with the lowest mean chip-delta is eliminated.
    - This means early models are NOT protected by stale scores — they get
      re-evaluated against the same field as the challenger.

Leaderboard entry format:
    {
        "path":    str,    # path to .pth file, or "SAgent" / "RAgent"
        "score":   float,  # mean chip-delta from the most recent tournament
        "epsilon": float,
        "episode": str,
    }

Usage:
    python evaluate.py
"""

import os
import re
import json
import numpy as np
import torch

from GameEngine import agents
from ML_Agent import dqn
import main

# ── Config ────────────────────────────────────────────────────────────────────
CHECKPOINT_DIR   = "checkpoints"
LEADERBOARD_PATH = "leaderboard.json"
STARTING_CHIPS   = 200
CARDS_PER_PLAYER = 4
EVAL_HANDS       = 50   # hands each agent plays as focal player per tournament
TOP_K            = 5     # leaderboard size — tournament pool is always TOP_K + 1
# ─────────────────────────────────────────────────────────────────────────────


# ── Agent construction ────────────────────────────────────────────────────────

def _MakeAgent(entry: dict) -> agents.Agent:
    """Reconstructs a playable agent from a leaderboard entry dict."""
    if entry["path"] == "SAgent":
        return agents.SAgent(STARTING_CHIPS, entry["episode"])
    if entry["path"] == "RAgent":
        return agents.RAgent(STARTING_CHIPS, entry["episode"])

    agent = dqn.DQNAgent(STARTING_CHIPS, entry["episode"], training=False,
                         file=entry["path"])
    checkpoint = torch.load(entry["path"], weights_only=True)
    agent.online_net.load_state_dict(checkpoint["model_state"])
    agent.target_net.load_state_dict(agent.online_net.state_dict())
    agent.epsilon = 0.0  # always greedy during evaluation
    return agent


# ── Core evaluation ───────────────────────────────────────────────────────────

def _ScoreFocal(focal_entry: dict, opponent_entries: list) -> float:
    """
    Plays one agent (focal) for EVAL_HANDS hands against a fixed set of
    opponents. Opponents are re-instantiated each hand so chip counts reset,
    but the same set of agent types is used throughout.

    Returns the focal agent's mean chip-delta per hand.
    """
    focal       = _MakeAgent(focal_entry)
    chip_deltas = []

    for hand_num in range(EVAL_HANDS):
        # Fresh opponents each hand so chip counts reset cleanly
        opponents = [_MakeAgent(e) for e in opponent_entries]

        focal.chips  = STARTING_CHIPS
        focal.In     = 0
        chips_before = focal.chips

        agent_list = opponents + [focal]

        for _step in main.Round(agent_list, CARDS_PER_PLAYER, hand_num):
            pass  # no ML bookkeeping — all agents are frozen

        main.Cleanup(agent_list)
        chip_deltas.append(focal.chips - chips_before)

    return float(np.mean(chip_deltas))


def RunTournament(pool: list) -> list:
    """
    Round-robin: every agent in `pool` takes a turn as the focal player
    while all others act as opponents.

    Each agent's score is its mean chip-delta when it's the focal player.
    This is fair because every agent faces the exact same opposition.

    Returns the pool sorted best-first with updated scores.
    """
    num = len(pool)
    print(f"\n  Running round-robin ({num} agents x {EVAL_HANDS} hands each)...")

    for i, entry in enumerate(pool):
        opponents      = [pool[j] for j in range(num) if j != i]
        score          = _ScoreFocal(entry, opponents)
        entry["score"] = score
        print(f"    [{i+1}/{num}] {entry['episode']:>12s}  ->  {score:+.2f} chips/hand")

    pool.sort(key=lambda e: e["score"], reverse=True)
    return pool


# ── Leaderboard I/O ───────────────────────────────────────────────────────────

def _LoadLeaderboard() -> list:
    if os.path.exists(LEADERBOARD_PATH):
        with open(LEADERBOARD_PATH) as f:
            board = json.load(f)
        print(f"Loaded leaderboard ({len(board)} entries).")
        return board

    # Cold-start: seed with rule-based agents and run an initial tournament
    print("No leaderboard found — seeding with rule-based baselines...")
    baselines = [
        {"path": "SAgent", "score": 0.0, "epsilon": 0.0, "episode": "SAgent_A"},
        {"path": "SAgent", "score": 0.0, "epsilon": 0.0, "episode": "SAgent_B"},
        {"path": "SAgent", "score": 0.0, "epsilon": 0.0, "episode": "SAgent_C"},
        {"path": "RAgent", "score": 0.0, "epsilon": 0.0, "episode": "RAgent_A"},
        {"path": "RAgent", "score": 0.0, "epsilon": 0.0, "episode": "RAgent_B"},
    ]
    baselines = RunTournament(baselines)
    _SaveLeaderboard(baselines)
    return baselines


def _SaveLeaderboard(board: list):
    with open(LEADERBOARD_PATH, "w") as f:
        json.dump(board, f, indent=2)


def _PrintLeaderboard(board: list):
    print("\n+------+------------------+--------------+---------------+")
    print("|                    TOP-5 LEADERBOARD                   |")
    print("+------+------------------+--------------+---------------+")
    print("| Rank | Model            | Mean D Chips | Epsilon       |")
    print("+------+------------------+--------------+---------------+")
    for rank, entry in enumerate(board, 1):
        name  = str(entry["episode"])[:16].ljust(16)
        score = f"{entry['score']:+.2f}".rjust(12)
        eps   = f"{entry['epsilon']:.3f}".rjust(13)
        print(f"|  {rank}   | {name} | {score} | {eps} |")
    print("+------+------------------+--------------+---------------+\n")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _GetEpisodeNumber(filename: str) -> int:
    match = re.search(r"ep(\d+)", filename)
    return int(match.group(1)) if match else -1


def _AlreadyEvaluated(path: str, board: list) -> bool:
    return any(entry["path"] == path for entry in board)


# ── Main loop ─────────────────────────────────────────────────────────────────

def Run():
    """
    For each checkpoint (in episode order):
      1. Skip if it's already on the leaderboard.
      2. Build a pool of TOP_K + 1 agents: leaderboard + challenger.
      3. Run a full round-robin — everyone gets re-scored against the same field.
      4. Drop the lowest scorer. The surviving TOP_K become the new leaderboard.
      5. Save and print after every change.
    """
    leaderboard = _LoadLeaderboard()
    _PrintLeaderboard(leaderboard)

    if not os.path.isdir(CHECKPOINT_DIR):
        print(f"No '{CHECKPOINT_DIR}/' directory found — nothing to evaluate.")
        return

    files = sorted(
        [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith(".pth")],
        key=_GetEpisodeNumber
    )

    if not files:
        print("No checkpoint files found.")
        return

    print(f"Found {len(files)} checkpoint(s).\n")

    for filename in files:
        path    = os.path.join(CHECKPOINT_DIR, filename)
        episode = _GetEpisodeNumber(filename)
        label   = f"ep{episode}"

        if _AlreadyEvaluated(path, leaderboard):
            print(f"[SKIP] {label}")
            continue

        print(f"\n[CHALLENGER] {label}")
        checkpoint = torch.load(path, weights_only=True)
        epsilon    = checkpoint.get("epsilon", 0.0)
        print(f"             epsilon at save time: {epsilon:.3f}")

        challenger_entry = {
            "path":    path,
            "score":   0.0,
            "epsilon": epsilon,
            "episode": label,
        }

        # Pool = current top-K + challenger  (TOP_K + 1 agents total)
        pool = leaderboard + [challenger_entry]

        # Re-score everyone on equal footing
        pool = RunTournament(pool)

        # The bottom agent is eliminated
        eliminated = pool[-1]
        survivors  = pool[:TOP_K]

        if eliminated["episode"] == label:
            print(f"\n  -> Challenger ELIMINATED (ranked last)")
        else:
            print(f"\n  -> Challenger PROMOTED | Eliminated: {eliminated['episode']} "
                  f"({eliminated['score']:+.2f})")

        leaderboard = survivors
        _SaveLeaderboard(leaderboard)
        _PrintLeaderboard(leaderboard)

    print("=== Evaluation complete ===")
    _PrintLeaderboard(leaderboard)


if __name__ == "__main__":
    Run()