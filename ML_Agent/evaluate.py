import numpy as np
import matplotlib.pyplot as plt
import os
from GameEngine import game, agents
from ML_Agent import dqn
import main

# ── Config ────────────────────────────────────────────────────
STARTING_CHIPS   = 500
CARDS_PER_PLAYER = 4
EVAL_EPISODES    = 1000  # hands per matchup — more = more reliable numbers
MODEL_PATH       = "dqn_final.pth"


# ── Single matchup evaluator ──────────────────────────────────
def EvaluateMatchup(dqn_agent, opponents, label, episodes=EVAL_EPISODES):
    """
    Runs the DQN agent against a fixed set of opponents for N episodes.
    Returns a dict of statistics for this matchup.

    Args:
        dqn_agent:  Trained DQNAgent with training=False
        opponents:  List of opponent Agent objects
        label:      Human-readable name for this matchup (for graphs)
        episodes:   Number of hands to play
    
    Returns dict with keys:
        win_rate, avg_chip_delta, action_dist, chip_history
    """
    agentLst = opponents + [dqn_agent]

    wins         = []
    chip_deltas  = []
    chip_history = [dqn_agent.chips]
    action_counts = [0, 0, 0, 0]  # fold, check, call, bet
    roundNum = 0

    for episode in range(episodes):
        chips_before = dqn_agent.chips

        for step in main.Round(agentLst, CARDS_PER_PLAYER, roundNum):
            if step['actor'] is not dqn_agent or step['action'] is None:
                continue

            # Count actions for distribution analysis
            action = step['action']
            if action.fold:    action_counts[0] += 1
            elif action.check: action_counts[1] += 1
            elif action.call:  action_counts[2] += 1
            else:              action_counts[3] += 1

        # End of hand stats
        chip_delta = dqn_agent.chips - chips_before
        chip_deltas.append(chip_delta)
        wins.append(1 if chip_delta > 0 else 0)
        chip_history.append(dqn_agent.chips)

        # Rebuy anyone who busted
        for agent in agentLst:
            if agent.chips == 0:
                agent.AddChips(STARTING_CHIPS)

        main.Cleanup(agentLst)
        roundNum += 1

    # Normalize action counts to percentages
    total_actions = max(sum(action_counts), 1)
    action_dist = [c / total_actions for c in action_counts]

    win_rate      = np.mean(wins)
    avg_chip_delta = np.mean(chip_deltas)

    print(f"\n── {label} ──────────────────────────────")
    print(f"  Win rate:        {win_rate:.2%}")
    print(f"  Avg chips/hand:  {avg_chip_delta:+.2f}")
    print(f"  Action dist:     Fold={action_dist[0]:.1%}  "
          f"Check={action_dist[1]:.1%}  "
          f"Call={action_dist[2]:.1%}  "
          f"Bet={action_dist[3]:.1%}")

    return {
        'label':          label,
        'win_rate':       win_rate,
        'avg_chip_delta': avg_chip_delta,
        'action_dist':    action_dist,
        'chip_history':   chip_history,
        'wins':           wins,
        'chip_deltas':    chip_deltas
    }


def ResetAgents(agentLst):
    """Resets all agents to starting chips between matchups."""
    for agent in agentLst:
        agent.chips = STARTING_CHIPS
        agent.In = 0
        agent.hand.ClearHand()


# ── Plotting ──────────────────────────────────────────────────
def PlotEvaluation(results):
    """
    Generates 4 comparison plots across all matchups.
    Each result in `results` is the dict returned by EvaluateMatchup.
    """
    labels     = [r['label'] for r in results]
    win_rates  = [r['win_rate'] for r in results]
    avg_chips  = [r['avg_chip_delta'] for r in results]
    colors     = plt.cm.Set2(np.linspace(0, 1, len(results)))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("DQN Poker Bot — Evaluation Results", fontsize=16)

    # ── 1. Win rate by matchup (bar chart) ───────────────────
    ax = axes[0, 0]
    bars = ax.bar(labels, win_rates, color=colors)
    ax.axhline(1 / (len(results[0]['chip_history']) > 0 and
               sum(1 for _ in results[0]['chip_history'])),
               color='gray', linestyle='--', alpha=0.5)

    # Add a "random baseline" line — with N players, random = 1/N
    num_players = 5  # adjust to match your actual player count
    ax.axhline(1 / num_players, color='red', linestyle='--',
               label=f'Random baseline (1/{num_players})')
    ax.set_title("Win Rate by Opponent Type")
    ax.set_ylabel("Win Rate")
    ax.set_ylim(0, 1)
    ax.legend()
    for bar, rate in zip(bars, win_rates):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{rate:.1%}", ha='center', fontsize=10)

    # ── 2. Avg chips won/lost per hand (bar chart) ───────────
    ax = axes[0, 1]
    bar_colors = ['seagreen' if v >= 0 else 'tomato' for v in avg_chips]
    bars = ax.bar(labels, avg_chips, color=bar_colors)
    ax.axhline(0, color='gray', linestyle='--')
    ax.set_title("Avg Chips Won/Lost per Hand")
    ax.set_ylabel("Chips")
    for bar, val in zip(bars, avg_chips):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (1 if val >= 0 else -3),
                f"{val:+.1f}", ha='center', fontsize=10)

    # ── 3. Chip stack over time (line chart) ─────────────────
    ax = axes[1, 0]
    for result, color in zip(results, colors):
        history = result['chip_history']
        ax.plot(history, label=result['label'], color=color, alpha=0.8)
    ax.axhline(STARTING_CHIPS, color='gray', linestyle='--',
               label='Starting chips')
    ax.set_title("Chip Stack Over Time")
    ax.set_xlabel("Hand")
    ax.set_ylabel("Chips")
    ax.legend()

    # ── 4. Action distribution comparison (grouped bar) ──────
    ax = axes[1, 1]
    action_labels = ['Fold', 'Check', 'Call', 'Bet']
    x = np.arange(len(action_labels))
    width = 0.8 / len(results)

    for i, (result, color) in enumerate(zip(results, colors)):
        offset = (i - len(results) / 2) * width + width / 2
        ax.bar(x + offset, result['action_dist'],
               width=width, label=result['label'], color=color, alpha=0.85)

    ax.set_title("Action Distribution by Matchup")
    ax.set_ylabel("Fraction of Actions")
    ax.set_xticks(x)
    ax.set_xticklabels(action_labels)
    ax.legend()

    plt.tight_layout()
    plt.savefig("evaluation_results.png", dpi=150)
    plt.show()
    print("\nPlot saved to evaluation_results.png")


# ── Rolling win rate comparison ───────────────────────────────
def PlotRollingWinRates(results, window=100):
    """
    Separate plot showing rolling win rate over time for each matchup.
    Shows whether the agent is consistently good or gets lucky in bursts.
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = plt.cm.Set2(np.linspace(0, 1, len(results)))

    for result, color in zip(results, colors):
        wins = result['wins']
        rolling = [np.mean(wins[max(0, i - window):i + 1])
                   for i in range(len(wins))]
        ax.plot(rolling, label=result['label'], color=color)

    ax.axhline(1 / 5, color='red', linestyle='--',
               label='Random baseline', alpha=0.7)
    ax.set_title(f"Rolling Win Rate (window={window}) by Opponent Type")
    ax.set_xlabel("Hand")
    ax.set_ylabel("Win Rate")
    ax.legend()
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig("rolling_winrates.png", dpi=150)
    plt.show()


# ── Main evaluation entry point ───────────────────────────────
def Evaluate():
    if not os.path.exists(MODEL_PATH):
        print(f"No model found at {MODEL_PATH}. Train first with train.py")
        return

    # Load trained agent — training=False means no exploration, no weight updates
    dqn_agent = dqn.DQNAgent(STARTING_CHIPS, "DQN", training=False)
    dqn_agent.Load(MODEL_PATH)
    print(f"Loaded model from {MODEL_PATH}")
    print(f"Running {EVAL_EPISODES} hands per matchup...\n")

    results = []

    # ── Matchup 1: All random opponents ──────────────────────
    ResetAgents([dqn_agent])
    r_opponents = [agents.RAgent(STARTING_CHIPS, f"R{i}") for i in range(4)]
    results.append(EvaluateMatchup(
        dqn_agent, r_opponents,
        label="vs 4x Random"
    ))

    # ── Matchup 2: All rule-based opponents ──────────────────
    ResetAgents([dqn_agent])
    s_opponents = [agents.SAgent(STARTING_CHIPS, f"S{i}") for i in range(4)]
    results.append(EvaluateMatchup(
        dqn_agent, s_opponents,
        label="vs 4x RuleBased"
    ))

    # ── Matchup 3: Mixed opponents (realistic table) ──────────
    ResetAgents([dqn_agent])
    mixed_opponents = [
        agents.RAgent(STARTING_CHIPS, "R1"),
        agents.RAgent(STARTING_CHIPS, "R2"),
        agents.SAgent(STARTING_CHIPS, "S1"),
        agents.SAgent(STARTING_CHIPS, "S2"),
    ]
    results.append(EvaluateMatchup(
        dqn_agent, mixed_opponents,
        label="vs Mixed"
    ))

    PlotEvaluation(results)
    PlotRollingWinRates(results)

    # Print summary table
    print("\n" + "=" * 55)
    print(f"{'Matchup':<20} {'Win Rate':>10} {'Avg Chips':>12}")
    print("=" * 55)
    for r in results:
        print(f"{r['label']:<20} {r['win_rate']:>10.2%} "
              f"{r['avg_chip_delta']:>+12.2f}")
    print("=" * 55)


if __name__ == "__main__":
    Evaluate()