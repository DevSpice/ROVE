import copy
import itertools
import numpy as np
import matplotlib.pyplot as plt
import os
from GameEngine import game, agents
from ML_Agent import dqn
import main

# ── Config ────────────────────────────────────────────────────
STARTING_CHIPS    = 500
CARDS_PER_PLAYER  = 4
EVAL_EPISODES     = 1000   # hands per matchup — more = more reliable numbers
NUM_PLAYERS       = 5      # total players at the table (opponents + DQN agent)
TOURNAMENT_HANDS  = 200    # hands per head-to-head matchup in the tournament

# ── ADD YOUR MODELS HERE ──────────────────────────────────────
# Each entry is a (file_path, display_label) tuple.
# Just append a new line to compare another checkpoint or architecture.
#
# Example:
#   ("models/v1.pt",       "v1 baseline"),
#   ("models/v2.pt",       "v2 deeper net"),
#   ("models/v3_best.pt",  "v3 best"),
#

MODELS = [
    ("/Users/trenthoule/Desktop/School/GenAI/Final Project/ROVE/BestModels/best_gen4.pth", "gen1"),
    ("/Users/trenthoule/Desktop/School/GenAI/Final Project/ROVE/evolution/best_gen9.pth", "gen9"),
    ("/Users/trenthoule/Desktop/School/GenAI/Final Project/ROVE/evolution/best_gen38.pth", "gen38"),
    ("/Users/trenthoule/Desktop/School/GenAI/Final Project/ROVE/evolution/best_gen49.pth", "LatestGen"),
    ("/Users/trenthoule/Desktop/School/GenAI/Final Project/ROVE/BestModels/6amIteration.pth", "Early Model"),
    ("/Users/trenthoule/Desktop/School/GenAI/Final Project/ROVE/checkpoints/dqn_ep957000.pth", "Best_In_Tournament"),
    # ("/Users/trenthoule/Desktop/School/GenAI/Final Project/ROVE/BestModels/betCheck.pth", "Bet+Check"),
    ("/Users/trenthoule/Desktop/School/GenAI/Final Project/ROVE/dqn_final.pth", "Main Model")
]

# If MODELS is empty above, fall back to the interactive prompt
if not MODELS:
    _path  = input("Path to model: ").strip()
    _label = input("Label for this model (e.g. 'v1 baseline'): ").strip() or "DQN"
    MODELS = [(_path, _label)]


# ── Helpers ───────────────────────────────────────────────────
def _build_matchups(starting_chips):
    """
    Returns a list of (label, opponents_factory) pairs.
    The factory is a callable so fresh opponent objects are created
    for every model — agents are stateful and must not be shared.
    """
    return [
        (
            "vs 4× Random",
            lambda: [agents.RAgent(starting_chips, f"R{i}") for i in range(4)],
        ),
        (
            "vs 4× RuleBased",
            lambda: [agents.SAgent(starting_chips, f"S{i}") for i in range(4)],
        ),
        (
            "vs Mixed",
            lambda: [
                agents.RAgent(starting_chips, "R1"),
                agents.RAgent(starting_chips, "R2"),
                agents.SAgent(starting_chips, "S1"),
                agents.SAgent(starting_chips, "S2"),
            ],
        ),
    ]


def _reset_agents(agent_list):
    """Resets every agent to starting chips between matchups."""
    for agent in agent_list:
        agent.chips = STARTING_CHIPS
        agent.In    = 0
        agent.hand.ClearHand()


# ── Single matchup evaluator ──────────────────────────────────
def EvaluateMatchup(dqn_agent, opponents, label, episodes=EVAL_EPISODES):
    """
    Runs dqn_agent against a fixed set of opponents for N episodes.

    Args:
        dqn_agent:  Trained DQNAgent (training=False)
        opponents:  List of opponent Agent objects
        label:      Human-readable matchup name
        episodes:   Number of hands to play

    Returns a dict with keys:
        label, win_rate, avg_chip_delta, action_dist, chip_history,
        wins, chip_deltas
    """
    agent_list    = opponents + [dqn_agent]
    wins          = []
    chip_deltas   = []
    chip_history  = [dqn_agent.chips]
    action_counts = [0, 0, 0, 0]   # fold, check, call, bet

    for episode in range(episodes):
        chips_before = dqn_agent.chips

        for step in main.Round(agent_list, CARDS_PER_PLAYER, episode):
            if step['actor'] is not dqn_agent or step['action'] is None:
                continue
            action = step['action']
            if   action.fold:  action_counts[0] += 1
            elif action.check: action_counts[1] += 1
            elif action.call:  action_counts[2] += 1
            else:              action_counts[3] += 1

        chip_delta = dqn_agent.chips - chips_before
        chip_deltas.append(chip_delta)
        wins.append(1 if chip_delta > 0 else 0)
        chip_history.append(dqn_agent.chips)

        for agent in agent_list:
            if agent.chips == 0:
                agent.AddChips(STARTING_CHIPS)

        main.Cleanup(agent_list)

    total_actions = max(sum(action_counts), 1)
    action_dist   = [c / total_actions for c in action_counts]
    win_rate      = np.mean(wins)
    avg_chip_delta = np.mean(chip_deltas)

    print(f"    Win rate: {win_rate:.2%}  |  "
          f"Avg chips/hand: {avg_chip_delta:+.2f}  |  "
          f"Fold={action_dist[0]:.1%} Check={action_dist[1]:.1%} "
          f"Call={action_dist[2]:.1%} Bet={action_dist[3]:.1%}")

    return {
        'label':           label,
        'win_rate':        win_rate,
        'avg_chip_delta':  avg_chip_delta,
        'action_dist':     action_dist,
        'chip_history':    chip_history,
        'wins':            wins,
        'chip_deltas':     chip_deltas,
    }


# ── Head-to-head evaluator ────────────────────────────────────
def EvaluateHeadToHead(dqn_agents, episodes=EVAL_EPISODES):
    """
    Runs all DQN agents against each other at the same table simultaneously.
    Every agent's stats are tracked independently within the same hands.

    If there are fewer models than NUM_PLAYERS, the remaining seats are filled
    with Random agents as neutral filler (they don't appear in the results).

    Args:
        dqn_agents: list of loaded DQNAgent objects, one per model
        episodes:   number of hands to play

    Returns:
        dict  { model_label -> stats_dict }
        where stats_dict has the same keys as EvaluateMatchup returns,
        plus 'rank' (1 = best avg chip delta).
    """
    # Fill empty seats with neutral random agents so table size stays consistent
    n_fillers = max(0, NUM_PLAYERS - len(dqn_agents))
    fillers   = [agents.RAgent(STARTING_CHIPS, f"Filler{i}") for i in range(n_fillers)]
    agent_list = dqn_agents + fillers

    # Per-agent tracking, keyed by the agent object itself
    tracking = {
        agent: {
            'wins':          [],
            'chip_deltas':   [],
            'chip_history':  [agent.chips],
            'action_counts': [0, 0, 0, 0],
        }
        for agent in dqn_agents
    }

    for episode in range(episodes):
        chips_before = {agent: agent.chips for agent in dqn_agents}

        for step in main.Round(agent_list, CARDS_PER_PLAYER, episode):
            actor  = step['actor']
            action = step['action']
            if actor not in tracking or action is None:
                continue
            ac = tracking[actor]['action_counts']
            if   action.fold:  ac[0] += 1
            elif action.check: ac[1] += 1
            elif action.call:  ac[2] += 1
            else:              ac[3] += 1

        for agent in dqn_agents:
            delta = agent.chips - chips_before[agent]
            tracking[agent]['chip_deltas'].append(delta)
            tracking[agent]['wins'].append(1 if delta > 0 else 0)
            tracking[agent]['chip_history'].append(agent.chips)

        for agent in agent_list:
            if agent.chips == 0:
                agent.AddChips(STARTING_CHIPS)

        main.Cleanup(agent_list)

    # Build result dicts — same schema as EvaluateMatchup for consistency
    h2h_results = {}
    print(f"\n── Head-to-Head ({'  vs  '.join(a.name for a in dqn_agents)}) ──")
    for agent in dqn_agents:
        t             = tracking[agent]
        total_actions = max(sum(t['action_counts']), 1)
        action_dist   = [c / total_actions for c in t['action_counts']]
        win_rate      = np.mean(t['wins'])
        avg_chip_delta = np.mean(t['chip_deltas'])

        print(f"  {agent.name:<20}  WR={win_rate:.2%}  "
              f"Avg chips/hand={avg_chip_delta:+.2f}  "
              f"Fold={action_dist[0]:.1%} Check={action_dist[1]:.1%} "
              f"Call={action_dist[2]:.1%} Bet={action_dist[3]:.1%}")

        h2h_results[agent.name] = {
            'label':          agent.name,
            'win_rate':       win_rate,
            'avg_chip_delta': avg_chip_delta,
            'action_dist':    action_dist,
            'chip_history':   t['chip_history'],
            'wins':           t['wins'],
            'chip_deltas':    t['chip_deltas'],
        }

    # Attach rank by avg chip delta (1 = highest earner)
    sorted_labels = sorted(h2h_results, key=lambda k: h2h_results[k]['avg_chip_delta'], reverse=True)
    for rank, lbl in enumerate(sorted_labels, start=1):
        h2h_results[lbl]['rank'] = rank

    return h2h_results


# ── Head-to-head plot ─────────────────────────────────────────
def PlotHeadToHead(h2h_results, window=100):
    """
    Three-panel figure showing head-to-head results:
      Left:   Win rate & avg chip delta (grouped bars, one group per model)
      Middle: Chip stack over time (one line per model)
      Right:  Rolling win rate over time (one line per model)

    Args:
        h2h_results: dict returned by EvaluateHeadToHead
        window:      rolling average window size
    """
    if len(h2h_results) < 2:
        print("Head-to-head plot skipped — need at least 2 models.")
        return

    model_labels = list(h2h_results.keys())
    colors       = plt.cm.Set2(np.linspace(0, 1, len(model_labels)))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Head-to-Head: Models vs Each Other", fontsize=15, fontweight='bold')

    # ── Left: win rate + avg chip delta side-by-side bars ────
    ax   = axes[0]
    x    = np.arange(len(model_labels))
    w    = 0.35

    win_rates  = [h2h_results[l]['win_rate']       for l in model_labels]
    avg_chips  = [h2h_results[l]['avg_chip_delta']  for l in model_labels]

    bars_wr = ax.bar(x - w / 2, win_rates, width=w, color=colors)
                     
def PlotEvaluation(all_results):
    """
    Generates 4 comparison plots.

    Args:
        all_results: dict  { model_label -> { matchup_label -> stats_dict } }
    """
    model_labels   = list(all_results.keys())
    matchup_labels = list(next(iter(all_results.values())).keys())
    colors         = plt.cm.Set2(np.linspace(0, 1, len(model_labels)))

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle("DQN Poker Bot — Multi-Model Evaluation", fontsize=16, fontweight='bold')

    x     = np.arange(len(matchup_labels))
    width = 0.8 / len(model_labels)   # bars shrink as more models are added

    # ── 1. Win rate — grouped bar per matchup ────────────────
    ax = axes[0, 0]
    for i, (model_label, color) in enumerate(zip(model_labels, colors)):
        offsets   = x + (i - len(model_labels) / 2 + 0.5) * width
        win_rates = [all_results[model_label][m]['win_rate'] for m in matchup_labels]
        bars = ax.bar(offsets, win_rates, width=width,
                      label=model_label, color=color, alpha=0.88)
        for bar, rate in zip(bars, win_rates):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{rate:.0%}", ha='center', fontsize=8)

    ax.axhline(1 / NUM_PLAYERS, color='red', linestyle='--',
               alpha=0.6, label=f'Random baseline (1/{NUM_PLAYERS})')
    ax.set_title("Win Rate by Matchup")
    ax.set_ylabel("Win Rate")
    ax.set_ylim(0, 1)
    ax.set_xticks(x)
    ax.set_xticklabels(matchup_labels, fontsize=9)
    ax.legend(fontsize=8)

    # ── 2. Avg chips won/lost — grouped bar per matchup ──────
    ax = axes[0, 1]
    for i, (model_label, color) in enumerate(zip(model_labels, colors)):
        offsets    = x + (i - len(model_labels) / 2 + 0.5) * width
        avg_chips  = [all_results[model_label][m]['avg_chip_delta'] for m in matchup_labels]
        bar_colors = ['seagreen' if v >= 0 else 'tomato' for v in avg_chips]
        bars = ax.bar(offsets, avg_chips, width=width,
                      label=model_label, color=bar_colors, alpha=0.88)
        for bar, val in zip(bars, avg_chips):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + (0.5 if val >= 0 else -2.5),
                    f"{val:+.1f}", ha='center', fontsize=8)

    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_title("Avg Chips Won/Lost per Hand")
    ax.set_ylabel("Chips")
    ax.set_xticks(x)
    ax.set_xticklabels(matchup_labels, fontsize=9)
    ax.legend(fontsize=8)

    # ── 3. Chip stack over time — one line per (model, matchup) ─
    ax = axes[1, 0]
    line_styles = ['-', '--', ':']   # cycle if many matchups
    for i, (model_label, color) in enumerate(zip(model_labels, colors)):
        for j, matchup_label in enumerate(matchup_labels):
            history = all_results[model_label][matchup_label]['chip_history']
            ax.plot(history,
                    label=f"{model_label} | {matchup_label}",
                    color=color,
                    linestyle=line_styles[j % len(line_styles)],
                    alpha=0.75)

    ax.axhline(STARTING_CHIPS, color='gray', linestyle='--',
               label='Starting chips', alpha=0.5)
    ax.set_title("Chip Stack Over Time")
    ax.set_xlabel("Hand")
    ax.set_ylabel("Chips")
    ax.legend(fontsize=7, ncol=2)

    # ── 4. Action distribution — grouped bar per action type ─
    ax = axes[1, 1]
    action_labels = ['Fold', 'Check', 'Call', 'Bet']
    xa = np.arange(len(action_labels))

    # Average action dist across all matchups for each model
    for i, (model_label, color) in enumerate(zip(model_labels, colors)):
        all_dists = [all_results[model_label][m]['action_dist'] for m in matchup_labels]
        avg_dist  = np.mean(all_dists, axis=0)
        offsets   = xa + (i - len(model_labels) / 2 + 0.5) * width
        ax.bar(offsets, avg_dist, width=width,
               label=model_label, color=color, alpha=0.88)

    ax.set_title("Action Distribution (averaged across matchups)")
    ax.set_ylabel("Fraction of Actions")
    ax.set_xticks(xa)
    ax.set_xticklabels(action_labels)
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("evaluation_results.png", dpi=150)
    plt.show()
    print("\nPlot saved → evaluation_results.png")


def PlotRollingWinRates(all_results, window=100):
    """
    One subplot per matchup; each model gets its own line.
    Shows whether performance is consistent or luck-driven.
    """
    model_labels   = list(all_results.keys())
    matchup_labels = list(next(iter(all_results.values())).keys())
    colors         = plt.cm.Set2(np.linspace(0, 1, len(model_labels)))

    fig, axes = plt.subplots(1, len(matchup_labels),
                             figsize=(6 * len(matchup_labels), 5), sharey=True)
    if len(matchup_labels) == 1:
        axes = [axes]

    fig.suptitle(f"Rolling Win Rate (window={window})", fontsize=14, fontweight='bold')

    for ax, matchup_label in zip(axes, matchup_labels):
        for model_label, color in zip(model_labels, colors):
            wins    = all_results[model_label][matchup_label]['wins']
            rolling = [np.mean(wins[max(0, i - window): i + 1])
                       for i in range(len(wins))]
            ax.plot(rolling, label=model_label, color=color)

        ax.axhline(1 / NUM_PLAYERS, color='red', linestyle='--',
                   alpha=0.6, label='Random baseline')
        ax.set_title(matchup_label, fontsize=10)
        ax.set_xlabel("Hand")
        ax.set_ylim(0, 1)

    axes[0].set_ylabel("Win Rate")
    axes[-1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("rolling_winrates.png", dpi=150)
    plt.show()
    print("Plot saved → rolling_winrates.png")


# ── Tournament ────────────────────────────────────────────────
def _RunHeadsUp(agent_a, agent_b, num_hands=TOURNAMENT_HANDS):
    """
    Plays num_hands of poker between two deep-copies of agent_a and agent_b
    so their real chip counts are never touched.

    Why deep-copy?  Agents are stateful (chips, hand, In).  If we mutated
    them in place the later matchups in the round-robin would start from
    whatever chips they had left over from earlier ones — which would make
    the scores meaningless.

    Returns (net_chips_a, net_chips_b) relative to STARTING_CHIPS.
    """
    a = copy.deepcopy(agent_a)
    b = copy.deepcopy(agent_b)
    a.chips   = STARTING_CHIPS
    b.chips   = STARTING_CHIPS
    a.epsilon = 0.0   # greedy — no random exploration during evaluation
    b.epsilon = 0.0

    agent_list = [a, b]

    for hand in range(num_hands):
        for _step in main.Round(agent_list, CARDS_PER_PLAYER, hand):
            pass   # just play; no training
        main.Cleanup(agent_list)

        # Rebuy anyone who busted so the game continues
        for agent in agent_list:
            if agent.chips == 0:
                agent.AddChips(STARTING_CHIPS)

    return a.chips - STARTING_CHIPS, b.chips - STARTING_CHIPS


def RunTournament(loaded_agents):
    """
    Round-robin tournament: every model plays every other model
    for TOURNAMENT_HANDS hands, head-to-head.

    Args:
        loaded_agents:  list of (model_label, DQNAgent) tuples

    Returns:
        scores  – dict { model_label -> total_net_chips }
        matrix  – dict { (label_a, label_b) -> net_chips_a }
                  (useful for the heat-map; note matrix[a,b] == -matrix[b,a])
    """
    scores = {label: 0 for label, _ in loaded_agents}
    matrix = {}   # (winner_label, loser_label) -> net chips won by winner

    matchups = list(itertools.combinations(loaded_agents, 2))
    total    = len(matchups)

    print(f"\n{'─'*60}")
    print(f"  TOURNAMENT  —  {total} matchup(s), {TOURNAMENT_HANDS} hands each")
    print(f"{'─'*60}")

    for i, ((label_a, agent_a), (label_b, agent_b)) in enumerate(matchups):
        chips_a, chips_b = _RunHeadsUp(agent_a, agent_b)
        scores[label_a] += chips_a
        scores[label_b] += chips_b
        matrix[(label_a, label_b)] =  chips_a
        matrix[(label_b, label_a)] = -chips_a   # symmetric

        print(f"  [{i+1}/{total}]  {label_a} vs {label_b}  →  "
              f"{label_a}: {chips_a:+d}   {label_b}: {chips_b:+d}")

    ranked = sorted(scores, key=scores.get, reverse=True)

    print(f"\n  Standings:")
    for rank, label in enumerate(ranked):
        print(f"    {rank+1}. {label:20s}  {scores[label]:+d} chips")

    return scores, matrix


def PlotTournament(scores, matrix, model_labels):
    """
    Two side-by-side panels:

    Left  — bar chart of total net chips per model (tournament leaderboard)
    Right — head-to-head heat-map showing chips won by the row model
            against each column model (diagonal is blank — no self-play)

    Reading the heat-map:
      • A green cell means the row model beat the column model on average.
      • A red cell means it lost.
      • The colour intensity shows how decisive the margin was.
    """
    n      = len(model_labels)
    colors = plt.cm.Set2(np.linspace(0, 1, n))

    fig, axes = plt.subplots(1, 2, figsize=(7 + n * 1.5, 5))
    fig.suptitle("Model Tournament Results", fontsize=14, fontweight='bold')

    # ── Left: leaderboard bar chart ───────────────────────────
    ax = axes[0]
    ranked_labels = sorted(model_labels, key=lambda l: scores[l], reverse=True)
    ranked_scores = [scores[l] for l in ranked_labels]
    bar_colors    = ['seagreen' if s >= 0 else 'tomato' for s in ranked_scores]

    bars = ax.barh(ranked_labels[::-1], ranked_scores[::-1],
                   color=bar_colors[::-1], alpha=0.88)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_title("Total Net Chips (round-robin)")
    ax.set_xlabel("Net Chips Won")

    for bar, val in zip(bars, ranked_scores[::-1]):
        x_pos = bar.get_width() + (max(abs(s) for s in ranked_scores) * 0.02)
        ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
                f"{val:+d}", va='center', fontsize=9)

    # ── Right: head-to-head heat-map ──────────────────────────
    ax = axes[1]

    # Build an n×n matrix; NaN on the diagonal (no self-play)
    grid = np.full((n, n), np.nan)
    for r, label_r in enumerate(model_labels):
        for c, label_c in enumerate(model_labels):
            if r != c and (label_r, label_c) in matrix:
                grid[r, c] = matrix[(label_r, label_c)]

    # Symmetric colour scale so 0 is always the midpoint
    vmax = np.nanmax(np.abs(grid)) if not np.all(np.isnan(grid)) else 1
    im   = ax.imshow(grid, cmap='RdYlGn', vmin=-vmax, vmax=vmax, aspect='auto')

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(model_labels, rotation=30, ha='right', fontsize=8)
    ax.set_yticklabels(model_labels, fontsize=8)
    ax.set_title("Head-to-Head Net Chips\n(row model vs col model)")

    # Annotate each cell with the chip value
    for r in range(n):
        for c in range(n):
            if not np.isnan(grid[r, c]):
                ax.text(c, r, f"{grid[r, c]:+.0f}",
                        ha='center', va='center', fontsize=8,
                        color='black')

    plt.colorbar(im, ax=ax, shrink=0.8, label="Net chips (row wins →)")

    plt.tight_layout()
    plt.savefig("tournament_results.png", dpi=150)
    plt.show()
    print("Plot saved → tournament_results.png")


# ── Main evaluation entry point ───────────────────────────────
def Evaluate():
    matchup_defs = _build_matchups(STARTING_CHIPS)

    # Validate all model paths up front — fail fast before any computation
    for path, label in MODELS:
        if not os.path.exists(path):
            print(f"[ERROR] Model not found: {path!r}  (label={label!r})")
            print("  → Check the MODELS list at the top of this file.")
            return

    # all_results[model_label][matchup_label] = stats dict
    all_results   = {}
    loaded_agents = []   # (label, agent) kept for the tournament

    for model_path, model_label in MODELS:
        print(f"\n{'='*60}")
        print(f"  Evaluating: {model_label}  ({model_path})")
        print(f"{'='*60}")

        dqn_agent = dqn.DQNAgent(STARTING_CHIPS, model_label, training=False, file=model_path)
        dqn_agent.Load()
        loaded_agents.append((model_label, dqn_agent))

        all_results[model_label] = {}

        for matchup_label, make_opponents in matchup_defs:
            print(f"\n  ▶ {matchup_label}")
            _reset_agents([dqn_agent])
            opponents = make_opponents()   # fresh opponents every time

            stats = EvaluateMatchup(
                dqn_agent,
                opponents,
                label=matchup_label,
                episodes=EVAL_EPISODES,
            )
            all_results[model_label][matchup_label] = stats

    # ── Summary table ─────────────────────────────────────────
    matchup_labels = [m for m, _ in matchup_defs]
    col = 14
    header = f"\n{'Model':<22}" + "".join(
        f"  {'WR':>5} {'Chips':>7}" for _ in matchup_labels
    )
    subheader = f"{'':22}" + "".join(
        f"  {m[:12]:>12}" for m in matchup_labels
    )
    print("\n" + "=" * len(header))
    print(subheader)
    print(header)
    print("=" * len(header))
    for model_label in all_results:
        row = f"{model_label:<22}"
        for matchup_label in matchup_labels:
            r = all_results[model_label][matchup_label]
            row += f"  {r['win_rate']:>5.1%} {r['avg_chip_delta']:>+7.1f}"
        print(row)
    print("=" * len(header))

    PlotEvaluation(all_results)
    PlotRollingWinRates(all_results)

    # ── Tournament: models play each other ────────────────────
    if len(loaded_agents) >= 2:
        t_scores, t_matrix = RunTournament(loaded_agents)
        PlotTournament(t_scores, t_matrix, [l for l, _ in loaded_agents])
    else:
        print("\n(Skipping tournament — need at least 2 models in MODELS list)")


if __name__ == "__main__":
    Evaluate()