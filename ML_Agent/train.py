import numpy as np
import os
import matplotlib.pyplot as plt
import json
import torch
from GameEngine import agents
from ML_Agent import dqn
import main

# ── Config ────────────────────────────────────────────────────
STARTING_CHIPS  = 500
CARDS_PER_PLAYER = 4
NUM_EPISODES    = 50000
LOG_INTERVAL    = 100   # print to console every N hands
SAVE_INTERVAL   = 500   # save model checkpoint every N hands
TARGET_UPDATE   = 50    # sync target network every N hands

# ── Statistics tracker ────────────────────────────────────────
class TrainingStats:
    """
    Collects per-episode metrics and computes rolling statistics.
    All data is stored so you can plot anything after training.
    """
    def __init__(self, window=100):
        self.window = window

        # Per-episode raw data
        self.chip_deltas    = []   # chips won/lost each hand
        self.wins           = []   # 1 if won, 0 if lost
        self.losses         = []   # network loss values
        self.epsilons       = []   # epsilon at end of each episode
        self.actions        = []   # (fold, check, call, bet) counts per episode

        # Cumulative action counts for this episode
        self._episode_actions = [0, 0, 0, 0]

    def LogAction(self, action_idx):
        self._episode_actions[action_idx] += 1

    def LogEpisode(self, chip_delta, won, loss, epsilon):
        self.chip_deltas.append(chip_delta)
        self.wins.append(1 if won else 0)
        self.losses.append(loss if loss is not None else 0)
        self.epsilons.append(epsilon)
        self.actions.append(self._episode_actions.copy())
        self._episode_actions = [0, 0, 0, 0]

    def RollingWinRate(self):
        if len(self.wins) < self.window:
            return np.mean(self.wins) if self.wins else 0
        return np.mean(self.wins[-self.window:])

    def RollingAvgChips(self):
        if len(self.chip_deltas) < self.window:
            return np.mean(self.chip_deltas) if self.chip_deltas else 0
        return np.mean(self.chip_deltas[-self.window:])

    def Save(self, path="training_stats.json"):
        data = {
            "chip_deltas": self.chip_deltas,
            "wins": self.wins,
            "losses": self.losses,
            "epsilons": self.epsilons,
            "actions": self.actions
        }
        with open(path, "w") as f:
            json.dump(data, f)
        print(f"Stats saved to {path}")


    def Load(self, path="training_stats.json"):
        with open(path, "r") as f:
            data = json.load(f)

        self.chip_deltas = data["chip_deltas"]
        self.wins = data["wins"]
        self.losses = data["losses"]
        self.epsilons = data["epsilons"]
        self.actions = data["actions"]

        print(f"Stats loaded from {path}")


# ── Plotting ──────────────────────────────────────────────────
def PlotResults(stats):
    """Generates 4 subplots summarizing training."""
    episodes = range(len(stats.wins))
    window   = stats.window

    # Compute rolling stats
    def rolling(data):
        return [np.mean(data[max(0, i - window):i + 1])
                for i in range(len(data))]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("DQN Poker Bot — Training Results", fontsize=16)

    # ── 1. Rolling win rate ───────────────────────────────────
    ax = axes[0, 0]
    ax.plot(episodes, rolling(stats.wins), color="steelblue")
    ax.axhline(0.5, color="gray", linestyle="--", label="50% baseline")
    ax.set_title(f"Win Rate (rolling {window})")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Win Rate")
    ax.legend()

    # ── 2. Average chips won/lost ─────────────────────────────
    ax = axes[0, 1]
    ax.plot(episodes, rolling(stats.chip_deltas), color="seagreen")
    ax.axhline(0, color="gray", linestyle="--")
    ax.set_title(f"Avg Chips Won/Lost (rolling {window})")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Chips")

    # ── 3. Action distribution over time ─────────────────────
    ax = axes[1, 0]
    actions_arr = np.array(stats.actions, dtype=float)
    totals = actions_arr.sum(axis=1, keepdims=True)
    totals[totals == 0] = 1  # avoid divide by zero
    pcts = actions_arr / totals

    labels = ["Fold", "Check", "Call", "Bet"]
    colors = ["tomato", "gold", "cornflowerblue", "mediumorchid"]
    for idx, (label, color) in enumerate(zip(labels, colors)):
        smoothed = rolling(pcts[:, idx].tolist())
        ax.plot(episodes, smoothed, label=label, color=color)
    ax.set_title("Action Distribution Over Time")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Fraction of Actions")
    ax.legend()

    # ── 4. Training loss + epsilon ────────────────────────────
    ax = axes[1, 1]
    ax2 = ax.twinx()
    ax.plot(episodes, rolling(stats.losses), color="coral", label="Loss")
    ax2.plot(episodes, stats.epsilons, color="slategray",
             linestyle="--", label="Epsilon")
    ax.set_title("Network Loss & Epsilon Decay")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Loss", color="coral")
    ax2.set_ylabel("Epsilon", color="slategray")
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig("training_results.png", dpi=150)
    plt.show()
    print("Plot saved to training_results.png")


# ── Training loop ─────────────────────────────────────────────
def Train():
    # Build agent list — DQN trains against a mix of opponents
    dqn_agent = dqn.DQNAgent(STARTING_CHIPS, "DQN", training=True)

    if os.path.exists("dqn_final.pth"):
        dqn_agent.Load("dqn_final.pth")
        print("Resuming from saved model...")
    else:
        print("Starting fresh training run...")

    stats = TrainingStats(window=100)

    opponent_pool = [
        agents.SAgent(STARTING_CHIPS, "Tyler"),
        agents.SAgent(STARTING_CHIPS, "Cheo"),
        agents.RAgent(STARTING_CHIPS, "Jed"),
    ]
    agentLst = opponent_pool + [dqn_agent]

    roundNum = 0

    if os.path.exists("training_stats.json"):
        stats.Load("training_stats.json")
        print(f"Loaded {len(stats.wins)} previous episodes of stats")
        roundNum = len(stats.wins)

    for episode in range(NUM_EPISODES):
        chips_before = dqn_agent.chips
        episode_loss = []

        for step in main.Round(agentLst, CARDS_PER_PLAYER, roundNum):
            if step['actor'] is not dqn_agent or step['action'] is None:
                continue

            # Track what action the DQN took
            action = step['action']
            if action.fold:   stats.LogAction(0)
            elif action.check: stats.LogAction(1)
            elif action.call:  stats.LogAction(2)
            else:              stats.LogAction(3)

            # Build next state
            next_state = agents.BuildStateVector(
                dqn_agent, step['table'], step['agentLst'], step['currBet']
            )

            # Determine reward
            done = step['done']
            won_hand = None
            if done:
                w = step['winner']
                won_hand = (dqn_agent in w) if isinstance(w, list) else (w is dqn_agent)

            reward = dqn.CalculateReward(
                agent=dqn_agent,
                action=step['action'],
                chips_before=chips_before,
                chips_after=dqn_agent.chips,
                won_hand=won_hand,
                pot_size=step['table'].chips
            )

            dqn_agent.StoreExperience(next_state, reward, done)
            loss = dqn_agent.Train()
            if loss is not None:
                episode_loss.append(loss)

        # ── End of hand bookkeeping ───────────────────────────
        chip_delta = dqn_agent.chips - chips_before
        won        = chip_delta > 0
        avg_loss   = np.mean(episode_loss) if episode_loss else None

        stats.LogEpisode(chip_delta, won, avg_loss, dqn_agent.epsilon)

        dqn_agent.DecayEpsilon()
        if episode % TARGET_UPDATE == 0:
            dqn_agent.UpdateTargetNetwork()

        # Rebuy anyone who busted
        for agent in agentLst:
            if agent.chips == 0:
                agent.AddChips(STARTING_CHIPS)

        main.Cleanup(agentLst)
        roundNum += 1

        # ── Logging ───────────────────────────────────────────
        if episode % LOG_INTERVAL == 0:
            print(f"Episode {episode:>5} | "
                  f"Win rate: {stats.RollingWinRate():.2%} | "
                  f"Avg chips: {stats.RollingAvgChips():+.1f} | "
                  f"Epsilon: {dqn_agent.epsilon:.3f} | "
                  f"Buffer: {len(dqn_agent.buffer)}")

        if episode % SAVE_INTERVAL == 0 and episode > 0:
            dqn_agent.Save(f"checkpoints/dqn_ep{episode}.pth")
            stats.Save()

    # ── Final save and plot ───────────────────────────────────
    dqn_agent.Save("dqn_final.pth")
    stats.Save()
    PlotResults(stats)


if __name__ == "__main__":
    os.makedirs("checkpoints", exist_ok=True)
    Train()