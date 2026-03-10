import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from GameEngine import game, agents
import main

# ─────────────────────────────────────────
# Hyperparameters — tune these later
# ─────────────────────────────────────────
GAMMA           = 0.99    # Discount factor
LR              = 1e-3    # Learning rate
BATCH_SIZE      = 64      # How many experiences to sample per training step
BUFFER_SIZE     = 50_000  # Max experiences stored in replay buffer
EPSILON_START   = 1.0     # Start fully random
EPSILON_END     = 0.05    # Never go below 5% random
EPSILON_DECAY   = 0.9995   # Multiply epsilon by this each episode
TARGET_UPDATE   = 50      # Copy online → target network every N episodes
STATE_DIM       = 119      # Must match your BuildStateVector output
ACTION_DIM      = 4       # fold, check, call, bet


# ─────────────────────────────────────────
# 1. The Neural Network
# ─────────────────────────────────────────
class PokerNet(nn.Module):
    """
    Simple 3-layer fully connected network.
    Input:  63-dimensional state vector
    Output: 4 Q-values (one per action)
    """
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )

    def forward(self, x):
        return self.net(x)


# ─────────────────────────────────────────
# 2. Replay Buffer
# ─────────────────────────────────────────
class ReplayBuffer:
    """
    Stores (state, action, reward, next_state, done) tuples.
    'done' is True if the hand ended after this action.
    Sampling random batches breaks temporal correlation.
    """
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def Push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def Sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(np.array(states)),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(dones)
        )

    def __len__(self):
        return len(self.buffer)
    

def CalculateReward(agent, action, chips_before, chips_after, won_hand, pot_size):
    """
    Combines a sparse terminal reward (chips won/lost) with
    light intermediate shaping to reduce the credit assignment problem.
    
    Args:
        agent:        The DQN agent
        action:       The game.Action namedtuple taken
        chips_before: Agent's chip count before this hand started
        chips_after:  Agent's chip count after this action resolves
        won_hand:     True if the agent won the hand (only set at showdown)
        pot_size:     Current pot size (for scaling fold penalty)
    
    Returns:
        float reward
    """
    reward = 0.0

    # ── Terminal reward (end of hand) ──────────────────────────────
    # This is the primary learning signal. Chip delta is normalized
    # by the big blind so rewards stay in a consistent scale regardless
    # of stack sizes.
    if won_hand is not None:
        chip_delta = chips_after - chips_before
        change = chip_delta / main.BIG_BLIND
        # print(f"Change from winning: {change}")
        reward += change  # normalize by big blind

    # ── Intermediate shaping ───────────────────────────────────────
    # Small signal to help with credit assignment mid-hand.
    # These are intentionally small so they don't override the
    # terminal signal.
    if action is not None: 
        if action.fold == 1:
            # Folding loses whatever you put in — penalize proportionally
            # but lightly (the terminal reward will handle the rest)
            sub = (agent.In / main.BIG_BLIND)
            # print(f"Change in reward from folding: -{sub}")
            reward -= sub

        # if action.bet > 0:
            # change = (agent.In) /  pot_size
            # print(f"Change in reward from betting: {0}")
        #     reward += (agent.In) /  pot_size

    # if chips_after == 0:
    #     reward -= 10

    return reward


# ─────────────────────────────────────────
# 3. The DQN Agent
# ─────────────────────────────────────────
class DQNAgent(agents.Agent):
    """
    Wraps the neural network in a poker Agent.
    
    During training:  uses epsilon-greedy (sometimes random, sometimes network)
    During inference: always uses the network (epsilon=0)
    """
    def __init__(self, chips, name, training=True, file="dqn_final.pth"):
        super().__init__(chips, name)
        self.file       = file
        self.training   = training
        self.epsilon    = EPSILON_START if training else 0.0

        # Two networks — online gets trained, target provides stable Q targets
        self.online_net = PokerNet(STATE_DIM, ACTION_DIM)
        self.target_net = PokerNet(STATE_DIM, ACTION_DIM)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()  # Target network never trains directly

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=LR)
        self.buffer    = ReplayBuffer(BUFFER_SIZE)

        # Tracks the last state/action so we can store experience after
        # the environment tells us the reward
        self.last_state  = None
        self.last_action = None

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------
    def Action(self, isIn=None, table=None, agentLst=None, currBet=0, history=None):

        state = agents.BuildStateVector(self, table, agentLst, currBet, history)

        # Epsilon-greedy: explore randomly or exploit the network
        if self.training and random.random() < self.epsilon:
            action_idx = random.randint(0, ACTION_DIM - 1)
        else:
            with torch.no_grad():
                q_values   = self.online_net(torch.FloatTensor(state))
                action_idx = q_values.argmax().item()

        # Store for when we receive the reward
        self.last_state  = state
        self.last_action = action_idx

        return self._IndexToAction(action_idx, currBet)

    # ------------------------------------------------------------------
    # Reward storage + training
    # ------------------------------------------------------------------
    def StoreExperience(self, next_state, reward, done):
        """Call this after the environment resolves an action."""
        if self.last_state is not None:
            # last = np.array(self.last_state, dtype=np.float32).flatten()
            # next_ = np.array(next_state, dtype=np.float32).flatten()
            
            # assert last.shape == next_.shape, \
            #     f"State shape mismatch: last={last.shape}, next={next_.shape}"
            
            # self.buffer.Push(last, self.last_action, reward, next_, float(done))

            self.buffer.Push(
                self.last_state, self.last_action,
                reward, next_state, float(done)
            )

    def Train(self):
        """
        One gradient update step. Call this after StoreExperience.
        Does nothing until the buffer has enough experiences.
        """
        if len(self.buffer) < BATCH_SIZE:
            return

        states, actions, rewards, next_states, dones = self.buffer.Sample(BATCH_SIZE)

        # Current Q-values for the actions we actually took
        current_q = self.online_net(states).gather(1, actions.unsqueeze(1)).squeeze()

        # Target Q-values via Bellman equation, using frozen target network
        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(1)[0]
            target_q   = rewards + GAMMA * max_next_q * (1 - dones)

        loss = nn.MSELoss()(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def UpdateTargetNetwork(self):
        """Copy online network weights → target network."""
        self.target_net.load_state_dict(self.online_net.state_dict())

    def DecayEpsilon(self):
        """Call once per episode during training."""
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------
    # def Save(self, path="dqn_poker.pth"):
    #     torch.save(self.online_net.state_dict(), path)
    #     print(f"Model saved to {path}")

    # def Load(self, path="dqn_poker.pth"):
    #     self.online_net.load_state_dict(torch.load(path))
    #     self.target_net.load_state_dict(self.online_net.state_dict())
    #     print(f"Model loaded from {path}")

    def Save(self, path=None):
        if path is None:
            path = self.file
        torch.save({
            'model_state': self.online_net.state_dict(),
            'epsilon':     self.epsilon
        }, path)
        print(f"Model saved to {path} (epsilon: {self.epsilon:.3f})")

    def Load(self):
        path = self.file
        checkpoint = torch.load(path, weights_only=True)
        self.online_net.load_state_dict(checkpoint['model_state'])
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.epsilon = checkpoint.get('epsilon', EPSILON_END)
        print(f"Model loaded from {path} (epsilon: {self.epsilon:.3f})")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _IndexToAction(self, idx, currBet):
        """Converts a network output index to a game Action."""
        if idx == 0:  # fold
            return game.Action(check=0, fold=1, call=0, bet=0)
        elif idx == 1:  # check
            if currBet > 0:
                # Can't check when there's a bet — fall back to call
                return game.Action(check=0, fold=0, call=1, bet=currBet)
            return game.Action(check=1, fold=0, call=0, bet=0)
        elif idx == 2:  # call
            if currBet == 0:
                return game.Action(check=1, fold=0, call=0, bet=0)
            return game.Action(check=0, fold=0, call=1, bet=currBet)
        else:  # bet
            amount = max(currBet + 1, int(self.chips * 0.2))
            amount = min(amount, self.chips)
            if amount <= currBet:
                return game.Action(check=0, fold=0, call=1, bet=currBet)
            return game.Action(check=0, fold=0, call=0, bet=amount)