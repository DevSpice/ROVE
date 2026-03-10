import itertools
import numpy as np
import copy
from GameEngine import agents
from ML_Agent import dqn, train
import main

STARTING_CHIPS    = 200
CARDS_PER_PLAYER  = 4
TOURNAMENT_HANDS  = 10   # hands per matchup
TRAIN_EPISODES    = 2000  # hands each agent trains before tournament
POPULATION_SIZE   = 20
SURVIVORS         = 10   # top N kept each generation
GENERATIONS       = 50


def RunMatchup(agent_a, agent_b, num_hands=TOURNAMENT_HANDS):
    """
    Runs num_hands of heads-up between two agents.
    Returns (chips_won_by_a, chips_won_by_b) relative to starting chips.
    Both agents play greedily (no training, epsilon=0).
    """
    # Clone so we don't mess up their chip counts
    a = copy.deepcopy(agent_a)
    b = copy.deepcopy(agent_b)
    a.chips   = STARTING_CHIPS
    b.chips   = STARTING_CHIPS
    a.epsilon = 0.0
    b.epsilon = 0.0

    agentLst = [a, b]

    for hand in range(num_hands):
        for step in main.Round(agentLst, CARDS_PER_PLAYER, hand):
            pass  # no training, just play
        main.Cleanup(agentLst)

        # Rebuy if busted so the game keeps going
        for agent in agentLst:
            if agent.chips == 0:
                agent.AddChips(STARTING_CHIPS)

    return a.chips - STARTING_CHIPS, b.chips - STARTING_CHIPS


def RunTournament(population):
    """
    Round-robin tournament — every agent plays every other agent.
    Returns agents sorted by total chips won (best first).
    """
    scores = {agent.name: 0 for agent in population}

    matchups = list(itertools.combinations(population, 2))
    total    = len(matchups)

    print(f"\n  Running {total} matchups ({TOURNAMENT_HANDS} hands each)...")

    for i, (a, b) in enumerate(matchups):
        chips_a, chips_b = RunMatchup(a, b)
        scores[a.name] += chips_a
        scores[b.name] += chips_b
        print(f"  [{i+1}/{total}] {a.name} vs {b.name} → "
              f"{a.name}: {chips_a:+d}  {b.name}: {chips_b:+d}")

    # Sort population by score
    ranked = sorted(population, key=lambda a: scores[a.name], reverse=True)

    print("\n  Tournament standings:")
    for rank, agent in enumerate(ranked):
        print(f"    {rank+1}. {agent.name}: {scores[agent.name]:+d} chips")

    return ranked, scores


def Reproduce(survivors, population_size, generation, new_epsilon):
    """
    Fills a new population by cloning survivors.
    Each survivor spawns (population_size // len(survivors)) children.
    Children get a fresh replay buffer but inherit the network weights.
    """
    new_population = []
    children_per_survivor = population_size // len(survivors)

    for i, parent in enumerate(survivors):
        for j in range(children_per_survivor):
            child_name = f"G{generation}_S{i}_C{j}"
            child = dqn.DQNAgent(STARTING_CHIPS, child_name, training=True)

            # Inherit network weights from parent
            child.online_net.load_state_dict(
                copy.deepcopy(parent.online_net.state_dict())
            )
            child.target_net.load_state_dict(
                child.online_net.state_dict()
            )

            # Fresh replay buffer — child explores on its own
            child.epsilon = new_epsilon
            new_population.append(child)

    return new_population


def TrainAgent(agent, num_episodes, opponent_pool_agents):
    """
    Trains a single agent for num_episodes against a mixed opponent pool.
    Uses the same training loop logic as train.py.
    """
    # from train2 import SnapshotPool, BuildOpponentList, TrainingStats
    import main

    pool  = train.SnapshotPool(max_size=5)
    stats = train.TrainingStats(window=100)

    for episode in range(num_episodes):
        if episode % 200 == 0:
            pool.AddSnapshot(agent)

        opponents = train.BuildOpponentList(agent, pool, num_opponents=4)
        agentLst  = opponents + [agent]
        chips_before = agent.chips
        episode_loss = []

        for step in main.Round(agentLst, CARDS_PER_PLAYER, episode):
            if step['actor'] is not agent or step['action'] is None:
                continue

            next_state = agents.BuildStateVector(
                agent, step['table'], step['agentLst'],
                step['currBet'], history=step.get('history')
            )

            done = step['done']
            won_hand = None
            if done:
                w = step['winner']
                won_hand = (agent in w) if isinstance(w, list) else (w is agent)

            reward = dqn.CalculateReward(
                agent=agent,
                action=step['action'],
                chips_before=chips_before,
                chips_after=agent.chips,
                won_hand=won_hand,
                pot_size=step['table'].chips
            )

            agent.StoreExperience(next_state, reward, done)
            loss = agent.Train()
            if loss is not None:
                episode_loss.append(loss)

        agent.chips = STARTING_CHIPS
        agent.In    = 0
        main.Cleanup([agent])
        agent.DecayEpsilon()

        if episode % 50 == 0:
            print(f"    {agent.name} ep {episode}/{num_episodes} "
                  f"ε={agent.epsilon:.3f}")


def RunEvolution():
    """
    Main evolutionary loop.
    """
    import os
    os.makedirs("evolution", exist_ok=True)

    # Epsilon schedule across generations — starts high, decays each gen
    epsilon_schedule = [1.0, 0.5, 0.25, 0.1, 0.05]

    # Generation 0 — fresh random population
    population = [
        dqn.DQNAgent(STARTING_CHIPS, f"G0_Agent_{i}", training=True)
        for i in range(POPULATION_SIZE)
    ]

    all_generation_scores = []

    for gen in range(GENERATIONS):
        starting_epsilon = epsilon_schedule[min(gen, len(epsilon_schedule) - 1)]
        print(f"\n{'='*60}")
        print(f"GENERATION {gen}  |  epsilon: {starting_epsilon}  "
              f"|  {POPULATION_SIZE} agents  |  {TRAIN_EPISODES} episodes each")
        print(f"{'='*60}")

        # Set epsilon for this generation
        for agent in population:
            agent.epsilon = starting_epsilon

        # Train every agent in the population
        for i, agent in enumerate(population):
            print(f"\n  Training {agent.name} ({i+1}/{POPULATION_SIZE})...")
            TrainAgent(agent, TRAIN_EPISODES, population)

        # Tournament
        print(f"\n  Running Generation {gen} tournament...")
        ranked, scores = RunTournament(population)
        all_generation_scores.append((gen, scores))

        # Save the best agent from this generation
        best = ranked[0]
        best.Save(f"evolution/best_gen{gen}.pth")
        print(f"\n  Best agent: {best.name} — saved to evolution/best_gen{gen}.pth")

        # Stop after last generation — no need to reproduce
        if gen == GENERATIONS - 1:
            break

        # Reproduce from survivors
        survivors        = ranked[:SURVIVORS]
        next_epsilon     = epsilon_schedule[min(gen + 1, len(epsilon_schedule) - 1)]
        population       = Reproduce(survivors, POPULATION_SIZE, gen + 1, next_epsilon)
        print(f"\n  Survivors: {[s.name for s in survivors]}")
        print(f"  New population of {len(population)} agents created")

    # Final summary
    print(f"\n{'='*60}")
    print("EVOLUTION COMPLETE")
    print(f"{'='*60}")
    for gen, scores in all_generation_scores:
        best_name  = max(scores, key=scores.get)
        best_score = scores[best_name]
        print(f"  Gen {gen} best: {best_name} ({best_score:+d} chips)")

    print(f"\nLoad the final best agent with:")
    print(f"  agent.Load('evolution/best_gen{GENERATIONS-1}.pth')")


if __name__ == "__main__":
    RunEvolution()
