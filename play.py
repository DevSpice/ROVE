from GameEngine import game, agents
from ML_Agent import dqn
import main

STARTING_CHIPS   = 200
CARDS_PER_PLAYER = 4
MODEL_PATH       = "dqn_final.pth"

def PrintAction(agent, action):

    if action[0] == 1:
        print(f"{agent.name} checked")
    elif action[1] == 1:
        print(f"{agent.name} folded")
    elif action[2] == 1:
        print(f"{agent.name} called")
    else:
        print(f"{agent.name} raised to {action[3]}")

if __name__ == "__main__":
    # Load trained bot
    dqn_agent = dqn.DQNAgent(STARTING_CHIPS, "Cheo", training=False, file=MODEL_PATH)
    dqn_agent.Load()
    dqn_agent.epsilon = 0.0  # always greedy, no random actions

    # Build your table
    name = input("What is your name?\n")
    you = agents.UserAgent(STARTING_CHIPS, name)
    agentLst = [
        you,
        dqn_agent,
        agents.SAgent(STARTING_CHIPS, "Tyler"),
        agents.SAgent(STARTING_CHIPS, "Jed"),
    ]

    roundNum = 0
    while True:
        print(f"\n=== Round {roundNum} ===")

        for step in main.Round(agentLst, CARDS_PER_PLAYER, roundNum):
            print("\n" + "=" * 50)
            print(f" -- ACTIONS -- ")
            print("=" * 50)
            if step['action'] is not None and step['actor'] is not None:
                
                PrintAction(step['actor'], step['action'])
            if step["winner"] is not None:
                winners = step["winner"]
                if type(winners) == list:
                    for winner in winners:
                        print(f"{winner.name} is the winner!\n")
                        print(f"With hand: {winner.hand.ShowHand()}")     

                else:
                    print(f"{winners.name} is the winner!\n")
                    print(f"With hand: {winners.hand.ShowHand()}")
            pass

        main.Cleanup(agentLst)
        roundNum += 1

        # Show standings
        print("\n--- Standings ---")
        for agent in agentLst:
            print(f"  {agent.name}: {agent.chips} chips")

        ans = input("\nPlay again? y/n: ").strip().lower()
        if ans == "n":
            break

        # Rebuy anyone who busted
        # for agent in agentLst:
        #     if agent.chips == 0:
        #         agent.AddChips(STARTING_CHIPS)