
from GameEngine import game, agents

'''
This file starts a game with N agents playing against each other. 
As the game progresses, it 
This is where you can change the type of agent playing.

'''

SMALL_BLIND = 10
BIG_BLIND = 20

def CheckWinner(isIn):
    if isIn["numIn"] == 1:
        for agent, value in isIn.items():
            if value == True:
                return agent


def PostBlind(agent, amount, table):
    """Forces an agent to post a blind bet"""
    actual = min(amount, agent.chips)  # can't post more than they have
    agent.RemoveChips(actual)
    agent.In += actual
    table.AddChips(actual)

def BettingRound(table, agentLst, roundNum, isIn, currBet=0, dealer=None):
    numAgents = len(agentLst)

    if dealer is None:
        dealer = roundNum % numAgents

    # Build the set of active players who still need to act
    to_act = {agent for agent in agentLst if isIn[agent]}
    i = 0

    while to_act:
        idx = (dealer + i) % numAgents
        currAgent = agentLst[idx]
        i += 1

        if not isIn[currAgent]:
            continue  # Skip folded players

        if currAgent not in to_act:
            continue  # This player already acted and no one re-raised

        to_act.discard(currAgent)  # They're about to act

        action = currAgent.Action(table, agentLst, currBet)

        if action.check == 1:
            print("check")
            # No change needed — player acted, already removed from to_act

        elif action.fold == 1:
            print("fold")
            isIn[currAgent] = False
            isIn["numIn"] -= 1

            if isIn["numIn"] == 1:
                winner = CheckWinner(isIn)
                return winner

        elif action.call == 1:
            print("call")
            diff = currBet - currAgent.In
            diff = min(diff, currAgent.chips)
            currAgent.RemoveChips(diff)
            currAgent.In += diff
            table.AddChips(diff)

        else:  # Raise/Bet
            print("bet")
            diff = action.bet - currAgent.In
            currAgent.In = action.bet
            table.AddChips(diff)
            currAgent.RemoveChips(diff)
            currBet = action.bet
            # Re-add all OTHER active players — they must respond to the raise
            to_act = {a for a in agentLst if isIn[a] and a != currAgent}


def Cleanup(agentLst):
    for agent in agentLst:
        agent.hand.ClearHand()
        agent.In = 0

def ResetBets(agentLst):
    for agent in agentLst:
        agent.In = 0

def Standing(agentList):
    for agent in agentList:
        print(f"{agent.name}: {agent.chips} - Buy ins: {agent.BuyIns}")

def Winner(result, table):
    winnings = table.chips
    result.AddChips(winnings)
    table.RemoveChips(winnings)

def Round(agentLst, CPP, roundNum):
    print('New Round')
    deck = game.Deck()
    table = agents.Agent(0, "T")
    isIn = {agent: True for agent in agentLst}
    numAgents = len(agentLst)
    isIn["numIn"] = numAgents

    sb_idx = (roundNum + 1) % numAgents
    bb_idx = (roundNum + 2) % numAgents
    first_preflop = (roundNum + 3) % numAgents

    PostBlind(agentLst[sb_idx], SMALL_BLIND, table)
    PostBlind(agentLst[bb_idx], BIG_BLIND, table)
    # print(f"Blinds posted. Pot: {table.chips}")

    deck.RShuffle()

    # Deals the cards to each agent
    for i in range(CPP):
        # print(i)
        for agent in agentLst:
            # print("Gave card to agent")
            agent.hand.AddCard(deck.GetCard())


    result = BettingRound(table, agentLst, roundNum, isIn, currBet=BIG_BLIND, dealer=first_preflop)
    if result is not None:
        Winner(result, table)
        return
    ResetBets(agentLst)

    first_postflop = (roundNum + 1) % numAgents
    
    for street, num_cards in [("Flop", 3), ("Turn", 1), ("River", 1)]:
        for _ in range(num_cards):
            table.hand.AddCard(deck.GetCard())
        # print(f"\n--- {street} ---")
        table.hand.ShowHand()
        
        result = BettingRound(table, agentLst, roundNum, isIn, currBet=0, dealer=first_postflop)
        if result is not None:
            Winner(result, table); return
        ResetBets(agentLst)
        
    # print("\n--- Showdown ---")
    winners = agents.CompareHands(agentLst, isIn, table.hand)
    split = table.chips // len(winners)
    for w in winners:
        w.AddChips(split)
        print(f"{w.name} wins {split} chips!")
    table.RemoveChips(table.chips)



if __name__ == "__main__": 

    # We are playing omaha, so each player gets 4 cards
    cardsPerPlayer = 4

    agentLst = []

    # agent1 = agents.UserAgent(200, "Trent")
    # agentLst.append(agent1)

    agent2 = agents.SAgent(200, "Tyler")
    agentLst.append(agent2)

    agent3 = agents.SAgent(200, "Cheo")
    agentLst.append(agent3)

    agent4 = agents.RAgent(200, "Jed")
    agentLst.append(agent4)


    roundNum = 0
    rounds = 10000


    while rounds > 0 :
        
        print("round number = " + str(roundNum))

        Round(agentLst, cardsPerPlayer, roundNum)

        Cleanup(agentLst)
        
        # ans = input("play again? y/n\n")
        # if ans == "n":
        #     playing = False
        roundNum += 1
        for agent in agentLst:
            if agent.chips == 0:
                print(f"{agent.name} bought back in!")
                agent.BuyBack(200)

        rounds -= 1
    
    Standing(agentLst)