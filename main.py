
from GameEngine import game, agents
from ML_Agent import dqn

'''
This file starts a game with N agents playing against each other. 
As the game progresses, it 
This is where you can change the type of agent playing.

'''
rounds = 1000
startingChips = 500
SMALL_BLIND = startingChips // 10
BIG_BLIND = startingChips // 20

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

    # Everyone active needs to act at least once
    to_act = [agentLst[(dealer + j) % numAgents] 
              for j in range(numAgents) 
              if isIn[agentLst[(dealer + j) % numAgents]]]
    
    while to_act:
        currAgent = to_act.pop(0)

        # Agent may have been made inactive since being added to to_act
        if not isIn[currAgent]:
            continue

        action = currAgent.Action(table, agentLst, currBet)

        if action.check == 1:
            pass

        elif action.fold == 1:
            isIn[currAgent] = False
            isIn["numIn"] -= 1
            if isIn["numIn"] == 1:
                winner = CheckWinner(isIn)
                yield {
                    'actor': currAgent, 'action': action,
                    'table': table, 'agentLst': agentLst,
                    'currBet': currBet, 'done': True, 'winner': winner
                }
                return

        elif action.call == 1:
            diff = min(currBet - currAgent.In, currAgent.chips)
            currAgent.RemoveChips(diff)
            currAgent.In += diff
            table.AddChips(diff)

        else:  # bet/raise
            diff = action.bet - currAgent.In
            currAgent.In = action.bet
            table.AddChips(diff)
            currAgent.RemoveChips(diff)
            currBet = action.bet
            curr_idx = agentLst.index(currAgent)
            to_act = [
                agentLst[(curr_idx + j) % numAgents]
                for j in range(1, numAgents)
                if isIn[agentLst[(curr_idx + j) % numAgents]]
            ]

        yield {
            'actor': currAgent, 'action': action,
            'table': table, 'agentLst': agentLst,
            'currBet': currBet, 'done': False, 'winner': None
        }


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

    deck.RShuffle()
    for i in range(CPP):
        for agent in agentLst:
            agent.hand.AddCard(deck.GetCard())

    # Pre-flop
    for step in BettingRound(table, agentLst, roundNum, isIn,
                              currBet=BIG_BLIND, dealer=first_preflop):
        yield step
        if step['done']:
            Winner(step['winner'], table)
            return
    ResetBets(agentLst)

    # Flop, Turn, River
    first_postflop = (roundNum + 1) % numAgents
    for street, num_cards in [("Flop", 3), ("Turn", 1), ("River", 1)]:
        for _ in range(num_cards):
            table.hand.AddCard(deck.GetCard())

        for step in BettingRound(table, agentLst, roundNum, isIn,
                                  currBet=0, dealer=first_postflop):
            yield step
            if step['done']:
                Winner(step['winner'], table)
                return
        ResetBets(agentLst)

    # Showdown
    winners = agents.CompareHands(agentLst, isIn, table.hand)
    split = table.chips // len(winners)
    for w in winners:
        w.AddChips(split)
    table.RemoveChips(table.chips)

    # Yield a final done signal so the training loop knows the hand ended
    yield {
        'actor': None, 'action': None,
        'table': table, 'agentLst': agentLst,
        'currBet': 0, 'done': True, 'winner': winners
    }


if __name__ == "__main__": 

    # We are playing omaha, so each player gets 4 cards
    cardsPerPlayer = 4

    agentLst = []

    # agent1 = agents.UserAgent(200, "Trent")
    # agentLst.append(agent1)

    agent2 = agents.SAgent(startingChips, "Tyler")
    agentLst.append(agent2)

    agent3 = agents.SAgent(startingChips, "Cheo")
    agentLst.append(agent3)

    agent4 = agents.RAgent(startingChips, "Jed")
    agentLst.append(agent4)

    agent6 = dqn.DQNAgent(startingChips, "MLBot")
    agentLst.append(agent6)

    roundNum = 0

    # while rounds > 0:

    #     print("round number = " + str(roundNum))

    #     Round(agentLst, cardsPerPlayer, roundNum)

    #     Cleanup(agentLst)

    #     roundNum += 1

    #     rounds -= 1
    # else:
    #     print("\nRound limit reached!")

    # Standing(agentLst)

    while rounds > 0:
        print("round number = " + str(roundNum))

        chips_before = agent6.chips  # track DQN chips before hand

        for step in Round(agentLst, cardsPerPlayer, roundNum):
            # Only do ML bookkeeping when the DQN acted
            if step['actor'] is not agent6 or step['action'] is None:
                continue

            next_state = agents.BuildStateVector(
                agent6, step['table'], step['agentLst'], step['currBet']
            )

            done = step['done']
            won_hand = None
            if done:
                w = step['winner']
                won_hand = (agent6 in w) if isinstance(w, list) else (w is agent6)

            reward = dqn.CalculateReward(
                agent=agent6,
                action=step['action'],
                chips_before=chips_before,
                chips_after=agent6.chips,
                won_hand=won_hand,
                pot_size=step['table'].chips
            )

            agent6.StoreExperience(next_state, reward, done)
            agent6.Train()

        Cleanup(agentLst)
        agent6.DecayEpsilon()

        if roundNum % dqn.TARGET_UPDATE == 0:
            agent6.UpdateTargetNetwork()

        # Rebuy anyone who busted
        for agent in agentLst:
            if agent.chips == 0:
                agent.AddChips(startingChips)
                agent.BuyIns += 1

        roundNum += 1
        rounds -= 1

    Standing(agentLst)