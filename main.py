
from GameEngine import game, agents
from ML_Agent import dqn

'''
This file starts a game with N agents playing against each other. 
As the game progresses, it 
This is where you can change the type of agent playing.

'''
rounds = 1000
startingChips = 200
SMALL_BLIND = startingChips // 20
BIG_BLIND = startingChips // 10

def EmptyHistory(agentLst):
    """Creates a blank history dict for all agents across all streets."""
    return {
        agent.name: {
            street: [0.0, 0.0, 0.0, 0.0]
            for street in ["Preflop", "Flop", "Turn", "River"]
        }
        for agent in agentLst
    }

def RecordAction(history, agent, street, action, pot_size):
    """
    Updates history for an agent on a given street.
    Features: [folded, passive, aggressive, amount_fraction]
    
    amount_fraction = how much they put in relative to pot size.
    This captures bet sizing which is a key poker tell.
    """
    entry = history[agent.name][street]

    if action.fold:
        entry[0] = 1.0
    elif action.check or action.call:
        entry[1] = 1.0
    else:
        entry[2] = 1.0
        if pot_size > 0:
            entry[3] = min(action.bet / pot_size, 2.0) / 2.0

def CheckWinner(isIn, agentLst):
    for agent in agentLst:
        if isIn.get(agent) is True:
            return agent

def PostBlind(agent, amount, table):
    """Forces an agent to post a blind bet"""
    actual = min(amount, agent.chips)  # can't post more than they have
    agent.RemoveChips(actual)
    agent.In += actual
    table.AddChips(actual)

def BettingRound(table, agentLst, roundNum, isIn, currBet=0, dealer=None, history=None, street="Preflop"):
    numAgents = len(agentLst)
    if dealer is None:
        dealer = roundNum % numAgents
    isIn['D'] = agentLst[dealer].name

    # Everyone active needs to act at least once
    to_act = [agentLst[(dealer + j) % numAgents] 
              for j in range(numAgents) 
              if isIn[agentLst[(dealer + j) % numAgents]]]
    
    lastBet = None
    min_raise = BIG_BLIND 
    
    while to_act:
        currAgent = to_act.pop(0)

        # Agent may have been made inactive since being added to to_act
        if not isIn[currAgent]:
            continue

        isIn['lastBet'] = lastBet 

        action = currAgent.Action(isIn, table, agentLst, currBet, history)

        if history is not None:
            RecordAction(history, currAgent, street, action, table.chips)

        if action.check == 1:
            pass

        elif action.fold == 1:
            isIn[currAgent] = False
            isIn["numIn"] -= 1
            if isIn["numIn"] == 1:
                winner = CheckWinner(isIn, agentLst)
                yield {
                    'actor': currAgent, 'action': action,
                    'table': table, 'agentLst': agentLst,
                    'currBet': currBet, 'done': True, 
                    'winner': winner, 'history': history
                }
                return

        elif action.call == 1:
            diff = min(currBet - currAgent.In, currAgent.chips)
            currAgent.RemoveChips(diff)
            currAgent.In += diff
            table.AddChips(diff)

        else:  # bet/raise
            lastBet = currAgent
            max_bet = currAgent.chips + currAgent.In  # total they can commit

            min_legal_bet = currBet + min_raise
            actual_bet = max(min(action.bet, max_bet), min_legal_bet)

            new_raise_size = actual_bet - currBet
            min_raise = new_raise_size

            diff = actual_bet - currAgent.In
            currAgent.In = actual_bet
            table.AddChips(diff)
            currAgent.RemoveChips(diff)
            currBet = actual_bet
            curr_idx = agentLst.index(currAgent)
            to_act = [
                agentLst[(curr_idx + j) % numAgents]
                for j in range(1, numAgents)
                if isIn[agentLst[(curr_idx + j) % numAgents]]
            ]


        yield {
            'actor': currAgent, 'action': action,
            'table': table, 'agentLst': agentLst,
            'currBet': currBet, 'done': False, 
            'winner': None, 'history': history
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
    isIn['Big'] = BIG_BLIND
    isIn['Small'] = SMALL_BLIND

    history = EmptyHistory(agentLst) 

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
                              currBet=BIG_BLIND, dealer=first_preflop, history=history, street="Preflop"):
        yield step
        if step['done']:
            Winner(step['winner'], table)
            return
    ResetBets(agentLst)

    # Flop, Turn, River
    first_postflop = (roundNum + 1) % numAgents
    for street_name, num_cards in [("Flop", 3), ("Turn", 1), ("River", 1)]:
        for _ in range(num_cards):
            table.hand.AddCard(deck.GetCard())

        for step in BettingRound(table, agentLst, roundNum, isIn,
                                  currBet=0, dealer=first_postflop,
                                  history=history, street=street_name):
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
        'currBet': 0, 'done': True, 
        'winner': winners, 'history' : history
    }


# if __name__ == "__main__": 

#     # We are playing omaha, so each player gets 4 cards
#     cardsPerPlayer = 4

#     agentLst = []

#     # agent1 = agents.UserAgent(200, "Trent")
#     # agentLst.append(agent1)

#     agent2 = agents.SAgent(startingChips, "Tyler")
#     agentLst.append(agent2)

#     agent3 = agents.SAgent(startingChips, "Cheo")
#     agentLst.append(agent3)

#     agent4 = agents.RAgent(startingChips, "Jed")
#     agentLst.append(agent4)

#     agent6 = dqn.DQNAgent(startingChips, "MLBot")
#     agentLst.append(agent6)

#     roundNum = 0

#     # while rounds > 0:

#     #     print("round number = " + str(roundNum))

#     #     Round(agentLst, cardsPerPlayer, roundNum)

#     #     Cleanup(agentLst)

#     #     roundNum += 1

#     #     rounds -= 1
#     # else:
#     #     print("\nRound limit reached!")

#     # Standing(agentLst)

#     while rounds > 0:
#         print("round number = " + str(roundNum))

#         chips_before = agent6.chips  # track DQN chips before hand

#         for step in Round(agentLst, cardsPerPlayer, roundNum):
#             # Only do ML bookkeeping when the DQN acted
#             if step['actor'] is not agent6 or step['action'] is None:
#                 continue

#             next_state = agents.BuildStateVector(
#                 agent6, step['table'], step['agentLst'], step['currBet']
#             )

#             done = step['done']
#             won_hand = None
#             if done:
#                 w = step['winner']
#                 won_hand = (agent6 in w) if isinstance(w, list) else (w is agent6)

#             reward = dqn.CalculateReward(
#                 agent=agent6,
#                 action=step['action'],
#                 chips_before=chips_before,
#                 chips_after=agent6.chips,
#                 won_hand=won_hand,
#                 pot_size=step['table'].chips
#             )

#             agent6.StoreExperience(next_state, reward, done)
#             agent6.Train()

#         Cleanup(agentLst)
#         agent6.DecayEpsilon()

#         if roundNum % dqn.TARGET_UPDATE == 0:
#             agent6.UpdateTargetNetwork()

#         # Rebuy anyone who busted
#         for agent in agentLst:
#             if agent.chips == 0:
#                 agent.AddChips(startingChips)
#                 agent.BuyIns += 1

#         roundNum += 1
#         rounds -= 1

#     Standing(agentLst)