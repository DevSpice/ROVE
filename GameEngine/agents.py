from GameEngine import game
import random


class Agent:

    def Action(self, table=None, agents=None, currBet=0):
        return game.Action(check=0, fold=1, call=0, bet=0)

    def __init__(self, chips, name):
        self.hand = game.Hand()
        self.chips = int(chips)
        self.name = name
        self.In = 0
        self.BuyIns = 0
    
    def AddChips(self, numChips):
        self.chips += numChips
    
    def RemoveChips(self, numChips):
        self.chips -= numChips
    
    def BuyBack(self, numChips):
        self.AddChips(numChips)
        self.BuyIns += 1


class RAgent(Agent):
    """
    Random Agent — makes legally valid moves chosen uniformly at random.
    
    When there is no outstanding bet, it can check or bet.
    When there is a bet to face, it can fold, call, or raise.
    This gives us a useful baseline: any 'smart' agent should
    win significantly more often than RAgent over many hands.
    """
    def __init__(self, chips, name):
        super().__init__(chips, name)

    def Action(self, table=None, agents=None, currBet=0):
        if currBet == 0:
            # No bet to face — choose randomly between checking and betting
            if random.random() < 0.5 or self.chips == 0:
                return game.Action(check=1, fold=0, call=0, bet=0)
            else:
                # Bet a random amount between 1 chip and half our stack
                amount = random.randint(1, max(1, self.chips // 2))
                return game.Action(check=0, fold=0, call=0, bet=amount)
        else:
            # There is an outstanding bet — must fold, call, or raise
            options = ['fold', 'call', 'raise']
            choice = random.choice(options)

            if choice == 'fold':
                return game.Action(check=0, fold=1, call=0, bet=0)

            elif choice == 'call':
                if self.chips >= currBet:
                    return game.Action(check=0, fold=0, call=1, bet=currBet)
                # Can't afford to call — forced fold
                return game.Action(check=0, fold=1, call=0, bet=0)

            else:  # raise
                min_raise = currBet + 1
                if self.chips > min_raise:
                    amount = random.randint(min_raise, max(min_raise, self.chips // 2))
                    return game.Action(check=0, fold=0, call=0, bet=amount)
                elif self.chips >= currBet:
                    # Not enough to raise, but can still call
                    return game.Action(check=0, fold=0, call=1, bet=currBet)
                else:
                    return game.Action(check=0, fold=1, call=0, bet=0)

class UserAgent(Agent):
    def __init__(self, chips, name):
        super().__init__(chips, name)

    def Action(self, table=None, agents=None, currBet=0):
        """
        Prompts the user via the command line to make a betting decision.
        Displays the current game state before asking for input.

        Args:
            table:    The table Agent object (so we can show community cards + pot)
            agents:   List of all agents (so we can show who's still in)
            currBet:  The current bet that needs to be matched to call
        """

        # ---- Display Game State ----
        print("\n" + "=" * 50)
        print(f"  YOUR TURN: {self.name}")
        print("=" * 50)

        # Show the community cards on the table
        if table is not None:
            print(f"\n  [Table - Pot: {table.chips} chips]")
            if table.hand.cards:
                table.hand.ShowHand()
            else:
                print("  No community cards yet.")

        # Show all players and their chip counts
        if agents is not None:
            print("\n  [Players]")
            for agent in agents:
                tag = " <- YOU" if agent is self else ""
                print(f"    {getattr(agent, 'name', 'Agent')}: {agent.chips} chips{tag}")

        # Show the user their own hand
        print(f"\n  [Your Hand]")
        self.hand.ShowHand()

        # Show what the current bet floor is
        print(f"\n  Current bet to call: {currBet} chips")
        print(f"  Your chips:          {self.chips} chips")

        # ---- Input Loop ----
        # Keep asking until the user gives a valid action
        while True:
            print("\n  Actions: [c]heck | [f]old | [call] | [b]et <amount>")
            raw = input("  > ").strip().lower()
            tokens = raw.split()

            if not tokens:
                print("  Please enter an action.")
                continue

            command = tokens[0]

            if command == "c" or command == "check":
                if currBet > 0:
                    # Can't check if there's an outstanding bet — must call or fold
                    print(f"  You can't check — there's a bet of {currBet}. Call or fold instead.")
                    continue
                print("  You check.")
                return game.Action(check=1, fold=0, call=0, bet=0)

            elif command == "f" or command == "fold":
                print("  You fold.")
                return game.Action(check=0, fold=1, call=0, bet=0)

            elif command == "call":
                if currBet == 0:
                    print("  Nothing to call — use 'check' instead.")
                    continue
                if self.chips < currBet:
                    print(f"  Not enough chips to call! You have {self.chips}, need {currBet}.")
                    print("  You can fold instead.")
                    continue
                print(f"  You call {currBet}.")
                return game.Action(check=0, fold=0, call=1, bet=currBet)

            elif command == "b" or command == "bet":
                if len(tokens) < 2:
                    print("  Specify an amount, e.g. 'bet 50'")
                    continue
                try:
                    amount = int(tokens[1])
                except ValueError:
                    print("  Bet amount must be a whole number.")
                    continue
                if amount <= currBet:
                    print(f"  Bet must be greater than the current bet ({currBet}).")
                    continue
                if amount > self.chips:
                    print(f"  You only have {self.chips} chips!")
                    continue
                print(f"  You bet {amount}.")
                return game.Action(check=0, fold=0, call=0, bet=amount)

            else:
                print(f"  Unknown action '{command}'. Try: check, fold, call, bet <amount>")


VALUE_MAP = {v: i for i, v in enumerate(game.values)}  # "2"->0, "A"->12

def BestFiveOmaha(playerHand, tableHand):
    """
    Omaha rule: exactly 2 cards from hole, exactly 3 from board.
    Tries all C(4,2) * C(5,3) = 60 combinations and returns the best score.
    """
    from itertools import combinations
    
    best = None
    for hole in combinations(playerHand.cards, 2):
        for board in combinations(tableHand.cards, 3):
            score = EvaluateFive(list(hole) + list(board))
            if best is None or score > best:
                best = score
    # print(f"Best hand is {best}")
    return best

def EvaluateFive(cards):
    """
    Scores a 5-card hand as a tuple — tuples compare element by element in Python,
    so (6, 14, ...) beats (5, ...) automatically. Higher is better.
    
    Returns: (rank, *tiebreakers)
      8 = Straight flush
      7 = Four of a kind
      6 = Full house
      5 = Flush
      4 = Straight
      3 = Three of a kind
      2 = Two pair
      1 = One pair
      0 = High card
    """
    vals = sorted([VALUE_MAP[c.value] for c in cards], reverse=True)
    suits_in_hand = [c.suit for c in cards]
    
    is_flush = len(set(suits_in_hand)) == 1
    
    # Straight detection (also handles A-2-3-4-5 wheel)
    is_straight = (vals[0] - vals[4] == 4 and len(set(vals)) == 5)
    wheel = (vals == [12, 3, 2, 1, 0])  # A-2-3-4-5
    if wheel:
        is_straight = True
        vals = [3, 2, 1, 0, -1]  # treat ace as low
    
    if is_straight and is_flush:
        return (8,) + tuple(vals)
    
    # Count occurrences of each value
    from collections import Counter
    counts = Counter(vals)
    # Sort by count descending, then value descending (for tiebreakers)
    groups = sorted(counts.items(), key=lambda x: (x[1], x[0]), reverse=True)
    group_counts = [g[1] for g in groups]
    group_vals  = [g[0] for g in groups]
    
    if group_counts[0] == 4:
        return (7,) + tuple(group_vals)
    if group_counts[:2] == [3, 2]:
        return (6,) + tuple(group_vals)
    if is_flush:
        return (5,) + tuple(vals)
    if is_straight:
        return (4,) + tuple(vals)
    if group_counts[0] == 3:
        return (3,) + tuple(group_vals)
    if group_counts[:2] == [2, 2]:
        return (2,) + tuple(group_vals)
    if group_counts[0] == 2:
        return (1,) + tuple(group_vals)
    return (0,) + tuple(vals)


def CompareHands(agents, isIn, tableHand):
    """
    Finds the winner(s) among active players using Omaha hand rules.
    Returns a list of winners (multiple on a tie).
    """
    best_score = None
    winners = []
    
    for agent in agents:
        if not isIn[agent]:
            continue
        score = BestFiveOmaha(agent.hand, tableHand)
        if best_score is None or score > best_score:
            best_score = score
            winners = [agent]
        elif score == best_score:
            winners.append(agent)  # split pot
    
    return winners


def _PreFlopStrength(hand):
    """
    Estimates hole-card strength before any community cards are dealt.

    Scoring rubric (all values additive, capped at 1.0):
      - High-card baseline:  scales from 0.0 (all 2s) to 0.25 (all Aces)
      - Pair among hole cards: +0.40  (pairs are strong pre-flop starters)
      - Two or more suited:    +0.10  (flush potential)
      - Connected (gap ≤ 2):  +0.10  (straight potential)
    
    Omaha players are dealt 4 hole cards and must use exactly 2, so
    we reward having multiple 'synergies' among the four cards.
    """
    from collections import Counter
    cards = hand.cards
    if not cards:
        return 0.0

    vals  = [VALUE_MAP[c.value] for c in cards]
    suits_list = [c.suit for c in cards]

    score = 0.0

    # High-card component: average card value, scaled 0–0.25
    score += (sum(vals) / len(vals)) / 12.0 * 0.25

    # Pair bonus
    if max(Counter(vals).values()) >= 2:
        score += 0.40

    # Suited bonus (≥2 cards share a suit → flush draw possible)
    if max(Counter(suits_list).values()) >= 2:
        score += 0.10

    # Connectedness bonus (any two cards within 2 pips → straight draw)
    sorted_vals = sorted(set(vals))
    for i in range(len(sorted_vals) - 1):
        if sorted_vals[i + 1] - sorted_vals[i] <= 2:
            score += 0.10
            break  # Count at most once

    return min(score, 1.0)


def _PostFlopStrength(hand, table):
    """
    Uses BestFiveOmaha to get the true best-hand rank (0–8),
    then normalises it to [0, 1] so it's comparable to _PreFlopStrength.

    Rank meanings (from EvaluateFive):
      0=High card, 1=Pair, 2=Two pair, 3=Trips,
      4=Straight, 5=Flush, 6=Full house, 7=Quads, 8=Straight flush
    
    Dividing by 8 gives a rough probability-like score: a flush (5/8 = 0.625)
    will trigger more aggressive play than a pair (1/8 = 0.125).
    """
    score = BestFiveOmaha(hand, table.hand)
    return score[0] / 8.0   # Normalise rank to [0, 1]


class SAgent(Agent):
    """
    Simple Agent — makes rule-based decisions driven by estimated hand strength
    and basic pot-odds reasoning.

    Decision framework
    ------------------
    1. Estimate hand strength (0.0 = weakest, 1.0 = strongest).
       - Pre-flop  (< 3 board cards): use hole-card heuristics.
       - Post-flop (≥ 3 board cards): use the full Omaha evaluator.

    2. Compute pot odds — the fraction of the total pot we must invest to call:
           pot_odds = currBet / (pot + currBet)
       If our estimated win probability (hand strength) exceeds pot_odds by a
       comfortable margin, calling (or raising) is profitable in expectation.

    3. Act:
       - No bet to face  → strong hand: value-bet; weak hand: check.
       - Bet to face     → strong hand: raise; medium hand: call; weak: fold.

    Thresholds (tunable class constants):
       BET_THRESHOLD   – minimum strength needed to open-bet (no prior bet)
       CALL_MARGIN     – how much strength must EXCEED pot_odds to justify a call
       RAISE_THRESHOLD – minimum strength to raise instead of merely calling
       BET_FRACTION    – fraction of chips to bet when value-betting
    """

    BET_THRESHOLD   = 0.55   # Must be > 55% strong to open-bet
    CALL_MARGIN     = 0.15   # Strength must beat pot_odds by 15 pp to call
    RAISE_THRESHOLD = 0.50   # Must be > 70% strong to raise
    BET_FRACTION    = 0.20   # Open-bet sizing: 20% of current stack

    def __init__(self, chips, name):
        super().__init__(chips, name)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _HandStrength(self, table):
        """Dispatches to pre-flop or post-flop evaluator based on board size."""
        if table is None or len(table.hand.cards) < 3:
            return _PreFlopStrength(self.hand)
        return _PostFlopStrength(self.hand, table)

    def _PotOdds(self, table, currBet):
        """
        Returns the fraction of (pot + call) that the call represents.
        A pot of 100 with a call of 20 → pot_odds = 20/120 ≈ 0.167.
        We need a win probability > 0.167 to make a call break-even.
        """
        pot = table.chips if table else 0
        total = pot + currBet
        return currBet / total if total > 0 else 0.0

    def _BetSize(self, floor=0):
        """
        Calculates a bet size as a fraction of our stack, ensuring it's
        always larger than any outstanding bet (the 'floor').
        """
        amount = max(int(self.chips * self.BET_FRACTION), floor + 1)
        return min(amount, self.chips)  # Never bet more than we have

    # ------------------------------------------------------------------
    # Core decision logic
    # ------------------------------------------------------------------

    def Action(self, table=None, agents=None, currBet=0):
        strength  = self._HandStrength(table)
        pot_odds  = self._PotOdds(table, currBet)

        # ---- No outstanding bet: check or open-bet ----
        if currBet == 0:
            if strength >= self.BET_THRESHOLD and self.chips > 0:
                amount = self._BetSize(floor=0)
                return game.Action(check=0, fold=0, call=0, bet=amount)
            return game.Action(check=1, fold=0, call=0, bet=0)

        # ---- There is a bet to face ----
        # Does our strength justify involvement?
        if strength >= pot_odds + self.CALL_MARGIN:
            # Strong enough to raise?
            if strength >= self.RAISE_THRESHOLD:
                amount = self._BetSize(floor=currBet)
                if amount > currBet and self.chips >= amount:
                    return game.Action(check=0, fold=0, call=0, bet=amount)
            # Otherwise just call
            if self.chips >= currBet:
                return game.Action(check=0, fold=0, call=1, bet=currBet)

        # Strength didn't justify continuing — fold
        return game.Action(check=0, fold=1, call=0, bet=0)