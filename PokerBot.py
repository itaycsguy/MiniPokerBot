##version 0.6
##developers: Itay Guy & Din Sharaby
import random,math,os,sys
import numpy as np
import itertools
import operator
import json
import copy
import time
import pickle
from collections import Counter
import matplotlib.pyplot as plt

class logger:
    DEBUG = 0
    INFO = 1
    WARNING = 2
    CRITICAL = 3

    @staticmethod
    def log(level,*args):
        if (level >= 0):
            print (*args)

class Card:
  RANKS = (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14)
  SUITS = (u"\u2665", u"\u2666", u"\u2663", u"\u2660")

  def __init__ (self, rank, suit):
    self.rank = rank
    self.suit = suit

  def isEqual(self,cards):
      for card in cards:
          if card.rank == self.rank and card.suit == self.suit:
              return True
      return False

  def __repr__(self):
    return self.__str__()

  def __str__ (self):
    if self.rank == 14:
      rank = 'A'
    elif self.rank == 13:
      rank = 'K'
    elif self.rank == 12:
      rank = 'Q'
    elif self.rank == 11:
      rank = 'J'
    else:
      rank = self.rank
    return str(rank) + self.suit

  def __eq__ (self, other):
    return (self.rank == other.rank)

  def __ne__ (self, other):
    return (self.rank != other.rank)

  def __lt__ (self, other):
    return (self.rank < other.rank)

  def __le__ (self, other):
    return (self.rank <= other.rank)

  def __gt__ (self, other):
    return (self.rank > other.rank)

  def __ge__ (self, other):
    return (self.rank >= other.rank)

class Dealer:
    def __init__(self):
        self.deck = []
        for suit in Card.SUITS:
            for rank in Card.RANKS:
                card = Card (rank, suit)
                self.deck.append(card)

    # to reduce the amount and to sample from smaller sub-space
    def removeCards(self,cards):
        idx2remove = 0
        for idx,card in enumerate(self.deck):
            if card.rank == cards[0].rank and card.suit == cards[0].suit:
                idx2remove = idx
        del self.deck[idx2remove]
        for idx,card in enumerate(self.deck):
            if card.rank == cards[1].rank and card.suit == cards[1].suit:
                idx2remove = idx
        del self.deck[idx2remove]


    def shuffle (self):
        random.shuffle (self.deck)

    def __len__ (self):
        return len(self.deck)

    def deal(self,count = 1):
        if count == 1:
            return self.deck.pop(0)

        cards = []
        if len(self) >= count:
            for i in range(count):
                cards.append(self.deck.pop(0))
        return cards

class Player:
	id = 0
	money = 0
	cards = []
	dividend = 0
	

	def __init__(self,id):
		self.id = id
		
	def withdrawAllMoneyByLimit(self,limit):
		m = min(self.money,limit)
		self.money -= m
		return m

	def deal(self,cards):
		self.cards = cards

	def getMoney(self):
		return self.money
		
	def setDividends(self,pot):
		self.dividend = pot
		if pot > 0:
			self.money += pot

	def getDividends(self):
		return self.dividend

class Poker:
    HANDS = ('High Card','Pair','Two pair','Three of a kind','Straight','Flush','Full house','Four of a kind','Straight flush','Royal flush')
    FOLD  = 0
    ALLIN = 1
    SBTURN = 0
    BBTURN = 1
    ENDROUND = 2
    GAMEOVER = 3
    TOTAL_ENTRYBET = 1.5
    TOTAL_CASH = 10

    def __init__(self):
        self.players = []

	##control the game
    def addPlayer(self):
        id = len(self.players)
        self.players.append(Player(id))
        return id

    def reset(self):
        for player in self.players:
            player.money = Poker.TOTAL_CASH

        self._isFinishRound  = False
        self.sb = random.randint(0,1)
        self.bb = int(not self.sb)
        self.roundWinners = []

    def sampleSubspace(self,state,playerId):
        state0 = state.getMultiIndex(random.choice(list(state.getStateSeeds().keys())))
        r1,r2 = random.sample(Card.SUITS,2)
        if state0[2] == 1:
            r2 = r1
        cards = [Card(state0[0]+2,r1),Card(state0[1]+2, r2)]
        self.players[playerId].deal(cards)
        return cards


    def deal(self,states0=None,states1=None):
        self.dealer = Dealer()
        self.dealer.shuffle()
        self.sb = (self.sb +1) % len(self.players)
        self.bb = (self.sb +1) % len(self.players)
        self.effetivePot = min(self.players[0].getMoney(),self.players[1].getMoney())
        logger.log(logger.INFO,"player 1: ", self.players[0].cards)        
        logger.log(logger.INFO,"player 2: ", self.players[1].cards)     
        self._gameState = 0
        self.winnersIDs = []
        self.pot = self.players[self.sb].withdrawAllMoneyByLimit(0.5) + self.players[self.bb].withdrawAllMoneyByLimit(1.0)
        ## getting into this case while the QAgent playing at test game with sub-states to deal from
        if not states0 and not states1:
            self.players[self.sb].deal(sorted(self.dealer.deal(2), key=operator.attrgetter('rank')))
            self.players[self.bb].deal(sorted(self.dealer.deal(2), key=operator.attrgetter('rank')))
        else:                
            if states0:
                cards = self.sampleSubspace(states0,0)
                self.dealer.removeCards(cards)
            if states1:
                self.sampleSubspace(states1,1)
            else:
                cards = self.dealer.deal(2)
                self.players[1].deal(sorted(cards, key=operator.attrgetter('rank')))


    def setAction(self,action):
        prev_pot = self.pot
        if (self._gameState == Poker.SBTURN):
            if (action == Poker.FOLD):
                self.winnersIDs = [self.bb]
                logger.log(logger.INFO,"player",self.sb+1,"makes fold")
                self.setEndStatus() 
            elif (action == Poker.ALLIN):
                self.pot += self.players[self.sb].withdrawAllMoneyByLimit(self.players[self.bb].getMoney())
                logger.log(logger.INFO,"player",self.sb+1,"makes allin")
                self._gameState = Poker.BBTURN
            else:
                logger.log(logger.INFO,'Unknown action... try again. pot: ' + str(self.pot))
        elif (self._gameState == Poker.BBTURN):
            if (action == Poker.FOLD):
                self.winnersIDs = [self.sb]
                logger.log(logger.INFO,"player",self.bb+1,"makes fold")
                self.setEndStatus()
            elif (action == Poker.ALLIN):
                self.pot += self.players[self.bb].withdrawAllMoneyByLimit(self.pot - Poker.TOTAL_ENTRYBET)
                self.showDown()
                logger.log(logger.INFO,"player",self.bb+1,"makes allin")
                self.setEndStatus()
            else:
                logger.log(logger.CRITICAL,'Unknown action... try again. pot: ')
        else:
            logger.log(logger.CRITICAL,'Unknown action... try again. pot: ')
        return ((self.pot - prev_pot)/max(1.0,max(self.pot,prev_pot)))

	###internal functions#####
    def setEndStatus(self):
        if len(self.winnersIDs) == self.getPlayersCount():
            self.players[0].setDividends(self.pot / 2)
            self.players[1].setDividends(self.pot / 2)
        elif self.winnersIDs[0] == 0:
            self.players[0].setDividends(self.pot)
            self.players[1].setDividends(-self.pot)
        elif self.winnersIDs[0] == 1:
            self.players[0].setDividends(-self.pot)
            self.players[1].setDividends(self.pot)
        else:
            raise Exception("very bad for you")

        if self.players[self.bb].getMoney() < 0.5 or self.players[self.sb].getMoney() < 1.0:
            self._gameState = Poker.GAMEOVER
        else:
            self._gameState = Poker.ENDROUND

        self.roundWinners.append(self.winnersIDs)


    def showDown(self):
        flop = self.dealer.deal(5)
        logger.log(logger.WARNING,"flop: {}".format(flop))
        sbhand,sbscore,sbrank = self.getBestHandByScore(flop + self.players[self.sb].cards)
        bbhand,bbscore,bbrank = self.getBestHandByScore(flop + self.players[self.bb].cards)

        if (sbscore > bbscore):
            self.winnersIDs =  [self.sb]
        elif (sbscore < bbscore):
            self.winnersIDs = [self.bb]
        else:
            self.winnersIDs = [self.sb,self.bb]

	##get game info
    def getplayerIDTurn(self):
        if (self._gameState == Poker.BBTURN):
            return self.bb
        else:
            return self.sb

    def getPlayersCount(self):
        return len(self.players)

    def printStatus(self):
        logger.log(logger.INFO,"gameover:",self.roundWinners)

    def getPot(self):
        return self.pot

    def getWinnerIDs(self):
        return self.winnersIDs

    def getPlayerState(self,id):
        return (self.players[id].cards[0].rank,
                self.players[id].cards[1].rank,
                int(self.players[id].cards[0].suit == self.players[id].cards[1].suit),
                self.players[id].getMoney(),
                int(self.sb == id))

    def isRoundFinished(self):
        return self._gameState == Poker.ENDROUND

    def isGameover(self):
        return self._gameState == Poker.GAMEOVER

    def getRoundDividends(self,id):
        return self.players[id].getDividends()

    def getBestHandByScore(self,largehand):
        hands = list(itertools.combinations(largehand,5))
        scores = []
        ranks = []
        for hand in hands:
            rank,score = self.getHandScore(hand)
            scores.append(score)
            ranks.append(rank)
        index = scores.index(max(scores))
        return hands[index],scores[index],ranks[index]

    def getHandScore(self,hand):
        sortedHand=sorted(hand,reverse=True)
        ranksCount = {}
        for card in sortedHand:
            if(card.rank not in ranksCount.keys()):
                ranksCount[card.rank] = 0
            ranksCount[card.rank] += 1

        score = self.isRoyal(sortedHand,ranksCount)
        finalscore = 0
        zeroes = 10
        for i in score:
            finalscore += i * (10**zeroes)
            zeroes -= 2

        return (self.HANDS[score[0]],finalscore)

    def isRoyal(self,hand,ranksCount):
        flushSuit = hand[0].suit #d
        currRank = 14
        
        for card in hand:
            if card.suit == flushSuit and card.rank == currRank:
                currRank -= 1
            else:
                return self.isStraightFlush(hand,ranksCount)
        return (9,currRank+1)

    def isStraightFlush(self,hand,ranksCount):
        flushSuit = hand[0].suit #d
        currRank = hand[0].rank #14
        
        for card in hand:
            if card.suit == flushSuit and card.rank == currRank:
                currRank -= 1
            else:
                return self.isFour(hand,ranksCount)
        return (8,currRank+1)

    def isFour(self,hand,ranksCount):
        ranks = list(ranksCount.keys())
        if (ranksCount[ranks[0]] == 4):
            return (7,ranks[0],ranks[1])
        
        if (ranksCount[ranks[1]] == 4):
            return (7,ranks[1],ranks[0])

        return self.isFull(hand,ranksCount)

    def isFull(self,hand,ranksCount):
        ranks =  sorted(ranksCount, key=ranksCount.get,reverse=True)
        if (ranksCount[ranks[0]] == 3 and ranksCount[ranks[1]] == 2):
            return (6,ranks[0],ranks[1])

        return self.isFlush(hand,ranksCount)

    def isFlush(self,hand,ranksCount):
        flushSuit = hand[0].suit #d
        
        for card in hand:
            if card.suit != flushSuit:
                return self.isStraight(hand,ranksCount)

        return (5,hand[0].rank,hand[1].rank,hand[2].rank,hand[3].rank,hand[4].rank)

    def isStraight(self,hand,ranksCount):
        currRank = hand[0].rank
        
        for card in hand:
            if card.rank == currRank:
                currRank -= 1
            else:
                return self.isThree(hand,ranksCount)
        return (4,currRank+1)

    def isThree(self,hand,ranksCount):
        ranks =  sorted(ranksCount, key=ranksCount.get,reverse=True)
        if (ranksCount[ranks[0]] == 3):
            return (3,ranks[0],ranks[1],ranks[2])

        return self.isTwo(hand,ranksCount)
     
    def isTwo(self,hand,ranksCount):
        ranks =  sorted(ranksCount, key=ranksCount.get,reverse=True)
        if (ranksCount[ranks[0]] == 2 and ranksCount[ranks[1]] == 2 ):
            return (2,ranks[0],ranks[1],ranks[2])

        return self.isOne(hand,ranksCount)
     
    def isOne(self,hand,ranksCount):
        ranks =  sorted(ranksCount, key=ranksCount.get,reverse=True)
        if (ranksCount[ranks[0]] == 2):
            return (1,ranks[0],ranks[1],ranks[2],ranks[3])

        return self.isHigh(hand,ranksCount)
   
    def isHigh(self,hand,ranksCount):
        ranks =  sorted(ranksCount, key=ranksCount.get,reverse=True)
        return (0,ranks[0],ranks[1],ranks[2],ranks[3],ranks[4])

## modeling the States of QAgent
class States():
    BACKTRACK_PENALTY = -0.5
    BACKTRACK_REWARD = 0.5
    WIN = "win"
    LOSE = "lose"
    COLOR_INTERVAL = 2
    BLIND_INTERVAL = 2
    CARDS_RANGE = 13
    POOR_PERCENTILE = 0.10
    ORD = 0
    ONE_HIGH = 1
    TWO_HIGHS = 2
    PAIR = 3
    HIGH_PAIR = 4
    HIGH_CARD = 8

    def getMultiIndex(self,index):
        return np.unravel_index(index,self.STATE_STRUCT)

    def getLinearIndex(self,indices):
        quan_amt = self.getQuantizeAmount(indices[3])
        return np.ravel_multi_index([indices[0]-2, indices[1]-2, indices[2],quan_amt,indices[4]],self.STATE_STRUCT)

    # V = (card1,card2,color/not color = {1,0},pot = [0-40],SB/BB = {1,0}) -> A = (allin = rewards,fold = rewards)
    def _init_qtable(self,enableLearning,agentId):
        start = time.time()
        time.clock()
        table = list()
        seeds = {}
        totalPercentile = 0.0
        if enableLearning:
            ## creating a flat table to use it as linear and faster functionality
            table = [np.zeros(States.CARDS_RANGE * States.CARDS_RANGE * States.COLOR_INTERVAL * self.POT_INTERVAL * States.BLIND_INTERVAL),np.zeros(States.CARDS_RANGE * States.CARDS_RANGE * States.COLOR_INTERVAL * self.POT_INTERVAL * States.BLIND_INTERVAL)]
            self._initNextStates()
        else:
            print("reading QTable..")
            path = str(os.path.dirname(str(os.path.abspath(__file__))))
            try:
                table = np.load(path + "\\Qtable.npy")
            except:
                logger.log(logger.WARNING,"file " + path + "\\Qtable.npy does not exist!")
                sys.exit(1)
            with open(path + "\\QstateSeeds.npy","rb") as f:
                seeds = pickle.load(f)
            try:
                with open(path + "\\Qcash.npy","rb") as f:
                    cash = int(pickle.load(f))
                    if cash != Poker.TOTAL_CASH:
                        logger.log(logger.WARNING,"enter the correct cash amount [" + str(cash) + "].")
                        sys.exit(1)
            except:
                if not os.path.exists(path + "\\Qcash.npy"):
                    logger.log(logger.WARNING,"file " + path + "\\Qcash.npy does not exist!")
                sys.exit(1)
            for key in seeds.keys():
                totalPercentile += (table[Poker.ALLIN][key] + table[Poker.FOLD][key]) / 2.0
        print("completed within: ",time.time() - start,"seconds")
        return (table,seeds,totalPercentile)

    def __init__(self,enableLearning,agentId):
        self.POT_INTERVAL = 4 * Poker.TOTAL_CASH + 1
        self.STATE_STRUCT = (States.CARDS_RANGE,States.CARDS_RANGE,States.COLOR_INTERVAL,self.POT_INTERVAL,States.BLIND_INTERVAL)
        self._idxListSB = list()
        self._idxListBB = list()
        self._qtable,self._stateSeeds,self.totalPercentile = self._init_qtable(enableLearning,agentId)
        self.seedsOrder = 1
        self._state = list()
        self._stateIndex = -1
        self._alpha = 0.2
        self._gamma = 0.01
        self._epsilon = 0.1
        self._finalState = ""
        self._R_simple = {States.WIN : 1.0,States.LOSE : -1.0}
        self._R_high = {States.WIN : 2.0,States.LOSE : -1.0}
        self._R_twoHighs = {States.WIN : 3.0,States.LOSE : -1.0}
        self._R_pair = {States.WIN : 6.0,States.LOSE : -2.0}
        self._R_color = {States.WIN : 6.0,States.LOSE : -2.0}
        self._R_highPair = {States.WIN : 12.0,States.LOSE : -3.0}
        self._R_highColor = {States.WIN : 12.0,States.LOSE : -3.0}
        self._penalties = list()

    ##get methods
    ##computing the expected action against all same hands which is changing their pot amount
    def getExpectedAction(self,state):
        avgAllin = 0.0
        avgFold = 0.0
        counter = 0
        for key in self._stateSeeds.keys():
            alterState = self.getMultiIndex(key)
            if alterState[0] == state[0] and alterState[1] == state[1] and \
               alterState[2] == state[2] and alterState[4] == state[4]:
                counter += 1
                weigth = 1
                if alterState[3] == state[3]:
                    weigth = 2
                avgAllin += weigth * self._qtable[Poker.ALLIN][key]
                avgFold += weigth * self._qtable[Poker.FOLD][key]
        if counter == 0:
            counter += 1
        return (((avgAllin / counter) >= (avgFold / counter)),(avgAllin + avgFold) / (2 * counter))


    def getTotalPercentile(self):
        return self.totalPercentile

    def getFinalState(self):
        return self._finalState

    def getQTable(self):
        return self._qtable

    def getEpsilon(self):
        return self._epsilon

    def getStateSeeds(self):
        return self._stateSeeds

    def getStateIndex(self):
        return self._stateIndex

    def getQuantizeAmount(self,amount):
        return int(2 * amount)

    def getSumPenalties(self):
        return sum(self._penalties)

    ##set methods
    def appendOnce(self,state,action):
        stateInd = self.getLinearIndex(state)
        self._stateSeeds[stateInd] = (self.seedsOrder,action)
        self.seedsOrder += 1

    def updateAllGameRounds(self,isReward,rounds):
        weight = States.BACKTRACK_PENALTY
        if isReward:
            weight = States.BACKTRACK_REWARD
        sortedSeeds = sorted(self._stateSeeds.keys(),reverse=True)
        idx = 0
        for key,value in zip(self._stateSeeds.keys(),self._stateSeeds.values()):
            if key in sortedSeeds:
                if idx == rounds:
                    break
                action = value[1]
                self._qtable[action][i] += weight
                idx += 1


    def decreaseGamma(self,epochs):
        if self._gamma > 0 and epochs > 0:
            self._gamma -= 1/epochs

    def setCurrentState(self,state):
        self._state = state

    def setStateIndex(self,stateIdx):
        self._stateIndex = stateIdx

    def setFinalState(self,final):
        self._finalState = final

    def _isOneHigh(self,state):
        return True if state[0] > States.HIGH_CARD or state[1] > States.HIGH_CARD else False

    def _isTwoHigh(self,state):
        return True if state[0] > States.HIGH_CARD and state[1] > States.HIGH_CARD else False

    def _isPair(self,state):
        return True if state[0] == state[1] else False

    def _isHighPair(self,state):
        return True if self._isPair(state) and self._isOneHigh(state) else False

    def getCurrCardsStatus(self):
        state = self.getMultiIndex(self._stateIndex)
        if self._isOneHigh(state):
            return States.ONE_HIGH
        elif self._isTwoHigh(state):
            return States.TWO_HIGHS
        elif self._isPair(state):
            return States.PAIR
        elif self._isHighPair(state):
            return States.HIGH_PAIR
        else:
            return States.ORD

    def setStateReward(self,action,weight):
        cardsStatus = self.getCurrCardsStatus()
        
        ##initialize
        if self.getFinalState() == States.WIN:
            R = self._R_simple[States.WIN]
        else:
            R = self._R_simple[States.LOSE]

        ##cases for better diversity between different states by action and its result
        if cardsStatus == States.ONE_HIGH:
            if self._finalState == States.LOSE and action == Poker.ALLIN:
                self._penalties.append(self._R_high[States.LOSE])
                R = self._R_high[States.LOSE]
            elif self._finalState == States.LOSE:
                R = self._R_simple[States.WIN]
            elif action == Poker.ALLIN:
                R = self._R_high[States.WIN]


        elif cardsStatus == States.TWO_HIGHS:
            if self._finalState == States.LOSE and action == Poker.ALLIN:
                self._penalties.append(self._R_twoHighs[States.LOSE])
                R = self._R_twoHighs[States.LOSE]
            elif self._finalState == States.LOSE:
                R = self._R_simple[States.WIN]
            elif action == Poker.ALLIN:
                R = self._R_twoHighs[States.WIN]

        elif cardsStatus == States.PAIR:
            if self._finalState == States.LOSE and action == Poker.ALLIN:
                self._penalties.append(self._R_pair[States.LOSE])
                R = self._R_pair[States.LOSE]
            elif self._finalState == States.LOSE:
                R = self._R_simple[States.WIN]
            elif action == Poker.ALLIN:
                R = self._R_pair[States.WIN]

        elif cardsStatus == States.HIGH_PAIR:
            if self._finalState == States.LOSE and action == Poker.ALLIN:
                self._penalties.append(self._R_highPair[States.LOSE])
                R = self._R_highPair[States.LOSE]
            elif self._finalState == States.LOSE:
                R = self._R_simple[States.WIN]
            elif action == Poker.ALLIN:
                R = self._R_highPair[States.WIN]

        elif cardsStatus == States.ORD:
            if self._finalState == States.LOSE and action == Poker.ALLIN:
                self._penalties.append(self._R_simple[States.LOSE])
                R = self._R_simple[States.LOSE]
            elif self._finalState == States.LOSE or action == Poker.ALLIN:
                R = self._R_simple[States.WIN]

        ## reward/penalty update function
        self._qtable[action][self._stateIndex] = (1.0 - self._alpha) * self._qtable[action][self._stateIndex] + \
                                                 self._alpha * (R*weight + self._gamma * self._getNextStateExpectedValue(self._state) - \
                                                 self._qtable[action][self._stateIndex])


    ##private methods
    def _initNextStates(self):
        for j0 in range(2,States.CARDS_RANGE + 2):
            for j1 in range(2,States.CARDS_RANGE + 2):
                for j2 in range(States.COLOR_INTERVAL):
                    for j3 in range(self.POT_INTERVAL - Poker.TOTAL_CASH * 2):
                        self._idxListSB.append(self.getLinearIndex([j0,j1,j2,j3,0]))
                        self._idxListBB.append(self.getLinearIndex([j0,j1,j2,j3,1]))


    ## computing the expected value for the next state which is SB/BB
    def _getNextStateExpectedValue(self,state):
        idxList = self._idxListBB if state[4] == 0 else self._idxListSB
        return (sum(self._qtable[0][idxList]) + sum(self._qtable[1][idxList]) + 1) / (2.0 * len(idxList) + 1)

    ##boolean methods
    def isEqual(self,state0,state1):
        return (self.getLinearIndex(state0) == self.getLinearIndex(state1))

    ##maintenance methods
    def saveData(self,agentId):
        path = str(os.path.dirname(str(os.path.abspath(__file__))))
        np.save(path + "\\Qtable.npy" ,self._qtable, allow_pickle=True, fix_imports=True)
        with open(path + "\\QstateSeeds.npy","wb") as f:
            pickle.dump(self._stateSeeds,f,pickle.HIGHEST_PROTOCOL)
        with open(path + "\\Qcash.npy","wb") as f:
            pickle.dump(Poker.TOTAL_CASH,f,pickle.HIGHEST_PROTOCOL)

## an interface to all agents
class Agent:
    def __init__(self,id,epochs):
        self.loc = 0
        self.wins = np.zeros(epochs)
        self.id = id
        self.gameWins = 0
        self.seriesWinsGraph = list()
        self.roundsPerGame = 0

    def setWin(self,num):
        if self.loc < len(self.wins):
            self.wins[self.loc] = num
            self.loc += 1
            self.roundsPerGame += 1

    def backtrackUpdateStates(self,isReward):
        pass

    def updateGameWins(self):
        self.gameWins += 1
        self.seriesWinsGraph.append(1)
        self.roundsPerGame = 0

    def updateGameLose(self):
        self.seriesWinsGraph.append(0)
        self.roundsPerGame = 0

    def plotSeriesWinsGraph(self,agentClass):
        pass

    def getGameWins(self):
        return self.gameWins

    def getWins(self):
        count = 0
        for i in self.wins:
            if i == 1:
                count += 1
        return count

    def getEpochs(self):
        count = 0
        for i in self.wins:
            if i == 1 or i == 0:
                count += 1
        return count

    def evalAct(self,state,enableLearning):
        pass

    def setReward(self,weight):
        pass

    def setStatus(self,statusString):
        pass
    
    def collect(self,state,action):
        pass

    def discount(self,epochs):
        pass

    def updateIfBluff(self,state):
        pass

    def getStatus(self):
        pass

    def getStatesObj(self):
        pass

    def getId(self):
        return self.id

    def save(self):
        pass

    def getAgentClass(self):
        return __class__.__name__

class RandomAgent(Agent):
    def __init__(self,id,epochs):
        super().__init__(id,epochs)
        self.finalStatus = ""

    def evalAct(self,state,enableLearning):
        return random.randint(0,1)

    def getStatesObj(self):
        return None

    def setStatus(self,statusString):
        self.finalStatus = statusString
        if statusString == States.WIN:
            self.setWin(1)
        else:
            self.setWin(0)

    def getStatus(self):
        return self.finalStatus

    def getAgentClass(self):
        return __class__.__name__

## outer player to play against the QAgent
class PlayerAgent(Agent):
    def evalAct(self,state,enableLearning):
        return getInput("choose action",['Fold','All-in'])

    def get_pretty_table(iterable, header):
        max_len = [len(x) for x in header]
        for row in iterable:
            row = [row] if type(row) not in (list, tuple) else row
            for index, col in enumerate(row):
                if max_len[index] < len(str(col)):
                    max_len[index] = len(str(col))
        output = ''
        output += '\t' + ''.join([h + ' ' * (l - len(h)) + '\t' for h, l in zip(header, max_len)]) + '\n'
        for row in iterable:
            row = [row] if type(row) not in (list, tuple) else row
            output += '\t' + ''.join([str(c) + ' ' * (l - len(str(c))) + '\t' for c, l in zip(row, max_len)]) + '\n'
        return output

    def getStatesObj(self):
        return None

    def getAgentClass(self):
        return __class__.__name__

     
class QAgent(Agent):
    def __init__(self,id,epochs,enableLearning):
        super().__init__(id,epochs)
        self._states = States(enableLearning,id)
        self.waitingForReward = False
        self.action = 0
        self._agentId = id
        self.bluffs = 0

    ##set methods
    ## evaluate the action should be taken
    def evalAct(self,state,enableLearning):
        self._states.setCurrentState(state)
        stateIdx = self._states.getLinearIndex(state)
        self._states.setStateIndex(stateIdx)
        qtable = self._states.getQTable()
        if not enableLearning:
            self.action = self._states.getExpectedAction(state)
            self.action = self.action[0]
        else:
            allinVal = qtable[Poker.ALLIN][stateIdx]
            foldVal = qtable[Poker.FOLD][stateIdx]
            self.action = (allinVal >= foldVal)
            if (np.random.rand() < self._states.getEpsilon()):
                self.action = not self.action
        self.action = int(self.action)
        self.waitingForReward = True
        return self.action

    ## update as backtracking all states who were piece of this game to winning
    def backtrackUpdateStates(self,isReward):
        self._states.updateAllGameRounds(isReward,self.roundsPerGame)

    def setStatus(self,statusString):
        if statusString == States.WIN:
            self.setWin(1)
        else:
            self.setWin(0)
        self._states.setFinalState(statusString)

    def plotSeriesWinsGraph(self,agentClass):
        accumulatedAxis = list()
        for counter,item in enumerate(self.seriesWinsGraph):
            if item == 1:
                counter += 1
                accumulatedAxis.append(counter)
        if accumulatedAxis:
            plt.plot(range(len(accumulatedAxis)),accumulatedAxis,linewidth=3,markersize=1)
            plt.xlabel('Wins')
            plt.ylabel('Epochs')
            plt.title('QAgent Improvement Curve')
            plt.grid(True)
            logger.log(logger.WARNING,agentClass + " plotting learning process graph..")
            plt.show()

    ## collect the state to use it in the game test mode
    def collect(self,state,action):
        self._states.appendOnce(state,action)

    def setReward(self,weight):
        if not self.waitingForReward:
            return
        self.waitingForReward = False
        self._states.setStateReward(self.action,weight)

    def discount(self,epochs):
        self._states.decreaseGamma(epochs)


    ## checking under percentile parameter if for this state it stays at low percentile, if it does - and he was winning so he was bluff his enemy
    def updateIfBluff(self,state):
        if self._states.getFinalState() == States.WIN and self.action == Poker.ALLIN:
            idx = self._states.getLinearIndex(state)
            qtable = self._states.getQTable()
            stateRank = self._states.getExpectedAction(state)
            stateRank = stateRank[1]
            if stateRank < States.POOR_PERCENTILE*self._states.getTotalPercentile():
                self.bluffs += 1


    ##get methods
    def getAvgWinsPerGame(self):
        if self.gameWins > 0:
            return sum(self.wins)/self.gameWins
        return 0

    def getAccumulatedSeq(self):
        return self.accumulatedSeq

    def getBluffs(self):
        return self.bluffs

    def getActionValue(self):
        qtable = self._states.getQTable()
        return qtable[self.action][self._states.getStateIndex()]

    def getAction(self):
        return self.action

    def action2String(self,action):
        if action == Poker.FOLD:
            return "fold"
        elif action == Poker.ALLIN:
            return "allin"

    def getStatesObj(self):
        return self._states

    def getStatus(self):
        return self._states.getFinalState()

    ##maintenance methods
    def getAgentClass(self):
        return __class__.__name__

    def getAgentId(self):
        return self._agentId

    def save(self):
        agentClass = self.getAgentClass()
        agentId = self.getAgentId()
        self._states.saveData(agentId)
        logger.log(logger.WARNING,agentClass + " (id:{}) saved all data.".format(str(agentId)))


def getInput(output,options = None,default = None):
    while (True):
        try:
            outMsg = output
            outOptions = ''
            if (options is not None):
                outOptions += ' ['
                comma = ""
                for i in range(len(options)):
                    outOptions += comma + str(i) +"-"+ options[i]
                    comma = ","
                    if (i == default):
                        outOptions += ' DEFAULT'
                outOptions += '] '
                
            response = input(outMsg  + outOptions + ": ")
            if (response == '' and default is not None):
                return default
            response = int(response)
            if (options is not None and (response < 0 or response > len(options))):
                logger.log(logger.WARNING,'Invalid input, Please choose: ', outOptions)
                continue
            return response
        except:
            logger.log(logger.WARNING,'Input must be an integer!')


def chooseAgent(game,epochs,enableLearning,player,default):
    chosenAgent = getInput("Choose Player " + str(player) ,['QAgent','RandomAgent','PlayerAgent','StrongValidatorAgent'],default)
    if (chosenAgent == 0):
        return QAgent(game.addPlayer(),epochs,enableLearning)
    if (chosenAgent == 1):
        return RandomAgent(game.addPlayer(),epochs)
    if (chosenAgent == 2):
        return PlayerAgent(game.addPlayer(),epochs)
    if (chosenAgent == 3):
        return StrongValidatorAgent(game.addPlayer(),epochs)


## class with apriori knowlenge to test the capabilities of our QAgent - doesn't learning the QAgent with that Validator agent
class StrongValidatorAgent(Agent):
    def __init__(self,id,epochs):
        super().__init__(id,epochs)
    def evalAct(self,state,enableLearning):
        if (state[3] < 5):
            return 1
        if (state[0] == state[1]):
           return 1
        if (state[0] > 8 and state[1] > 4):
           return 1
        if (state[0] > 4 and state[1] > 8):
           return 1
        if (state[0] > 11 or state[1] > 11):
           return 1
        return 0

    def getStatesObj(self):
        return None

    def getAgentClass(self):
        return __class__.__name__


def main(enableLearning):
    if enableLearning:
        logger.log(logger.WARNING,"starting learning process..")
    else:
        logger.log(logger.WARNING,"starting testing process..")

    win_count1 = 0
    win_count2 = 0
    epochs = getInput("Epochs ",None,200)
    Poker.TOTAL_CASH = getInput("Money ",None,10)
    if Poker.TOTAL_CASH < 0:
        logger.log(logger.WARNING,"Money cannot be negative.")
        sys.exit(1)
    totalRoundCount = 0

    game = Poker()
    agents = [chooseAgent(game,epochs,enableLearning,1,0),chooseAgent(game,epochs,enableLearning,2,1)]
    for i in range(epochs):
        logger.log(logger.WARNING,"^^^^^^^^^^^^^^^^^^^^^ game epoch {} ^^^^^^^^^^^^^^^^^^^^^".format(i+1))
        game.reset()
        round_count = 1
        print("----------- round {} -----------".format(round_count))
        game.deal()
        while not game.isGameover():
            if game.isRoundFinished():
                if enableLearning:
                    game.deal()
                else:
                    game.deal(agents[0].getStatesObj(),agents[1].getStatesObj())

            playerIdTurn = game.getplayerIDTurn()   
            state = game.getPlayerState(playerIdTurn)
            action = agents[playerIdTurn].evalAct(state,enableLearning)
            game.setAction(action)
            winners = game.getWinnerIDs()

            if enableLearning:
                agents[playerIdTurn].collect(state,action)

            if winners:
                if winners[0] == agents[0].getId():
                    agents[0].setStatus(States.WIN)
                    agents[1].setStatus(States.LOSE)
                elif winners[0] == agents[1].getId():
                    agents[1].setStatus(States.WIN)
                    agents[0].setStatus(States.LOSE)

                if enableLearning:
                    if agents[0].getStatus() == States.WIN:
                        actionValue = agents[0].getActionValue()
                        logger.log(logger.WARNING,agents[0].getAgentClass() + " (id:{}) wins: reward/penalty({}) = {}".format(1,agents[0].action2String(agents[0].getAction()),actionValue))
                    elif agents[1].getAgentClass() == QAgent and agents[1].getStatus() == States.WIN:
                        actionValue = agents[1].getActionValue()
                        logger.log(logger.WARNING,agents[0].getAgentClass() + " (id:{}) wins: reward/penalty({}) = {}".format(1,agents[1].action2String(agents[1].getAction()),actionValue))
                    else:
                        logger.log(logger.WARNING,agents[1].getAgentClass() + " (id:{}) wins".format(2))
                else:
                    logger.log(logger.WARNING,agents[playerIdTurn].getAgentClass() + " (id:{}) wins".format(playerIdTurn+1))

            if game.isRoundFinished() or game.isGameover():
                if enableLearning:
                    for idx,agent in enumerate(agents):
                        weight = -20
                        if game.isGameover():
                            if agent.getStatus() == States.WIN:
                                agent.updateGameWins()
                                agent.backtrackUpdateStates(True)
                                weight = 100
                            else:
                                agent.updateGameLose()
                                agent.backtrackUpdateStates(False)
                            logger.log(logger.WARNING,agent.getAgentClass() + "(id:{}) current round wins:".format(idx+1),(agent.getWins() / agent.getEpochs()) * 100.0,"%")
                            logger.log(logger.WARNING,agent.getAgentClass() + "(id:{}) current game wins:".format(idx+1),(agent.getGameWins() / agent.getEpochs()) * 100.0,"%")
                        else:
                            weight = -2
                            if agent.getStatus() == States.WIN:
                                weight = 10
                        agent.setReward(weight)
                        agent.discount(epochs)
                        agent.updateIfBluff(game.getPlayerState(idx))
                elif not enableLearning:
                    for idx,agent in enumerate(agents):
                        if game.isGameover():
                            if agent.getStatus() == States.WIN:
                                agent.updateGameWins()
                            else:
                                agent.updateGameLose()
                            logger.log(logger.WARNING,agent.getAgentClass() + "(id:{}) current round wins:".format(idx+1),(agent.getWins() / agent.getEpochs()) * 100.0,"%")
                            logger.log(logger.WARNING,agent.getAgentClass() + "(id:{}) current game wins:".format(idx+1),(agent.getGameWins() / agent.getEpochs()) * 100.0,"%")
                        agent.updateIfBluff(game.getPlayerState(idx))

            if game.isRoundFinished():
                round_count += 1
                totalRoundCount += 1
                print("----------- round {} -----------".format(round_count))

        
        game.printStatus()
    
    

    if enableLearning:
        logger.log(logger.WARNING,"model learning is finished.")
        logger.log(logger.WARNING,"")
        logger.log(logger.WARNING,"Summery Results:")
        logger.log(logger.WARNING,"----------------")
        for idx,agent in enumerate(agents):
            if type(agent) == QAgent:
                logger.log(logger.WARNING,agent.getAgentClass() + " (id:{}) average penalties:".format(idx+1),(agent.getStatesObj().getSumPenalties() / totalRoundCount))
            logger.log(logger.WARNING,agent.getAgentClass() + "(id:{}) total round wins:".format(idx+1),(agent.getWins() / agent.getEpochs()) * 100.0,"%")
            logger.log(logger.WARNING,agent.getAgentClass() + " (id:{}) total game wins:".format(idx+1),(agent.getGameWins() / agent.getEpochs()) * 100.0,"%")
            agent.save()
            agent.plotSeriesWinsGraph(agent.getAgentClass())
    else:
        logger.log(logger.WARNING,"mini AI poker game is finished.")
        logger.log(logger.WARNING,"")
        logger.log(logger.WARNING,"Summery Results:")
        logger.log(logger.WARNING,"----------------")
        for idx,agent in enumerate(agents):
            logger.log(logger.WARNING,agent.getAgentClass() + "(id:{}) total round wins:".format(idx+1),(agent.getWins() / agent.getEpochs()) * 100.0,"%")
            logger.log(logger.WARNING,agent.getAgentClass() + "(id:{}) total game wins:".format(idx+1),(agent.getGameWins() / agent.getEpochs()) * 100.0,"%")
            if type(agent) == QAgent:
                logger.log(logger.WARNING,agent.getAgentClass() + "(id:{}) average bluffs:".format(idx+1),(agent.getBluffs() / totalRoundCount))
                logger.log(logger.WARNING,agent.getAgentClass() + "(id:{}) average game wins:".format(idx+1),agent.getAvgWinsPerGame())


if __name__ == '__main__':
    enableLearning = getInput("Enable learning",['NO','YES'])
    main(enableLearning)