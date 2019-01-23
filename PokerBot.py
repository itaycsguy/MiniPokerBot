##version 0.5
import random,math,os
import numpy as np
import itertools
import operator
import json
import copy
import time
import pickle
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
    def __init__ (self):
        self.deck = []
        for suit in Card.SUITS:
            for rank in Card.RANKS:
                card = Card (rank, suit)
                self.deck.append(card)

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
        self.sb = 0
        self.bb = 1
        self.roundWinners = []

    def sampleSubspace(self,state,playerId):
        state0 = np.unravel_index(random.choice(list(state.getStateSeeds().keys())),
                                    (States.CARDS_RANGE,States.CARDS_RANGE,States.COLOR_INTERVAL,States.POT_INTERVAL,States.BLIND_INTERVAL))
        r1,r2 = random.sample(Card.SUITS,2)
        if state0[2] == 1:
            r2 = r1
        self.players[playerId].deal([Card(state0[0]+2,r1),Card(state0[1]+2, r2)])


    #TODO: need to implement deal to QAgent using it's QTable where is saved for this usage
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
        if not states0 and not states1:
            self.players[self.sb].deal(sorted(self.dealer.deal(2), key=operator.attrgetter('rank')))
            self.players[self.bb].deal(sorted(self.dealer.deal(2), key=operator.attrgetter('rank')))
        else:                
                if states0:
                    self.sampleSubspace(states0,0)
                if states1:
                    self.sampleSubspace(states1,1)
                else:
                    cards = self.dealer.deal(2)
                    while self.players[0].cards[0].isEqual(cards) or self.players[0].cards[1].isEqual(cards):
                        self.dealer.shuffle()
                        cards = self.dealer.deal(2)
                    self.players[1].deal(sorted(cards, key=operator.attrgetter('rank')))


    def setAction(self,action):
        prev_pot = self.pot
        if (self._gameState == Poker.SBTURN):
            if (action == Poker.FOLD):
                self.winnersIDs = [self.bb]
                logger.log(logger.INFO,"state 0: player",self.sb,"makes fold")
                self.setEndStatus() 
            elif (action == Poker.ALLIN):
                self.pot += self.players[self.sb].withdrawAllMoneyByLimit(self.players[self.bb].getMoney())
                logger.log(logger.INFO,"state 0:  player",self.sb,"makes allin")
                self._gameState = Poker.BBTURN
            else:
                logger.log(logger.INFO,'Unknown action... try again. pot: ' + str(self.pot))
        elif (self._gameState == Poker.BBTURN):
            if (action == Poker.FOLD):
                self.winnersIDs = [self.sb]
                logger.log(logger.INFO,"state 1:  player",self.bb,"makes fold")
                self.setEndStatus()
            elif (action == Poker.ALLIN):
                self.pot += self.players[self.bb].withdrawAllMoneyByLimit(self.pot - Poker.TOTAL_ENTRYBET)
                self.showDown()
                logger.log(logger.INFO,"state 1:  player",self.bb,"makes allin")
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

class States():
    WIN = "win"
    LOSE = "lose"
    POT_INTERVAL = 4 * Poker.TOTAL_CASH + 1
    COLOR_INTERVAL = 2
    BLIND_INTERVAL = 2
    CARDS_RANGE = 13
    POOR_PERCENTILE = 0.10


    def getLinearIndex(self,indices):
        return np.ravel_multi_index([indices[0]-2, indices[1]-2, indices[2],self.getQuantizeAmount(indices[3]),indices[4]], \
               [States.CARDS_RANGE,States.CARDS_RANGE,States.COLOR_INTERVAL,States.POT_INTERVAL,States.BLIND_INTERVAL])

    # V = (card1,card2,color/not color = {1,0},pot = [0-40],SB/BB = {1,0}) -> A = (allin = rewards,fold = rewards)
    def _init_qtable(self,enableLearning,agentId):
        start = time.time()
        time.clock()
        table = list()
        seeds = {}
        totalPercentile = 0.0
        if enableLearning:
            table = [np.zeros(States.CARDS_RANGE * States.CARDS_RANGE * States.COLOR_INTERVAL * States.POT_INTERVAL * States.BLIND_INTERVAL),np.zeros(States.CARDS_RANGE * States.CARDS_RANGE * States.COLOR_INTERVAL * States.POT_INTERVAL * States.BLIND_INTERVAL)]
            self._initNextStates()
        else:
            print("Reading Q-Table..")
            table = np.load('C:\\Users\\itgu1\\source\\repos\\PokerBot\\PokerBot\\Qtable_' + str(agentId) + '.npy')
            with open("C:\\Users\\itgu1\\source\\repos\\PokerBot\\PokerBot\\stateSeeds_" + str(agentId) + ".npy","rb") as f:
                seeds = pickle.load(f)
            for key in seeds.keys():
                totalPercentile += (table[0][key] + table[1][key]) / 2.0
        print("Completed within: ",time.time() - start,"seconds")
        return (table,seeds,totalPercentile)

    def __init__(self,enableLearning,agentId):
        self._idxListSB = list()
        self._idxListBB = list()
        self._qtable,self._stateSeeds,self.totalPercentile = self._init_qtable(enableLearning,agentId)
        self._state = list()
        self._stateIndex = -1
        self._alpha =    0.1
        self._gamma =    1.0
        self._epsilon =  0.01
        self._finalState = ""
        self._R = {States.WIN : 10.0,States.LOSE : -2.0}
        self._penalties = list()
        #currState = np.unravel_index(i,(States.CARDS_RANGE,States.CARDS_RANGE,States.COLOR_INTERVAL,States.POT_INTERVAL,States.BLIND_INTERVAL))

    ##get methods
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
    def appendOnce(self,state):
        stateInd = self.getLinearIndex(state)
        self._stateSeeds[stateInd] = 1

    def decreaseGamma(self,epochs):
        if self._gamma > 0 and epochs > 0:
            self._gamma -= 1/epochs

    def setCurrentState(self,state):
        self._state = state

    def setStateIndex(self,stateIdx):
        self._stateIndex = stateIdx

    def setFinalState(self,final):
        self._finalState = final

    def setStateReward(self,action):
        if self._finalState == States.LOSE:
            self._penalties.append(self._R[self._finalState])
        self._qtable[action][self._stateIndex] =    (1.0 - self._alpha) * self._qtable[action][self._stateIndex] + \
                                                    self._alpha * (self._R[self._finalState] + self._gamma * self._getNextStateExpectedValue(self._state) - \
                                                    self._qtable[action][self._stateIndex])


    ##private methods
    def _initNextStates(self):
        for j0 in range(2,States.CARDS_RANGE + 2):
            for j1 in range(2,States.CARDS_RANGE + 2):
                for j2 in range(States.COLOR_INTERVAL):
                    for j3 in range(States.POT_INTERVAL - 20):
                        self._idxListSB.append(self.getLinearIndex([j0,j1,j2,j3,0]))
                        self._idxListBB.append(self.getLinearIndex([j0,j1,j2,j3,1]))


    def _getNextStateExpectedValue(self,state):
        idxList = self._idxListBB if state[4] == 0 else self._idxListSB
        return (sum(self._qtable[0][idxList]) + sum(self._qtable[1][idxList])) / (2.0 * len(idxList))

    ##boolean methods
    def isEqual(self,state0,state1):
        return (self.getLinearIndex(state0) == self.getLinearIndex(state1))

    ##maintenance methods
    def saveData(self,agentId):
        path = str(os.path.dirname(str(os.path.abspath(__file__))))
        np.save(path + "\\Qtable_" + str(agentId) + ".npy" ,self._qtable, allow_pickle=True, fix_imports=True)
        with open(path + "\\stateSeeds_" + str(agentId) + ".npy","wb") as f:
            pickle.dump(self._stateSeeds,f,pickle.HIGHEST_PROTOCOL)

class Agent:
    def __init__(self,id,epochs):
        self.loc = 0
        self.wins = np.zeros(epochs)
        self.id = id

    def setWin(self,num):
        if self.loc < len(self.wins):
            self.wins[self.loc] = num
            self.loc += 1

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

    def evalAct(self):
        pass

    def setReward(self):
        pass

    def accumulateGraph(self):
        pass

    def setStatus(self,statusString):
        pass
    
    def collect(self,state):
        pass

    def discount(self,epochs):
        pass

    def updateIfBluff(self,state):
        pass

    def getAccumulatedSeq(self):
        return None

    def getBluffs(self):
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
    def evalAct(self,state):
        return random.randint(0,1)

    def getStatesObj(self):
        return None

    def getBluffs(self):
        return 0

    def getAgentClass(self):
        return __class__.__name__

class PlayerAgent(Agent):
    def evalAct(self,state):
        return getInput("choose action",['Fold','All-in'])

    def get_pretty_table(iterable, header):
        max_len = [len(x) for x in header]
        for row in iterable:
            row = [row] if type(row) not in (list, tuple) else row
            for index, col in enumerate(row):
                if max_len[index] < len(str(col)):
                    max_len[index] = len(str(col))
        output = ''
        #output = '-' * (sum(max_len) + 1) + '\n'
        output += '\t' + ''.join([h + ' ' * (l - len(h)) + '\t' for h, l in zip(header, max_len)]) + '\n'
        #output += '-' * (sum(max_len) + 1) + '\n'
        for row in iterable:
            row = [row] if type(row) not in (list, tuple) else row
            output += '\t' + ''.join([str(c) + ' ' * (l - len(str(c))) + '\t' for c, l in zip(row, max_len)]) + '\n'
        #output += '-' * (sum(max_len) + 1) + '\n'
        return output
       
class QAgent(Agent):
    def __init__(self,id,epochs,enableLearning):
        super().__init__(id,epochs)
        self._states = States(enableLearning,id)
        self.waitingForReward = False
        self.action = 0
        self._agentId = id
        self.bluffs = 0
        self.currWins = 0
        self.currentRounds = 0
        self.currAvgWinsPerRound = 1.0
        self.accumulatedSeq = list()
        self.rc = None

    ##set methods
    ## the curve should analyze hands [success - train & test (TP) <-> success - train but failed at the test (FP)]
    def buildRocCurve(self,y,scores,pos_labels=2):
        self.rc = metrics.roc_curve(y, scores, pos_label)

    def evalAct(self,state):
        self._states.setCurrentState(state)
        stateIdx = self._states.getLinearIndex(state)
        self._states.setStateIndex(stateIdx)
        qtable = self._states.getQTable()
        allinVal = qtable[0][stateIdx]
        foldVal = qtable[1][stateIdx]
        self.action = (allinVal >= foldVal)
        if (np.random.rand() < self._states.getEpsilon()):
            self.action = not self.action
        self.action = int(self.action)
        self.waitingForReward = True
        return self.action

    def setStatus(self,statusString):
        if statusString == States.WIN:
            self.currWins += 1
        self._states.setFinalState(statusString)

    def accumulateGraph(self):
        self.accumulatedSeq.append(self.currWins)


    def collect(self,state):
        self._states.appendOnce(state)

    def setReward(self):
        if not self.waitingForReward:
            return
        self.waitingForReward = False
        self._states.setStateReward(self.action)

    def discount(self,epochs):
        self._states.decreaseGamma(epochs)


    #TODO: need to approximate it state0 is less than state1 and who winner is 
    def updateIfBluff(self,state):
        if self.action == 0:
            idx = self._states.getLinearIndex(state)
            qtable = self._states.getQTable()
            stateRank = (qtable[0][idx] + qtable[1][idx]) / 2.0
            if stateRank < States.POOR_PERCENTILE*self._states.getTotalPercentile():
                self.bluffs += 1


    ##get methods
    def getAvgWinsRate(self):
        freq = list()
        prevWin = 0
        counter = 0
        for win in self.accumulatedSeq:
          if prevWin != win:
              prevWin = win
              freq.append(counter)
          else:
              counter += 1
        return int(sum(freq)/len(self.accumulatedSeq))




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
        logger.log(logger.WARNING,agentClass + " #{} saved all data.".format(str(agentId)))

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
    chosenAgent = getInput("Choose Player " + str(player) ,['QAgent','RandomAgent','PlayerAgent'],default)
    if (chosenAgent == 0):
        return QAgent(game.addPlayer(),epochs,enableLearning)
    if (chosenAgent == 1):
        return RandomAgent(game.addPlayer(),epochs)
    if (chosenAgent == 2):
        return PlayerAgent(game.addPlayer(),epochs)


def main(enableLearning):
    if enableLearning:
        logger.log(logger.WARNING,"Starting learning process..")
    else:
        logger.log(logger.WARNING,"Starting testing process..")

    win_count1 = 0
    win_count2 = 0
    epochs = 1000

    game = Poker()
    agents = [chooseAgent(game,epochs,enableLearning,1,0),chooseAgent(game,epochs,enableLearning,2,1)]
    for i in range(epochs):
        logger.log(logger.WARNING,"^^^^^^^^^^^^^^^^^^^^^ game epoch #{} ^^^^^^^^^^^^^^^^^^^^^".format(i))
        game.reset()
        round_count = 1
        print("----------- round #{} -----------".format(round_count))
        game.deal()
        while not game.isGameover():
            if game.isRoundFinished():
                if enableLearning:
                    game.deal()
                else:
                    game.deal(agents[0].getStatesObj(),agents[1].getStatesObj())

            playerIdTurn = game.getplayerIDTurn()   
            state = game.getPlayerState(playerIdTurn)
            action = agents[playerIdTurn].evalAct(state)
            game.setAction(action)
            winners = game.getWinnerIDs()

            if enableLearning:
                agents[playerIdTurn].collect(state)

            if winners:
                if winners[0] == agents[0].getId():
                    agents[0].setStatus(States.WIN)
                    agents[1].setStatus(States.LOSE)
                elif winners[0] == agents[1].getId():
                    agents[1].setStatus(States.WIN)
                    agents[0].setStatus(States.LOSE)

                if agents[0].getStatus() == States.WIN:
                    actionValue = agents[0].getActionValue()
                    logger.log(logger.WARNING,agents[0].getAgentClass() + "#{} wins: v({}) = {}".format(0,agents[0].action2String(agents[0].getAction()),actionValue))
                elif agents[1].getAgentClass() == QAgent and agents[1].getStatus() == "win":
                    actionValue = agents[1].getActionValue()
                    logger.log(logger.WARNING,agents[0].getAgentClass() + "#{} wins: v({}) = {}".format(1,agents[1].action2String(agents[1].getAction()),actionValue))
                else:
                    logger.log(logger.WARNING,agents[playerIdTurn].getAgentClass() + "#{} wins".format(playerIdTurn))

            if enableLearning and (game.isRoundFinished() or game.isGameover()):
                for idx,agent in enumerate(agents):
                    agent.updateIfBluff(game.getPlayerState(idx))
                    agent.setReward()
                    agent.discount(epochs)
                    agent.accumulateGraph()
            elif not enableLearning and (game.isRoundFinished() or game.isGameover()):
                for idx,agent in enumerate(agents):
                    agent.updateIfBluff(game.getPlayerState(idx))
                    agent.accumulateGraph()


            if game.isRoundFinished():
                round_count += 1
                print("----------- round #{} -----------".format(round_count))


        game.printStatus()
        winners = game.getWinnerIDs()
        if winners[0] == agents[0].getId():
            agents[0].setWin(1)
        elif winners[0] == agents[1].getId():
            agents[1].setWin(1)
        if i < epochs-1:
            logger.log(logger.WARNING,agents[0].getAgentClass() + "(id:0) current wins:",(agents[0].getWins() / agents[0].getEpochs()) * 100.0,"%")
            logger.log(logger.WARNING,agents[1].getAgentClass() + "(id:1) current wins:",(agents[1].getWins() / agents[1].getEpochs()) * 100.0,"%")
    

    if enableLearning:
        logger.log(logger.WARNING,"model learning Is finished.")
        for idx,agent in enumerate(agents):
            if type(agent) == QAgent:
                logger.log(logger.WARNING,agent.getAgentClass() + "(id:{}) average penalties:".format(idx),agent.getStatesObj().getSumPenalties() / agent.getEpochs() * 100.0,"%")
            logger.log(logger.WARNING,agent.getAgentClass() + "(id:0) total wins:",(agent.getWins() / agent.getEpochs()) * 100.0,"%")
            agent.save()
            accumulatedSeq = agent.getAccumulatedSeq()
            if accumulatedSeq:
                ##wins normalization
                for idx,_ in enumerate(accumulatedSeq):
                    accumulatedSeq[idx] = accumulatedSeq[idx]/len(accumulatedSeq)
                plt.plot(range(len(accumulatedSeq)),accumulatedSeq,'go',linewidth=2,markersize=5)
                plt.xlim((1,len(accumulatedSeq)))
                plt.xlabel('Trails')
                plt.ylabel('Wins')
                plt.title('QAgent Improvement')
                plt.grid(True)
                logger.log(logger.WARNING,agent.getAgentClass() + " plotting learning improvement graph..")
                plt.show()
    else:
        for idx,agent in enumerate(agents):
            logger.log(logger.WARNING,agent.getAgentClass() + "(id:{}) average bluffs:".format(idx),agent.getBluffs() / agent.getEpochs() * 100.0,"%")
            logger.log(logger.WARNING,agent.getAgentClass() + "(id:{}) total wins:".format(idx),(agent.getWins() / agent.getEpochs()) * 100.0,"%")
            if type(agent) == QAgent:
                logger.log(logger.WARNING,agent.getAgentClass() + "(id:{}) wins rate:".format(idx),agent.getAvgWinsRate())        
        logger.log(logger.WARNING,"mini poker game is over.")

if __name__ == '__main__':
    enableLearning = getInput("Enable learning",['NO','YES'])
    main(enableLearning)