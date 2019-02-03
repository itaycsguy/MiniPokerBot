##version 0.7
##developers: Itay & Dean
import random,math,os,sys
import numpy as np
import itertools
import operator
import json
import copy
import time
import pickle
from collections import Counter


######## AUX START ########
class logger:
    DEBUG = 0
    INFO = 1
    WARNING = 2
    CRITICAL = 3
    USERREQUEST = 4

    threshold = 0

    @staticmethod
    def setPrintThreshold(level):
         logger.threshold = level

    @staticmethod
    def log(level,*args):
        if (level >= logger.threshold):
            print (*args)

def get_pretty_table(iterable, header):
    max_len = [len(x) for x in header]
    for row in iterable:
        row = [row] if type(row) not in (list, tuple) else row
        for index, col in enumerate(row):
            if max_len[index] < len(str(col)):
                max_len[index] = len(str(col))
    output = '-' * (sum(max_len) + 1) + '\n'
    output += '|' + ''.join([h + ' ' * (l - len(h)) + '|' for h, l in zip(header, max_len)]) + '\n'
    output += '-' * (sum(max_len) + 1) + '\n'
    for row in iterable:
        row = [row] if type(row) not in (list, tuple) else row
        output += '|' + ''.join([str(c) + ' ' * (l - len(str(c))) + '|' for c, l in zip(row, max_len)]) + '\n'
    output += '-' * (sum(max_len) + 1)
    return output

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total: 
        print()

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

def chooseAgent(epochs,enableLearning,loadtable,player,default):
    chosenAgent = getInput("Choose Player " + str(player) ,['QAgent','RandomAgent','PlayerAgent','StrongValidatorAgent'],default)
    if (chosenAgent == 0):
        return QAgent(Poker.INSTANCE.addPlayer(),epochs,enableLearning,loadtable)
    if (chosenAgent == 1):
        return RandomAgent(Poker.INSTANCE.addPlayer(),epochs,enableLearning,loadtable)
    if (chosenAgent == 2):
        return PlayerAgent(Poker.INSTANCE.addPlayer(),epochs,enableLearning,loadtable)
    if (chosenAgent == 3):
        return StrongValidatorAgent(Poker.INSTANCE.addPlayer(),epochs,enableLearning,loadtable)

def print_states(states):
    qtable = states.getQTable()
    for i in states.getStateSeeds().keys():
        arr = states.getMultiIndex(i)
        arr = list(arr)
        arr[0] += 2
        arr[1] += 2
        print(arr,'\t',qtable[Poker.ALLIN][i],'\t',qtable[Poker.FOLD][i])

######## AUX END ##########
##-------------------------------------------------------------
#### POKER GAME START ####

class Card:
  RANKS = (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14)
  SUITS = (u"\u2665", u"\u2666", u"\u2663", u"\u2660")
  #SUITS = (u"a", u"b", u"c", u"d")

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
	status = -1
	
	def getStatus(self):
		return self.status

	def setStatus(self,status):
		self.status = status

	def getStatusStr(self):
		if (self.status == Poker.ALLIN):
			return "All in"
		if (self.status == Poker.FOLD):
			return "Fold"
		if (self.status == Poker.PENDING):
			return "Pending"

	def __init__(self,id):
		self.id = id
		
	def withdrawAllMoneyByLimit(self,limit):
		m = min(self.money,limit)
		self.money -= m
		return m

	def deal(self,cards):
		self.setStatus(Poker.PENDING)
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
    INSTANCE = None
    HANDS = ('High Card','Pair','Two pair','Three of a kind','Straight','Flush','Full house','Four of a kind','Straight flush','Royal flush')
    PENDING = -1
    FOLD  = 0
    ALLIN = 1
    SBTURN = 0
    BBTURN = 1
    ENDROUND = 2
    GAMEOVER = 3
    TOTAL_SB_ENTRYBET = 0.5
    TOTAL_BB_ENTRYBET = 1.0

    def __init__(self,CashStartingAmount):
        if CashStartingAmount < 0:
            logger.log(logger.WARNING,"Money cannot be negative.")
            os.system("PAUSE")
            sys.exit(0)
        self.playerCashStartingAmount = CashStartingAmount
        self.players = []

    

	##control the game
    def addPlayer(self):
        id = len(self.players)
        self.players.append(Player(id))
        return id

    def reset(self):
        for player in self.players:
            player.money =  self.getPlayerStartingAmount()

        self.sb = random.randint(0,1)
        self.bb = self.sb ^ 1 ## (0 xor 1 = 1) (1 xor 1 = 0)
        self.roundWinners = []
        self._gameState = Poker.ENDROUND

    def sampleSubspace(self,state):
        state0 = state.getMultiIndex(random.choice(list(state.getStateSeeds().keys())))
        r1,r2 = random.sample(Card.SUITS,2)
        if state0[2] == 1:
            r2 = r1
        cards = [Card(state0[0]+2,r1),Card(state0[1]+2, r2)]
        return cards

    def getPlayerStatus(self,id):
        return (self.players[id].getStatus())

    def deal(self,states0=None,states1=None):
        self.dealer = Dealer()
        self.dealer.shuffle()
        self.sb = self.sb ^ 1 
        self.bb = self.bb ^ 1 
        self._gameState = 0 
        self.winnersIDs = []
        self.pot = self.players[self.sb].withdrawAllMoneyByLimit(0.5) + self.players[self.bb].withdrawAllMoneyByLimit(1.0)

        ## getting into this case while the QAgent playing at test game with sub-states to deal from
        if not states0 and not states1:
            self.players[0].deal(sorted(self.dealer.deal(2), key=operator.attrgetter('rank')))
            self.players[1].deal(sorted(self.dealer.deal(2), key=operator.attrgetter('rank')))
        else:                
            if states0:
                cards = self.sampleSubspace(states0)
                self.dealer.removeCards(cards)
                self.players[0].deal(sorted(cards, key=operator.attrgetter('rank')))
                self.players[1].deal(sorted(self.dealer.deal(2), key=operator.attrgetter('rank')))
            elif states1:
                cards = self.sampleSubspace(states1)
                self.dealer.removeCards(cards)
                self.players[1].deal(sorted(cards, key=operator.attrgetter('rank')))
                self.players[0].deal(sorted(self.dealer.deal(2), key=operator.attrgetter('rank')))
            else:
                logger.log(logger.CRITICAL,"2 Qagents is not suppoted in reduced states")
 

    def setAction(self,action):
        prev_pot = self.pot
        if (self._gameState == Poker.SBTURN):
            self.players[self.sb].setStatus(action)
            if (action == Poker.FOLD):
                self.winnersIDs = [self.bb]
                #logger.log(logger.DEBUG,"player",self.sb,"folds")
                self.renderGame()
                self.setEndStatus() 
            elif (action == Poker.ALLIN):
                self.pot += self.players[self.sb].withdrawAllMoneyByLimit(self.players[self.bb].getMoney())
                #logger.log(logger.DEBUG,"player",self.sb,"all-in")
                self._gameState = Poker.BBTURN
            else:
                logger.log(logger.CRITICAL,'Invalid poker action made by an agent!')
        elif (self._gameState == Poker.BBTURN):
            self.players[self.bb].setStatus(action)
            if (action == Poker.FOLD):
                self.winnersIDs = [self.sb]
                #logger.log(logger.DEBUG,"player",self.bb,"fold")
                self.renderGame()
                self.setEndStatus()
            elif (action == Poker.ALLIN):
                
                #logger.log(logger.DEBUG,"player",self.bb,"all-in\n")
                self.pot += self.players[self.bb].withdrawAllMoneyByLimit(self.pot - Poker.TOTAL_BB_ENTRYBET)
                self.renderGame()
                self.showDown()
                self.setEndStatus()
            else:
                logger.log(logger.CRITICAL,'Invalid poker action made by an agent!')
        else:
            logger.log(logger.CRITICAL,'Invalid poker action made by an agent!')
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
        logger.log(logger.DEBUG,"flop: {}".format(flop))
        sbhand,sbscore,sbrank = self.getBestHandByScore(flop + self.players[self.sb].cards)
        bbhand,bbscore,bbrank = self.getBestHandByScore(flop + self.players[self.bb].cards)

        if (sbscore > bbscore):
            self.winnersIDs =  [self.sb]
        elif (sbscore < bbscore):
            self.winnersIDs = [self.bb]
        else:
            self.winnersIDs = [self.sb,self.bb]

	##get game info
    def getPlayerStartingAmount(self):
        return self.playerCashStartingAmount

    def getplayerIDTurn(self):
        if (self._gameState == Poker.BBTURN):
            return self.bb
        else:
            return self.sb

    def getPlayersCount(self):
        return len(self.players)

    def getCurrentRoundNumber(self):
        return len(self.roundWinners)

    def printStatus(self):
        logger.log(logger.DEBUG,"Gameover:",self.roundWinners)

    def renderGame(self,id=None):
        table = []
        headers = ['id','type','money','cards','status']
        
        for player in self.players:
            row = []  
            row.append(player.id)
            if (self.bb == player.id):
                row.append('BB')  
            if (self.sb == player.id): 
                row.append('SB')  
            row.append(str(player.getMoney()) + '$')
            if id is None or id == player.id:
                row.append(str(player.cards))
            else:
                row.append(str('[HH,HH]'))
            row.append(" " + player.getStatusStr() + " ")
            table.append(row)

        logger.log(logger.DEBUG,get_pretty_table(table,headers))
        logger.log(logger.DEBUG,"Pot:",str(self.pot) + '%')

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
        return (self._gameState == Poker.ENDROUND) or self.isGameover()

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

#### POKER GAME END ####

##------------------------------------------------------------
#######  AGENTS  ############

class Agent:
    def __init__(self,id,epochs,enableLearning,loadtable):
        self.id = id
        self.roundWins = 0
        self.gameWins = 0
        self.seriesWinsGraph = list()

        self.enableLearning = enableLearning
        self.waitingForReward = False
        self.action = None
        self.bluffs = 0
        self._states = States(loadtable,id) ##loadtable = enableLearning
        self.status = None


    def getWinsFreqPerGame(self,totalRounds):
        return (0 if self.gameWins == 0 else (totalRounds / self.gameWins))

    def getBluffs(self):
        return self.bluffs

    def getRoundWins(self):
        return self.roundWins

    def backtrackUpdateStates(self):
        pass

    def trackAgentGames(self):
        self.gameWins += 1 if self.getStatesObj().getLastRoundResult() == States.WIN else 0
        self.seriesWinsGraph.append(self.gameWins)
        
    def plotSeriesWinsGraph(self):
        pass

    def getGameWins(self):
        return self.gameWins

    def evalAct(self,state):
        pass

    def setReward(self,epochs,R):
        pass

    def updateIfBluff(self):
        pass

    def getStatus(self):
        return self.status

    def getStatesObj(self):
        return self._states

    def getId(self):
        return self.id

    def save(self):
        pass
    
    def getTotalPenalties(self):
        return 0

    def setLastRoundResult(self,result):
        self.getStatesObj().setLastRoundResult(result)
        if result == States.WIN:
            self.roundWins += 1

    def getLastRoundResult(self):
        return self.getStatesObj().getLastRoundResult()

    def normalize(self):
        pass

    def getAgentClass(self):
        return __class__.__name__

class RandomAgent(Agent):

    def evalAct(self,state):
        return random.randint(0,1)

    def getAgentClass(self):
        return __class__.__name__

class PlayerAgent(Agent):
    def evalAct(self,state):
        Poker.INSTANCE.renderGame(self.id)
        return getInput("Please choose action",['Fold','All-in'])

    def getAgentClass(self):
        return __class__.__name__

class StrongValidatorAgent(Agent):
    ## class with apriori knowlenge to test the capabilities of our QAgent - doesn't learning the QAgent with that Validator agent
    def evalAct(self,state):
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

    def getAgentClass(self):
        return __class__.__name__

class QAgent(Agent):
    ##set methods
    ## evaluate the action should be taken
    def evalAct(self,state):
        self.getStatesObj().setState(state)
        stateIdx = self.getStatesObj().getcurrentStateIndex()
        qtable = self._states.getQTable()

        #self.action = self._states._getNextStateVotedValue(state)
        self.action = qtable[Poker.ALLIN][stateIdx] >= qtable[Poker.FOLD][stateIdx]
        logger.log(logger.DEBUG,"all in / fold",qtable[Poker.ALLIN][stateIdx],qtable[Poker.FOLD][stateIdx])    
        if self.enableLearning:      
            if (np.random.rand() < self._states.getEpsilon()):
                 self.action = not self.action
                 logger.log(logger.DEBUG,"Exploring triggered...")
        
        #if self._states.hasWaitingHistoryStatus(stateIdx):
        #    return Poker.FOLD
        self.action = int(self.action)
            
        
        self.waitingForReward = True
        return self.action

    ## update as backtracking all states who were piece of this game to winning
    def backtrackUpdateStates(self):
        if (self.enableLearning):
            self.getStatesObj().backtrackUpdateStates()

    def plotSeriesWinsGraph(self):
        import matplotlib.pyplot as plt
        if self.seriesWinsGraph:
            plt.plot(range(len(self.seriesWinsGraph)),self.seriesWinsGraph,linewidth=1,markersize=1)
            plt.xlabel('Epochs')
            plt.ylabel('Wins')
            plt.title('QAgent Improvement Curve')
            plt.grid(True)
            logger.log(logger.INFO,self.getAgentClass() + " plotting learning process graph..")
            plt.show()

    def setReward(self,epochs,R):
        if not self.waitingForReward or not self.enableLearning:
            return
        if self.enableLearning:    
            self.getStatesObj().saveStateHistoryPerRound(self.action,R)
        ## collect the state to use it in the game test mode
        if (self.getStatesObj().getcurrentState()[3] == Poker.INSTANCE.getPlayerStartingAmount()):
            self.getStatesObj().saveCurrentVisitedState()
        self.waitingForReward = False
        self.getStatesObj().setStateReward(self.action,R)

    ## checking under percentile parameter if for this state it stays at low percentile, if it does - and he was winning so he was bluff his enemy
    def updateIfBluff(self):
        state = self._states.getcurrentState()
        if self._states.getLastRoundResult() == States.WIN and self.action == Poker.ALLIN:
            idx = self._states.getLinearIndex(state)
            qtable = self._states.getQTable()
            se = (qtable[Poker.ALLIN][idx] - qtable[Poker.FOLD][idx])**2
            if se < States.BLUFF_UNDERBOUND:
                self.bluffs += 1

    def normalize(self):
        self._states.normalizeQTable()

    def getActionValue(self):
        qtable = self._states.getQTable()
        return qtable[self.action][self._states.getStateIndex()]

    def getTotalPenalties(self):
        return self.getStatesObj().getSumPenalties()

    def save(self):
        self._states.saveData(self.id)
        logger.log(logger.INFO,"Player {} ({}) saved Qtable".format(self.id,self.getAgentClass()))

    def getAgentClass(self):
        return __class__.__name__

class States():
    ## modeling the States of QAgent
    BACKTRACK_PENALTY = -0.01
    BACKTRACK_REWARD = 0.01

    COLOR_INTERVAL = 2
    BLIND_INTERVAL = 2
    CARDS_RANGE = 13

    BLUFF_UNDERBOUND = 0.001
    WIN = 1
    LOSE = 0

    def __init__(self,loadtable,agentId):
        self._qunatizationInterval = 5
        self._alpha = 0.2
        self._gamma = 0.2
        self._currentGamma = self._gamma
        self._epsilon = 0.1
        self._agentId = agentId

        self.POT_INTERVAL = self.getQuantizeAmount((2 * Poker.INSTANCE.getPlayerStartingAmount())) + 1
        self.STATE_STRUCT = (States.CARDS_RANGE,States.CARDS_RANGE,States.COLOR_INTERVAL,self.POT_INTERVAL,States.BLIND_INTERVAL)
        self._idxListSB = list()
        self._idxListBB = list()
        
        self._qtable,self._qtableVisitedStates,self.totalPercentile = self._init_qtable(loadtable,agentId)
        self.seedsOrder = 1
        self._state = -1
        self._stateIndex = -1

        self._lastRoundResult = None
        self._R_simple = {States.WIN : 0.1,States.LOSE : -0.1}
        self._penaltiesSum = 0
        self._linearStateHistoryPerRound = list()

    # V = (card1,card2,color/not color = {1,0},pot = [0-40],SB/BB = {1,0}) -> A = (allin = rewards,fold = rewards)
    def _init_qtable(self,loadtable,agentId):
        self._initNextStates()
        #start = time.time()
        #time.clock()
        table = list()
        seeds = {}
        totalPercentile = 0.0
        if not loadtable:
            ## creating a flat table to use it as linear and faster functionality
            table = [np.zeros(States.CARDS_RANGE * States.CARDS_RANGE * States.COLOR_INTERVAL * self.POT_INTERVAL * States.BLIND_INTERVAL),np.zeros(States.CARDS_RANGE * States.CARDS_RANGE * States.COLOR_INTERVAL * self.POT_INTERVAL * States.BLIND_INTERVAL)]
        else:
            logger.log(logger.INFO,"reading QTable..")
            path = str(os.path.dirname(str(os.path.abspath(__file__))))
            try:
                table = np.load(path + "\\Qtable.npy")
            except:
                logger.log(logger.CRITICAL,"file " + path + "\\Qtable.npy does not exist! you may want to enable learning first")
                os.system("PAUSE")
                sys.exit(0)
            with open(path + "\\QstateSeeds.npy","rb") as f:
                seeds = pickle.load(f)
            try:
                with open(path + "\\Qcash.npy","rb") as f:
                    cash = int(pickle.load(f))
                    if cash != Poker.INSTANCE.getPlayerStartingAmount():
                        logger.log(logger.CRITICAL,"The qtable does not correspoding to the entered Player Starting Amount amount [" + str(cash) + "]")
                        sys.exit(1)
            except:
                if not os.path.exists(path + "\\Qcash.npy"):
                    logger.log(logger.CRITICAL,"file " + path + "\\Qcash.npy does not exist! you may want to enable learning first")
                os.system("PAUSE")
                sys.exit(0)
            for key in seeds.keys():
                totalPercentile += (float(table[Poker.ALLIN][key]) + float(table[Poker.FOLD][key])) / 2.0
        #logger.log(logger.DEBUG,"loaded table in",time.time() - start,"seconds")
        logger.log(logger.DEBUG,"done")
        return (table,seeds,totalPercentile)

    ##get methods
    ## method to simulate next optional min cash loss against another random state
    ## make the loss of total reward be bigger or lower depend on its value as position correlation to the reward aim
    ## game success rate and the round success rate are closer and being almost the same rate
    ## cards reward more reasonable and no fold/allin cards changing
    def _getMaxCashLoss(self):
        simulationList = list()
        myState = self.getMultiIndex(self._stateIndex)
        for key in self._qtableVisitedStates.keys():
            alterState = self.getMultiIndex(key)
            simulationList.append((alterState[3] - myState[3])**2)
        if not simulationList:
            return 1.0
        ret = np.log2(max(simulationList) + 1)
        if abs(ret) == np.Inf:
            raise ""
        return ret


    def getcurrentState(self):
        return self._state

    def getcurrentStateIndex(self):
        return self._stateIndex

    def getTotalPercentile(self):
        return self.totalPercentile

    def getLastRoundResult(self):
        return self._lastRoundResult

    def getQTable(self):
        return self._qtable

    def getEpsilon(self):
        return self._epsilon

    def getStateSeeds(self):
        return self._qtableVisitedStates


    def saveStateHistoryPerRound(self,action,R):
        stateCurrGameStatus = 0
        if self._lastRoundResult == States.LOSE and action == Poker.ALLIN and Poker.INSTANCE.getPlayerStatus(int(not self._agentId)) == Poker.ALLIN:
            stateCurrGameStatus = 1
        self._linearStateHistoryPerRound.append((self._stateIndex,action,self._lastRoundResult,R,stateCurrGameStatus))

    def clearStateHistoryPerRound(self):
        self._linearStateHistoryPerRound = list()

    def getQuantizeAmount(self,amount):
        return int(amount / self._qunatizationInterval)

    def getSumPenalties(self):
        return self._penaltiesSum

    ##set methods
    def saveCurrentVisitedState(self):
        if (self._stateIndex not in self._qtableVisitedStates.keys()):
            self._qtableVisitedStates[self._stateIndex] = 0
        self._qtableVisitedStates[self._stateIndex] += 1

    def normalizeQTable(self):
        pass


    def backtrackUpdateStates(self):
        weight = States.BACKTRACK_REWARD if self._linearStateHistoryPerRound[-1][1] == States.WIN else States.BACKTRACK_PENALTY

        for stateOpCur in self._linearStateHistoryPerRound:
            state = stateOpCur[0]
            action = stateOpCur[1]
            # roundResult = stateOpCur[2]
            # R = stateOpCur[3]
            self._qtable[action][state] += weight
        self.clearStateHistoryPerRound()


    def setState(self,state):
        self._state = state
        self._stateIndex = self.getLinearIndex(state)

    def setLastRoundResult(self,result):
        self._lastRoundResult = result

    def setStateReward(self,action,R):

        if R < 0:
            self._penaltiesSum += R

        ## reward/penalty update function
        self._qtable[action][self._stateIndex] = \
                                 (1.0 - self._alpha) * self._qtable[action][self._stateIndex] + \
                                 self._alpha * (R + self._currentGamma * (self._getMaxCashLoss()))

        self._currentGamma -= (self._gamma / SETTINGS.epochs) # discount factor

    ##private methods
    def _initNextStates(self):
        for j0 in range(2,States.CARDS_RANGE + 2):
            for j1 in range(2,States.CARDS_RANGE + 2):
                for j2 in range(States.COLOR_INTERVAL):
                    for j3 in range(0,self._qunatizationInterval,self.POT_INTERVAL * self._qunatizationInterval):
                        self._idxListSB.append(self.getLinearIndex([j0,j1,j2,j3,0]))
                        self._idxListBB.append(self.getLinearIndex([j0,j1,j2,j3,1]))

   ## computing the expected value for the next state which is SB/BB
    def _getNextStateExpectedValue(self,state):
        idxList = self._idxListBB if state[4] == 0 else self._idxListSB
        return max(sum(self._qtable[Poker.ALLIN][idxList]) / len(idxList),sum(self._qtable[Poker.FOLD][idxList]) / len(idxList))

    ##
    def _getNextStateVotedValue(self,state):
        idxList = self._idxListBB if state[4] == Poker.SBTURN else self._idxListSB
        similarList = list()
        MAXNN = 3
        i = 0
        for idx in idxList:
            if i == MAXNN:
                break
            if self.isSimilar(state,self.getMultiIndex(idx)):
                similarList.append(self._qtable[Poker.ALLIN][idx] >= self._qtable[Poker.FOLD][idx])
                i = i + 1

        if not similarList:
            myIndex = self.getLinearIndex(state)
            return self._qtable[Poker.ALLIN][myIndex] >=  self._qtable[Poker.FOLD][myIndex]
  

    
        dic = Counter(similarList)
        m = sorted(dic.values())

        #x = {1: 2, 3: 4, 4: 3, 2: 1, 0: 0}
        m = sorted(dic.items(),reverse = True, key=operator.itemgetter(1))

        if dic[m[0][0]] >= int(MAXNN / 2):
            return random.randint(0,1)

        return m[0][1]

    def isSimilar(self,state1,state2):
        return state1[0] == state2[0] and state1[1] == state2[1] and state1[2] == state2[2]

    ##maintenance methods
    def saveData(self,agentId):
        path = str(os.path.dirname(str(os.path.abspath(__file__))))
        np.save(path + "\\Qtable.npy" ,self._qtable, allow_pickle=True, fix_imports=True)
        with open(path + "\\QstateSeeds.npy","wb") as f:
            pickle.dump(self._qtableVisitedStates,f,pickle.HIGHEST_PROTOCOL)
        with open(path + "\\Qcash.npy","wb") as f:
            pickle.dump(Poker.INSTANCE.getPlayerStartingAmount(),f,pickle.HIGHEST_PROTOCOL)

    def getMultiIndex(self,index):
        return np.unravel_index(index,self.STATE_STRUCT)

    def getLinearIndex(self,indices):
        return np.ravel_multi_index([indices[0]-2, indices[1]-2, indices[2],self.getQuantizeAmount(indices[3]),indices[4]],self.STATE_STRUCT)

#######  AGENTS ENDS  ##########
#-------------------------------------------------------------
############ MAIN ##############

def SimulateGames():
    enableLearning = SETTINGS.enableLearning
    totalRoundCount = 1
    Poker.INSTANCE = Poker(SETTINGS.chosenStartingAmount)
    agents = [chooseAgent(SETTINGS.epochs,enableLearning,SETTINGS.loadtable,1,0),chooseAgent(SETTINGS.epochs,enableLearning,SETTINGS.loadtable,2,1)]
    logger.log(logger.INFO,"Learning is in progess..." if enableLearning else "Testing is in progress...")

    for currentEpochCounter in range(1,SETTINGS.epochs+1):
        precnt = currentEpochCounter / SETTINGS.epochs * 100.0
        if (not SETTINGS.printgames and (currentEpochCounter == 1 or precnt.is_integer() and int(precnt) % 5 == 0) ):
                printProgressBar(currentEpochCounter, SETTINGS.epochs, prefix = 'Progress:', suffix = 'Complete', length = 50)

        logger.log(logger.DEBUG,"\n\n^^^^^^^^^^^^^^^^^^^^^ Epoch {} ^^^^^^^^^^^^^^^^^^^^^".format(currentEpochCounter))
        Poker.INSTANCE.reset()
        
        while not Poker.INSTANCE.isGameover():
            ####INIT THE ROUND ####
            if Poker.INSTANCE.isRoundFinished():
                logger.log(logger.DEBUG,"\n----------- round {} -----------".format(Poker.INSTANCE.getCurrentRoundNumber()))

                if SETTINGS.reducedStates:
                    Poker.INSTANCE.deal(agents[0].getStatesObj(),agents[1].getStatesObj())
                else:
                    Poker.INSTANCE.deal()
            
            
            ####PLAY THE GAME####
            playerIdTurn = Poker.INSTANCE.getplayerIDTurn()   
            state = Poker.INSTANCE.getPlayerState(playerIdTurn)
            action = agents[playerIdTurn].evalAct(state)
            Poker.INSTANCE.setAction(action)

            ####UPDATE WINNERS####
            if Poker.INSTANCE.isRoundFinished():
                winners = Poker.INSTANCE.getWinnerIDs()

                for idx,agent in enumerate(agents):
                    if (idx in winners):
                        agent.setLastRoundResult(States.WIN)
                        logger.log(logger.DEBUG,"Player " + str(idx) + "(" + agent.getAgentClass() + ") wins")
                    else:
                        agent.setLastRoundResult(States.LOSE)

                    agent.setReward(SETTINGS.epochs,Poker.INSTANCE.players[idx].getDividends())
                    #bluff has no bearing while playing against a random opponent
                    agent.updateIfBluff()
                totalRoundCount += 1
            
            if Poker.INSTANCE.isGameover():
                logger.log(logger.DEBUG,"\n<<<GAME OVER>>>>")
                Poker.INSTANCE.printStatus()
                for idx,agent in enumerate(agents):
                    # agent.normalize()
                    agent.trackAgentGames()
                    agent.backtrackUpdateStates()
                    logger.log(logger.DEBUG,"Player {} ({}) total games won: {}%".format(idx,agent.getAgentClass(),agent.getGameWins() / currentEpochCounter * 100.0 ))
                    logger.log(logger.DEBUG,"Player {} ({}) total rounds won: {}%".format(idx,agent.getAgentClass(),(agent.getRoundWins() / totalRoundCount) * 100.0))
                
    
    ##SUMMARY##  
    logger.log(logger.INFO,"\n\n\nAI has finished")
    logger.log(logger.INFO,"Summary results:")
    logger.log(logger.INFO,"-------------------------------") 
    
    for idx,agent in enumerate(agents):
        logger.log(logger.INFO,"Player {} ({}) games won: {}%".format(idx,agent.getAgentClass(),agent.getGameWins() / SETTINGS.epochs* 100.0))
        logger.log(logger.INFO,"Player {} ({}) rounds won: {}%".format(idx,agent.getAgentClass(),(agent.getRoundWins() / totalRoundCount) * 100.0))
        logger.log(logger.INFO,"Player {} ({}) frequency rounds per win: {}".format(idx,agent.getAgentClass(),agent.getWinsFreqPerGame(totalRoundCount)))
        if enableLearning:
            logger.log(logger.INFO,"Player {} ({}) average penalties: {}".format(idx,agent.getAgentClass(),agent.getTotalPenalties() / totalRoundCount))
            agent.save()
            agent.plotSeriesWinsGraph()
        else:
            logger.log(logger.INFO,"Player {} ({}) bluffs: {}%".format(idx,agent.getAgentClass(),((agent.getBluffs() / totalRoundCount) * 100.0)))

########## PRE SETUP ###########
class SETTINGS:
    enableLearning = True
    loadtable = False
    epochs = 1000
    chosenStartingAmount = 10
    printgames = False
    reducedStates = False
    chosenPreset = 0
logger.setPrintThreshold(logger.DEBUG if SETTINGS.printgames else logger.INFO)

def setPreset(presetID):
    SETTINGS.chosenPreset = presetID
    if (presetID == 0):
        SETTINGS.enableLearning = True
        SETTINGS.loadtable = False
        SETTINGS.epochs = 1000
        SETTINGS.chosenStartingAmount = 10
        SETTINGS.printgames = False
        SETTINGS.reducedStates = False
    elif (presetID == 1):
        SETTINGS.enableLearning = True
        SETTINGS.loadtable = True
        SETTINGS.epochs = 2500
        SETTINGS.chosenStartingAmount = 10
        SETTINGS.printgames = False
        SETTINGS.reducedStates = True
    elif (presetID == 2):
        SETTINGS.enableLearning = False
        SETTINGS.loadtable = True
        SETTINGS.epochs = 1000
        SETTINGS.chosenStartingAmount = 10
        SETTINGS.printgames = False
        SETTINGS.reducedStates = True
    elif (presetID == 3):
        SETTINGS.enableLearning = False
        SETTINGS.loadtable = True
        SETTINGS.epochs = 50
        SETTINGS.chosenStartingAmount = 10
        SETTINGS.printgames = True
        SETTINGS.reducedStates = True
    logger.setPrintThreshold(logger.DEBUG if SETTINGS.printgames else logger.INFO)

def main():
    while (True):
        logger.log(logger.INFO, \
    """
     __  __ _       _   _____      _                        _____ 
    |  \/  (_)     (_) |  __ \    | |                 /\   |_   _|
    | \  / |_ _ __  _  | |__) |__ | | _____ _ __     /  \    | |  
    | |\/| | | '_ \| | |  ___/ _ \| |/ / _ \ '__|   / /\ \   | |  
    | |  | | | | | | | | |  | (_) |   <  __/ |     / ____ \ _| |_ 
    |_|  |_|_|_| |_|_| |_|   \___/|_|\_\___|_|    /_/    \_\_____|
    """)

        logger.log(logger.INFO, \
"""
=========
Main Menu
=========
Settings:
0. Enable Learning: {}          [Create or updates the Qtable for the visited stated]
1. Load saved Table: {}         [Load the previous saved Qtable data]
2. Epochs: {}                   [Number of games to be simulated]
3. Starting Amount: {}          [The amount of chips the players will start with]
4. Print Games: {}              [Print the games results during execution or displays a progress bar]
5. Reduce States: {}            [Give the players cards that were played when Qtable was generated]

6. Choose Presets [{}]: (0-Initial Learning,1-Extended Learning,2-High Volume Testing,3-Displayed Testing)

Actions:
7. Start AI
8. Clear Screen
9. Exit
""".format(SETTINGS.enableLearning,SETTINGS.loadtable,SETTINGS.epochs ,SETTINGS.chosenStartingAmount,SETTINGS.printgames,SETTINGS.reducedStates,SETTINGS.chosenPreset))
        option = getInput("Command",None,None)
        if (option == 0):
            SETTINGS.enableLearning =  not SETTINGS.enableLearning
        elif (option == 1):
            SETTINGS.loadtable =  not SETTINGS.loadtable
        elif (option == 2):
            SETTINGS.epochs =   getInput("Epochs (default 200)",None,200)
        elif (option == 3):
            SETTINGS.chosenStartingAmount =  getInput("Money (default 10)",None,10)
        elif (option == 4):
            SETTINGS.printgames =  not SETTINGS.printgames
            logger.setPrintThreshold(logger.DEBUG if SETTINGS.printgames else logger.INFO)
        elif (option == 5):
            SETTINGS.reducedStates =  not SETTINGS.reducedStates
        elif (option == 6):
            setPreset(getInput("Choose preset",['Initial Learning','Extended Learning','High Volume Testing','Displayed Testing'],None))
        elif (option == 7):
            if (SETTINGS.enableLearning and not SETTINGS.loadtable):
                 answer =  getInput("WARNING Qtable will be overwritten. Continue?",['NO','YES'])
                 if (answer == 1):
                    SimulateGames()
                 else:
                    os.system("CLS")
            else:
                SimulateGames()
        elif (option == 8):
            os.system("CLS")
        elif (option == 9):
            sys.exit(0)
        else:
            logger.log(logger.WARNING, "Invalid option!")

        if (option != 7):
            os.system("CLS")

if __name__ == '__main__':
        main()