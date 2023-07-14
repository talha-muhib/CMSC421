# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood().asList()
        newGhostStates = successorGameState.getGhostStates()
        newGhostPositions = [ghostState.getPosition() for ghostState in newGhostStates]
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        foodDistances = [util.manhattanDistance(newPos, i) for i in newFood]
        ghostDistances = [util.manhattanDistance(newPos, i) for i in newGhostPositions]

        minFoodDistance = 1/(min(foodDistances, default = 0) + 0.01)
        minGhostDistance = 1/(min(ghostDistances, default = 0) + 0.001)
        minIndex = ghostDistances.index(min(ghostDistances))
        minScaredTime = newScaredTimes[minIndex]

        return successorGameState.getScore() + minFoodDistance - minGhostDistance + minScaredTime

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions(self.index)
        successors = [gameState.generateSuccessor(self.index, action) for action in legalMoves]
        scores = [self.minimax(nextState, self.depth, (self.index + 1) % gameState.getNumAgents()) for nextState in successors]

        # Choose one of the best actions
        #scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]
    
    def minimax(self, gameState, depth, agentIndex):
        if gameState.isWin() or gameState.isLose() or (depth == 0):
            return self.evaluationFunction(gameState)

        nextIndex = (agentIndex + 1) % gameState.getNumAgents()
        depth = (depth - 1) if nextIndex == 0 else depth
        legalMoves = gameState.getLegalActions(agentIndex)
        successors = [gameState.generateSuccessor(agentIndex, action) for action in legalMoves]
        scores = [self.minimax(nextState, depth, nextIndex) for nextState in successors]

        # Choose one of the best actions
        #scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        if agentIndex == 0:
            return max(scores)
        return min(scores)

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        # Collect legal moves and successor states
        score = self.alphaBeta(gameState, self.depth, self.index, float('-inf'), float('inf'))

        return score

    def alphaBeta(self, gameState, depth, agentIndex, alpha, beta):
        if gameState.isWin() or gameState.isLose() or (depth == 0):
            return self.evaluationFunction(gameState)

        maxScore = float('-inf')
        minScore = float('inf')
        nextIndex = (agentIndex + 1) % gameState.getNumAgents()
        d2 = (depth - 1) if nextIndex == 0 else depth
        legalMoves = gameState.getLegalActions(agentIndex)

        if agentIndex == 0:
            scores = []

            for action in legalMoves:
                state = gameState.generateSuccessor(agentIndex, action)
                maxScore = max(maxScore, self.alphaBeta(state, d2, nextIndex, alpha, beta))
                if maxScore > beta:
                    return maxScore
                alpha = max(alpha, maxScore)
                if (depth == self.depth) and (agentIndex == 0):
                    scores.append(maxScore)
            
            if (depth == self.depth) and (agentIndex == 0):
                return legalMoves[scores.index(max(scores))]
            else:
                return maxScore
        else:
            for action in legalMoves:
                state = gameState.generateSuccessor(agentIndex, action)
                minScore = min(minScore, self.alphaBeta(state, d2, nextIndex, alpha, beta))
                if minScore < alpha:
                    return minScore
                beta = min(beta, minScore)

            return minScore
            

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions(self.index)
        successors = [gameState.generateSuccessor(self.index, action) for action in legalMoves]
        scores = [self.expectimax(nextState, self.depth, (self.index + 1) % gameState.getNumAgents()) for nextState in successors]

        # Choose one of the best actions
        #scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        return legalMoves[chosenIndex]

    def expectimax(self, gameState, depth, agentIndex):
        if gameState.isWin() or gameState.isLose() or (depth == 0):
            return self.evaluationFunction(gameState)

        nextIndex = (agentIndex + 1) % gameState.getNumAgents()
        depth = (depth - 1) if nextIndex == 0 else depth
        legalMoves = gameState.getLegalActions(agentIndex)
        successors = [gameState.generateSuccessor(agentIndex, action) for action in legalMoves]
        scores = [self.expectimax(nextState, depth, nextIndex) for nextState in successors]

        # Choose one of the best actions
        #scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        if agentIndex == 0:
            return max(scores)
        return sum(scores)/len(scores)

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: I copied my evaluation function from task 1. Somehow it worked here too
    """
    "*** YOUR CODE HERE ***"
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood().asList()
    newGhostStates = currentGameState.getGhostStates()
    newGhostPositions = [ghostState.getPosition() for ghostState in newGhostStates]
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    foodDistances = [util.manhattanDistance(newPos, i) for i in newFood]
    ghostDistances = [util.manhattanDistance(newPos, i) for i in newGhostPositions]

    minFoodDistance = 1/(min(foodDistances, default = 0) + 0.01)
    minGhostDistance = 1/(min(ghostDistances, default = 0) + 0.001)
    minIndex = ghostDistances.index(min(ghostDistances))
    minScaredTime = newScaredTimes[minIndex]

    return currentGameState.getScore() + minFoodDistance - minGhostDistance + minScaredTime

# Abbreviation
better = betterEvaluationFunction