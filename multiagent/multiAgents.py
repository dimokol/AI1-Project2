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
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
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

    def evaluationFunction(self, currentGameState: GameState, action):
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
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        foodList = currentGameState.getFood().asList()

        foodDist = float('inf')
        for food in foodList:
            foodDist = min(util.manhattanDistance(newPos, food), foodDist)

        ghostDist = float('inf')
        for ghost in newGhostStates:
            if ghost.scaredTimer is 0:
                ghostDist = min(util.manhattanDistance(newPos, ghost.getPosition()), ghostDist)

        actionScore = 0
        if ghostDist <= 2:
            actionScore -= 100
        elif newPos in foodList:
            actionScore += 100
        else:
            actionScore += 100 - foodDist

        return actionScore

def scoreEvaluationFunction(currentGameState: GameState):
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

    def getAction(self, gameState: GameState):
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
        def minimax(gameState, depth, agentIndex, resaultAction):
            if depth == self.depth or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState), resaultAction

            if agentIndex == gameState.getNumAgents() - 1:
                nextAgent = self.index
                depth += 1
            else:
                nextAgent = agentIndex + 1

            if not agentIndex:
                maxScore = float('-inf')
                bestAction = None
                for action in gameState.getLegalActions(agentIndex):
                    if action == "Stop":
                        continue

                    score = minimax(gameState.generateSuccessor(agentIndex, action), depth, nextAgent, action)[0]
                    maxScore = max(score, maxScore)
                    if score == maxScore:
                        bestAction = action

                return maxScore, bestAction
            else:
                minScore = float('inf')
                bestAction = None
                for action in gameState.getLegalActions(agentIndex):
                    score = minimax(gameState.generateSuccessor(agentIndex, action), depth, nextAgent, action)[0]
                    minScore = min(score, minScore)
                    if score == minScore:
                        bestAction = action

                return minScore, bestAction
        
        return minimax(gameState, 0, 0, None)[1]

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def alphabeta(gameState, depth, agentIndex, resaultAction, a, b):
            if depth == self.depth or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState), resaultAction

            if agentIndex == gameState.getNumAgents() - 1:
                nextAgent = self.index
                depth += 1
            else:
                nextAgent = agentIndex + 1

            if not agentIndex:
                maxScore = float('-inf')
                bestAction = None
                for action in gameState.getLegalActions(agentIndex):
                    if action == "Stop":
                        continue

                    score = alphabeta(gameState.generateSuccessor(agentIndex, action), depth, nextAgent, action, a, b)[0]
                    maxScore = max(score, maxScore)
                    if score == maxScore:
                        bestAction = action

                    if maxScore > b:
                        return maxScore, bestAction

                    a = max(maxScore, a)

                return maxScore, bestAction
            else:
                minScore = float('inf')
                bestAction = None
                for action in gameState.getLegalActions(agentIndex):
                    score = alphabeta(gameState.generateSuccessor(agentIndex, action), depth, nextAgent, action, a, b)[0]
                    minScore = min(score, minScore)
                    if score == minScore:
                        bestAction = action

                    if minScore < a:
                        return minScore, bestAction

                    b = min(minScore, b)

                return minScore, bestAction
        
        return alphabeta(gameState, 0, 0, None, float('-inf'), float('inf'))[1]

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        def expectimax(gameState, depth, agentIndex, resaultAction):
            if depth == self.depth or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState), resaultAction

            if agentIndex == gameState.getNumAgents() - 1:
                nextAgent = self.index
                depth += 1
            else:
                nextAgent = agentIndex + 1

            if not agentIndex:
                maxScore = float('-inf')
                bestAction = None
                for action in gameState.getLegalActions(agentIndex):
                    if action == "Stop":
                        continue

                    score = expectimax(gameState.generateSuccessor(agentIndex, action), depth, nextAgent, action)[0]
                    maxScore = max(score, maxScore)
                    if score == maxScore:
                        bestAction = action

                return maxScore, bestAction
            else:
                averageScore = 0
                actions = gameState.getLegalActions(agentIndex)
                propability = 1.0 / len(actions)
                for action in actions:
                    score = expectimax(gameState.generateSuccessor(agentIndex, action), depth, nextAgent, action)[0]
                    averageScore += score * propability

                return averageScore, None
        
        return expectimax(gameState, 0, 0, None)[1]

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: I scored each gameState by: -food distance -food left - ghost distance - scared ghost distance - scared ghost left - capsyles left
    """
    "*** YOUR CODE HERE ***"
    if currentGameState.isLose():
        return float('-inf')
    elif currentGameState.isWin():
        return float('inf')

    pos = currentGameState.getPacmanPosition()
    foodList = currentGameState.getFood().asList()
    ghostStates = currentGameState.getGhostStates()

    foodLeft = currentGameState.getNumFood()
    capsulesLeft = len(currentGameState.getCapsules())
    notScaredGhostsLeft = 0
    scaredGhostsLeft = 0

    foodDist = float('inf')
    for food in foodList:
        foodDist = min(util.manhattanDistance(pos, food), foodDist)

    ghostDist = float('inf')
    scaredGhostDist = float('inf')
    for ghost in ghostStates:
        if ghost.scaredTimer is 0:
            ghostDist = min(util.manhattanDistance(pos, ghost.getPosition()), ghostDist)
            notScaredGhostsLeft += 1
            if ghostDist <= 2:
                return float('-inf')
        elif ghost.scaredTimer > 2:
            scaredGhostDist = min(util.manhattanDistance(pos, ghost.getPosition()), scaredGhostDist)
            scaredGhostsLeft += 1
    
    
    score = 1.0/(foodDist + 1) * 1500
    if notScaredGhostsLeft:
        if ghostDist < 4:
            score += ghostDist * (-1.0/(ghostDist + 1) * 100)
    if scaredGhostsLeft:
        score = 1.0/(scaredGhostDist + 1) * 1000
    score += 1.0/(foodLeft + 1) * 190000
    score += 1.0/(capsulesLeft + 1) * 10000
    score += 1.0/(scaredGhostsLeft + 1) * 1000
    return score

# Abbreviation
better = betterEvaluationFunction
