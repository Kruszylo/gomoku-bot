import numpy as np
import logging
EMPTY_FIELD = 0
WHITE_TOKEN = 1
BLACK_TONEK = -1
BOARD_SIZE = 15 # 15 x 15
class Game:

    def __init__(self, board_size = 15):   
        global BOARD_SIZE
        BOARD_SIZE = board_size 
        self.currentPlayer = 1
        self.gameState = GameState(np.array([0 for i in range(BOARD_SIZE*BOARD_SIZE)], dtype=np.int), 1)
        self.actionSpace = np.array([0 for i in range(BOARD_SIZE*BOARD_SIZE)], dtype=np.int)
        self.pieces = {'-1':'X', '0': '+', '1':'O'}
        self.grid_shape = (BOARD_SIZE,BOARD_SIZE)
        self.input_shape = (2,BOARD_SIZE,BOARD_SIZE)
        self.name = 'connect5'
        self.state_size = len(self.gameState.binary)
        self.action_size = len(self.actionSpace)

    def reset(self):
        self.gameState = GameState(np.array([0 for i in range(BOARD_SIZE*BOARD_SIZE)], dtype=np.int), 1)
        self.currentPlayer = 1
        return self.gameState

    def step(self, action):
        next_state, value, done = self.gameState.takeAction(action)
        self.gameState = next_state
        self.currentPlayer = -self.currentPlayer
        info = None
        return ((next_state, value, done, info))

    def identities(self, state, actionValues):
        identities = [(state,actionValues)]

        currentBoard = state.board
        currentAV = actionValues

        currentBoard = np.array([currentBoard[i] for i in range(BOARD_SIZE*BOARD_SIZE-1,0-1,-1)]
#      currentBoard[6], currentBoard[5],currentBoard[4], currentBoard[3], currentBoard[2], currentBoard[1], currentBoard[0]
#    , currentBoard[13], currentBoard[12],currentBoard[11], currentBoard[10], currentBoard[9], currentBoard[8], currentBoard[7]
#    , currentBoard[20], currentBoard[19],currentBoard[18], currentBoard[17], currentBoard[16], currentBoard[15], currentBoard[14]
#    , currentBoard[27], currentBoard[26],currentBoard[25], currentBoard[24], currentBoard[23], currentBoard[22], currentBoard[21]
#    , currentBoard[34], currentBoard[33],currentBoard[32], currentBoard[31], currentBoard[30], currentBoard[29], currentBoard[28]
#    , currentBoard[41], currentBoard[40],currentBoard[39], currentBoard[38], currentBoard[37], currentBoard[36], currentBoard[35]
#    ]
    )

        currentAV = np.array([currentBoard[i] for i in range(BOARD_SIZE*BOARD_SIZE-1,0-1,-1)]
#    currentAV[6], currentAV[5],currentAV[4], currentAV[3], currentAV[2], currentAV[1], currentAV[0]
#    , currentAV[13], currentAV[12],currentAV[11], currentAV[10], currentAV[9], currentAV[8], currentAV[7]
#    , currentAV[20], currentAV[19],currentAV[18], currentAV[17], currentAV[16], currentAV[15], currentAV[14]
#    , currentAV[27], currentAV[26],currentAV[25], currentAV[24], currentAV[23], currentAV[22], currentAV[21]
#    , currentAV[34], currentAV[33],currentAV[32], currentAV[31], currentAV[30], currentAV[29], currentAV[28]
#    , currentAV[41], currentAV[40],currentAV[39], currentAV[38], currentAV[37], currentAV[36], currentAV[35]
#    ]
    )

        identities.append((GameState(currentBoard, state.playerTurn), currentAV))

        return identities


class GameState():
    def __init__(self, board, playerTurn):
        self.board = board
        self.pieces = {'-1':'X', '0': '+', '1':'O'}
        board_2d_ind = np.array([list(range(i,i+BOARD_SIZE)) for i in range(0, len(board), BOARD_SIZE)])
        horizontal_winners = [board_2d_ind[row][i:i+5] for row in range(len(board_2d_ind)) for i in range(0,len(board_2d_ind[row])-5+1)]
        vertical_winners = [board_2d_ind[:,row][i:i+5] for row in range(len(board_2d_ind)) for i in range(0,len(board_2d_ind[row])-5+1)]
        diagonal_winners = [board_2d_ind.diagonal(diag)[i:i+5] for diag in range(-BOARD_SIZE+5,BOARD_SIZE-5+1) for i in range(0,len(board_2d_ind.diagonal(diag))-5+1)]
        reversed_diagonal_winners = [np.fliplr(board_2d_ind).diagonal(diag)[i:i+5] for diag in range(-BOARD_SIZE+5,BOARD_SIZE-5+1) for i in range(0,len(board_2d_ind.diagonal(diag))-5+1)]
        self.winners = horizontal_winners + vertical_winners + diagonal_winners + reversed_diagonal_winners
#        [
#            [0,1,2,3],
#            [1,2,3,4],
#            [2,3,4,5],
#            [3,4,5,6],
#            [7,8,9,10],
#            [8,9,10,11],
#            [9,10,11,12],
#            [10,11,12,13],
#            [14,15,16,17],
#            [15,16,17,18],
#            [16,17,18,19],
#            [17,18,19,20],
#            [21,22,23,24],
#            [22,23,24,25],
#            [23,24,25,26],
#            [24,25,26,27],
#            [28,29,30,31],
#            [29,30,31,32],
#            [30,31,32,33],
#            [31,32,33,34],
#            [35,36,37,38],
#            [36,37,38,39],
#            [37,38,39,40],
#            [38,39,40,41],
#
#            [0,7,14,21],
#            [7,14,21,28],
#            [14,21,28,35],
#            [1,8,15,22],
#            [8,15,22,29],
#            [15,22,29,36],
#            [2,9,16,23],
#            [9,16,23,30],
#            [16,23,30,37],
#            [3,10,17,24],
#            [10,17,24,31],
#            [17,24,31,38],
#            [4,11,18,25],
#            [11,18,25,32],
#            [18,25,32,39],
#            [5,12,19,26],
#            [12,19,26,33],
#            [19,26,33,40],
#            [6,13,20,27],
#            [13,20,27,34],
#            [20,27,34,41],
#
#            [3,9,15,21],
#            [4,10,16,22],
#            [10,16,22,28],
#            [5,11,17,23],
#            [11,17,23,29],
#            [17,23,29,35],
#            [6,12,18,24],
#            [12,18,24,30],
#            [18,24,30,36],
#            [13,19,25,31],
#            [19,25,31,37],
#            [20,26,32,38],
#
#            [3,11,19,27],
#            [2,10,18,26],
#            [10,18,26,34],
#            [1,9,17,25],
#            [9,17,25,33],
#            [17,25,33,41],
#            [0,8,16,24],
#            [8,16,24,32],
#            [16,24,32,40],
#            [7,15,23,31],
#            [15,23,31,39],
#            [14,22,30,38],
#            ]
        self.playerTurn = playerTurn
        self.binary = self._binary()
        self.id = self._convertStateToId()
        self.allowedActions = self._allowedActions()
        self.isEndGame = self._checkForEndGame()
        self.value = self._getValue()
        self.score = self._getScore()

    def _allowedActions(self):
        allowed = []
        for i in range(len(self.board)):
#            if i >= len(self.board) - 7:
            if self.board[i]==0:
                allowed.append(i)
#            else:
#                if self.board[i] == 0 and self.board[i+7] != 0:
#                    allowed.append(i)

        return allowed

    def _binary(self):

        currentplayer_position = np.zeros(len(self.board), dtype=np.int)
        currentplayer_position[self.board==self.playerTurn] = 1

        other_position = np.zeros(len(self.board), dtype=np.int)
        other_position[self.board==-self.playerTurn] = 1

        position = np.append(currentplayer_position,other_position)

        return (position)

    def _convertStateToId(self):
        player1_position = np.zeros(len(self.board), dtype=np.int)
        player1_position[self.board==1] = 1

        other_position = np.zeros(len(self.board), dtype=np.int)
        other_position[self.board==-1] = 1

        position = np.append(player1_position,other_position)

        id = ''.join(map(str,position))

        return id

    def _checkForEndGame(self):
        if np.count_nonzero(self.board) == BOARD_SIZE*BOARD_SIZE:
            return 1

        for x,y,z,a,b in self.winners:
            if (self.board[x] + self.board[y] + self.board[z] + self.board[a] + self.board[b] == 5 * -self.playerTurn):
                return 1
        return 0


    def _getValue(self):
        # This is the value of the state for the current player
        # i.e. if the previous player played a winning move, you lose
        for x,y,z,a,b in self.winners:
            if (self.board[x] + self.board[y] + self.board[z] + self.board[a] + self.board[b] == 5 * -self.playerTurn):
                return (-1, -1, 1)
        return (0, 0, 0)


    def _getScore(self):
        tmp = self.value
        return (tmp[1], tmp[2])




    def takeAction(self, action):
        newBoard = np.array(self.board)
        newBoard[action]=self.playerTurn
        
        newState = GameState(newBoard, -self.playerTurn)

        value = 0
        done = 0

        if newState.isEndGame:
            value = newState.value[0]
            done = 1

        return (newState, value, done) 




    def render(self, logger):
        for r in range(BOARD_SIZE):
            logger.info([self.pieces[str(x)] for x in self.board[BOARD_SIZE*r : (BOARD_SIZE*r + BOARD_SIZE)]])
        logger.info('--------------')