from __future__ import print_function
import sys
from collections import deque

from fow_chess.board import Board
from fow_chess.chesscolor import ChessColor
from fow_chess.piece import Piece, PieceType

sys.path.append("..")
from Game import Game
import numpy as np

"""
Game class implementation for the game of Fow of war chess.
Based on the OthelloGame then getGameEnded() was adapted to new rules.

Author: Hyeonseo Yang, github.com/KYHSGeekCode
Date: Sep 1, 2023.

Based on the TicTacToe by Evgeny Tyurin.
"""


class FowChessGame(Game):
    def __init__(self):
        super().__init__()
        self.board = Board("")
        self.previous_white_boards = deque(maxlen=8)
        self.previous_white_boards.append(self.board.to_fow_fen(ChessColor.WHITE))
        self.previous_black_boards = deque(maxlen=8)
        self.previous_black_boards.append(self.board.to_fow_fen(ChessColor.BLACK))

    def getInitBoard(self):
        # return initial board (numpy board)
        self.board = Board("")
        self.previous_white_boards = deque(maxlen=8)
        self.previous_white_boards.append(self.board.to_fow_fen(ChessColor.WHITE))
        self.previous_black_boards = deque(maxlen=8)
        self.previous_black_boards.append(self.board.to_fow_fen(ChessColor.BLACK))
        return self.board_to_npy(ChessColor.WHITE)

    def getBoardSize(self):
        # (a,b) tuple
        return (8, 8)

    def getActionSize(self):
        # return number of actions
        return 8 * 8 * 73

    def getNextState(self, board, player, action):
        # TODO action encoding
        # if player takes action on board, return next (board,player)
        # action must be a valid move
        if action == self.n * self.n:
            return (board, -player)
        b = Board(self.n)
        b.pieces = np.copy(board)
        move = (int(action / self.n), action % self.n)
        b.execute_move(move, player)
        return (b.pieces, -player)

    def getValidMoves(self, board, player):
        # TODO action encoding
        # return a fixed size binary vector
        valids = [0] * self.getActionSize()
        b = Board(self.n)
        b.pieces = np.copy(board)
        legalMoves = b.get_legal_moves(player)
        if len(legalMoves) == 0:
            valids[-1] = 1
            return np.array(valids)
        for x, y in legalMoves:
            valids[self.n * x + y] = 1
        return np.array(valids)

    def getGameEnded(self, board, player):
        # TODO save winner
        # return 0 if not ended, 1 if player 1 won, -1 if player 1 lost
        # player = 1
        b = Board(self.n)
        b.pieces = np.copy(board)

        if b.is_win(player):
            return 1
        if b.is_win(-player):
            return -1
        if b.has_legal_moves():
            return 0
        # draw has a very little value
        return 1e-4

    def getCanonicalForm(self, board, player):
        # return state if player==1, else return -state if player==-1
        # TODO return fogged board
        return player * board

    def getSymmetries(self, board, pi):
        # TODO: Maybe []?
        # mirror, rotational
        assert len(pi) == self.n ** 2 + 1  # 1 for pass
        pi_board = np.reshape(pi[:-1], (self.n, self.n))
        l = []

        for i in range(1, 5):
            for j in [True, False]:
                newB = np.rot90(board, i)
                newPi = np.rot90(pi_board, i)
                if j:
                    newB = np.fliplr(newB)
                    newPi = np.fliplr(newPi)
                l += [(newB, list(newPi.ravel()) + [pi[-1]])]
        return l

    def stringRepresentation(self, board):
        # 8x8 numpy array (canonical board)
        return board.tostring()

    @staticmethod
    def display(board):
        # TODO from fow main
        n = board.shape[0]

        print("   ", end="")
        for y in range(n):
            print(y, "", end="")
        print("")
        print("  ", end="")
        for _ in range(n):
            print("-", end="-")
        print("--")
        for y in range(n):
            print(y, "|", end="")  # print the row #
            for x in range(n):
                piece = board[y][x]  # get the piece to print
                if piece == -1:
                    print("X ", end="")
                elif piece == 1:
                    print("O ", end="")
                else:
                    if x == n:
                        print("-", end="")
                    else:
                        print("- ", end="")
            print("|")

        print("  ", end="")
        for _ in range(n):
            print("-", end="-")
        print("--")

    def board_to_npy(self, color: ChessColor):
        """
        https://pettingzoo.farama.org/environments/classic/chess/
        Like AlphaZero, the main observation space is an 8x8 image representing the board. It has 111 channels representing:

        Channels 0 - 3: Castling rights:
        Channel 0: All ones if white can castle queenside
        Channel 1: All ones if white can castle kingside
        Channel 2: All ones if black can castle queenside
        Channel 3: All ones if black can castle kingside
        Channel 4: Is black or white
        Channel 5: A move clock counting up to the 50 move rule. Represented by a single channel where the n th element in the flattened channel is set if there has been n moves
        Channel 6: All ones to help neural networks find board edges in padded convolutions
        Channel 7 - 18: One channel for each piece type and player color combination. For example, there is a specific channel that represents black knights. An index of this channel is set to 1 if a black knight is in the corresponding spot on the game board, otherwise, it is set to 0. Similar to LeelaChessZero, en passant possibilities are represented by displaying the vulnerable pawn on the 8th row instead of the 5th.
        Channel 19: represents whether a position has been seen before (whether a position is a 2-fold repetition)
        Channel 20 - 111 represents the previous 7 boards, with each board represented by 13 channels. The latest board occupies the first 13 channels, followed by the second latest board, and so on. These 13 channels correspond to channels 7 - 20.
        Similar to AlphaZero, our observation space follows a stacking approach, where it accumulates the previous 8 board observations.
        """
        number_of_channels = 112
        board_representation = np.zeros((8, 8, number_of_channels))
        board_deque = self.previous_white_boards if color == ChessColor.WHITE else self.previous_black_boards

        for idx, board in enumerate(board_deque):
            base_channel = idx * 14
            for position, piece in board.pieces.items():
                channel = get_channel_for_piece(piece)  # A function to get the correct channel based on the piece
                board_representation[position.file][position.rank][base_channel + 7 + channel] = 1
            board_representation[:, :, base_channel + 0] = board.castling[ChessColor.WHITE][0]  # White queenside
            board_representation[:, :, base_channel + 1] = board.castling[ChessColor.WHITE][1]  # White kingside
            board_representation[:, :, base_channel + 2] = board.castling[ChessColor.BLACK][0]  # Black queenside
            board_representation[:, :, base_channel + 3] = board.castling[ChessColor.BLACK][1]  # Black kingside
            board_representation[:, :, base_channel + 4] = 0 if board.side_to_move == ChessColor.WHITE else 1
            board_representation[:, :, base_channel + 5].flat[board.halfmove_clock] = 1
            board_representation[:, :, base_channel + 6] = 1
            board_representation[:, :, base_channel + 19] = 1  # TODO : Two fold repetition
        return board_representation


def get_channel_for_piece(piece: Piece) -> int:
    color = piece.color
    piece_type = piece.type
    ordinal = {
        PieceType.PAWN: 0,
        PieceType.KNIGHT: 1,
        PieceType.BISHOP: 2,
        PieceType.ROOK: 3,
        PieceType.QUEEN: 4,
        PieceType.KING: 5,
    }
    color_ordinal = {
        ChessColor.WHITE: 0,
        ChessColor.BLACK: 1,
    }
    return ordinal[piece_type] + color_ordinal[color] * len(ordinal)
