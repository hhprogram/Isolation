"""This file contains all the classes you must complete for this project.

You can use the test cases in agent_test.py to help during development, and
augment the test suite with your own test cases to further test your code.

You must test your agent's strength against a set of agents with known
relative strength using tournament.py and include the results in your report.
"""
import random


class Timeout(Exception):
    """Subclass base exception for code clarity."""
    pass


def custom_heuristics(game, player, choice):
    """
    wrapper method that calls the particular evaluation function based on variable CHOICE. All 
    functions returns -infitinity if legal moves for PLAYER == 0
    """
    '''HLI CODE'''
    def max_moves():
        """
        evaluation function that just returns number of legal moves
        """
        return len(game.get_legal_moves(player))
    def move_difference():
        """
        evaluation function that returns the difference between my legal moves and the opponents
        legal moves
        """
        return (len(game.get_legal_moves(player)) - len(game.get_opponent(player)))
    def mix_strategy():
        """
        most complex evaluation function. evaluation function Calculates:
            (1) number of blank spaces on PLAYER's side of the board (ie the part of the board that 
                PLAYER is 'closer' to than his opponent - cells equadistant considered part of 
                'wall' and not incl. in sum) plus
            (2) the difference between his legal moves and opponent's legal moves
            (3) The max number of legal moves after 1 forecasted move which cannot be blocked by 
                opponent in his ply right after the forecasted move
        """
        return
    if choice == 1:
        max_moves()
    elif choice == 2:
        move_difference()
    elif choice == 3:
        mix_strategy()
    else:
        raise ValueError(str(choice) + " not a valid option")
    '''HLI CODE'''


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the cu rrent game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    '''HLI CODE'''
    return custom_heuristics(game, player, 2)
    '''HLI CODE'''

class CustomPlayer:
    """Game-playing agent that chooses a move using your evaluation function
    and a depth-limited minimax algorithm with alpha-beta pruning. You must
    finish and test this player to make sure it properly uses minimax and
    alpha-beta to return a good move before the search time limit expires.

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    iterative : boolean (optional)
        Flag indicating whether to perform fixed-depth search (False) or
        iterative deepening search (True).

    method : {'minimax', 'alphabeta'} (optional)
        The name of the search method to use in get_move().

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """

    def __init__(self, search_depth=3, score_fn=custom_score,
                 iterative=True, method='minimax', timeout=10.):
        self.search_depth = search_depth
        self.iterative = iterative
        self.score = score_fn
        self.method = method
        self.time_left = None
        self.TIMER_THRESHOLD = timeout

    def get_move(self, game, legal_moves, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        This function must perform iterative deepening if self.iterative=True,
        and it must use the search method (minimax or alphabeta) corresponding
        to the self.method value.

        **********************************************************************
        NOTE: If time_left < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        legal_moves : list<(int, int)>
            A list containing legal moves. Moves are encoded as tuples of pairs
            of ints defining the next (row, col) for the agent to occupy.

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """

        self.time_left = time_left

        # TODO: finish this function!

        # Perform any required initializations, including selecting an initial
        # move from the game board (i.e., an opening book), or returning
        # immediately if there are no legal moves

        try:
            # The search method call (alpha beta or minimax) should happen in
            # here in order to avoid timeout. The try/except block will
            # automatically catch the exception raised by the search method
            # when the timer gets close to expiring
            '''HLI CODE'''
            chosen_value = float('-inf')
            chosen_move = None
            depth = 0
            if self.method == 'minimax':
                if self.iterative:
                    print(legal_moves)
                    while True:
                        for move in legal_moves:
                            print(move)
                            value, _ = self.minimax(game.forecast_move(move), depth)
                            if chosen_value < value:
                                chosen_value = value
                                chosen_move = move
                        depth += 1
                else:
                    chosen_value, chosen_move = self.minimax(game, self.search_depth)
            elif self.method == 'alphabeta':
                if self.iterative:
                    while True:
                        for move in legal_moves:
                            value, _ = self.alphabeta(game.forecast_move(move), depth)
                            if chosen_value < value:
                                chosen_move = value
                                chosen_move = move
                        depth += 1
                else:
                    chosen_value, chosen_move = self.alphabeta(game, self.search_depth)
            else:
                raise ValueError(self.method + " not a valid method")
            return chosen_move
            '''HLI CODE'''
        except Timeout:
            # Handle any actions required at timeout, if necessary
            return chosen_move


    '''HLI CODE'''
    def max_value(self, game, depth, player, alpha=None, beta=None):
        """
        Args:
            GAME: the game instance that is currently being evaluated
            DEPTH: the number of levels to search
            PLAYER: the player instance whose turn it is right now / point of view (ie if 
                player is SELF, then SELF is trying to minimize value of current GAME)
            ALPHA: the lower bound of the max player's obtainable value 
            BETA: the upper bound of the min player's obtainable value
                (both of these numbers are the scores that each of these player's can definitely
                    obtain and thus anything unfavorable to these numbers in their point of view
                    can be ignored)
        Returns:
            The value of the max value from this node -> ie the best value that MAX player can 
            obtain at this node 
        """
        if depth == 0 or self.time_left() < self.TIMER_THRESHOLD:
            # note: i need to do self.score(game, self) and NOT self.score(game,player) -> when
            # i did this one it gave me wrong answer. This is because since max_value method
            # is returning the max utitliy for SELF then i need to return the score at the 
            # current game state in the point of view of SELF
            return self.score(game, self)
        value = float('-inf')
        # this gets the next moves for the current PLAYER - which is not necessarily SELF
        # we change this variable because legal moves of SELF vs SELF's opponent will be 
        # different
        moves = [move for move in game.get_legal_moves(player)]
        # then we loop through the 'active' player's move and forecast the gameboard with each
        # move and then since we denoted in the minimax method call that one of the player's
        # was MAXIMIZING then we want to return the max value (in the point of view of SELF)
        # from the minimum values that the other player will choose
        for move in moves:
            proposed_score = self.min_value(game.forecast_move(move), depth-1
                , game.get_opponent(player), alpha, beta)
            value = max(value, proposed_score)
            # if BETA (ie beta has a value because we are doing alpha-beta pruning) then we check
            # if value > beta. If it is we just return value. We do this not necessarily to return
            # value but to basically cut the for loop short (prune) the remaining nodes. we do this
            # because if value > beta then we know (1) max player will at least want to pick VALUE
            # (2) but since this VALUE is greater than BETA - min player will never choose it and 
            # therefore the rest of the nodes are irrelevant
            if beta:
                if value >= beta:
                    return value
                # if it isn't greater than beta, then it's possible that VALUE is a better lower
                # bound for MAX player and therefore update ALPHA when this happens. AND since VALUE
                # isn't greater than BETA this is a possible result 
                else:
                    alpha = max(alpha, value)
            if self.time_left() < self.TIMER_THRESHOLD:
                return value
        return value

    def min_value(self, game, depth, player, alpha=None, beta=None):
        """
        Args:
            GAME: the game instance that is currently being evaluated
            DEPTH: the number of levels to search
            PLAYER: the player instance whose turn it is right now / point of view (ie if 
                player is SELF, then SELF is trying to minimize value of current GAME)
            ALPHA: the lower bound of the max player's obtainable value 
            BETA: the upper bound of the min player's obtainable value
                (both of these numbers are the scores that each of these player's can definitely
                    obtain and thus anything unfavorable to these numbers in their point of view
                    can be ignored)
        Returns:
            The value of the min value from this node. ie the best that the MIN player can obtain
            at this node
        """
        if depth == 0 or self.time_left() < self.TIMER_THRESHOLD:
            return self.score(game, self)
        value = float('inf')
        moves = [move for move in game.get_legal_moves(player)]
        for move in moves:
            proposed_score = self.max_value(game.forecast_move(move), depth-1
                , game.get_opponent(player), alpha, beta)
            value = min(value, proposed_score)
            if alpha:
                # same logic as in MAX_VALUE but reversed. If value is less than the lower bound for
                # MAX player (ALPHA) then this will never be an option as MAX has ALPHA as an option
                # which is strictly better than value but MIN player will definitely choose VALUE
                # over anything > ALPHA therefore can cut the for loop short
                if value <= alpha:
                    return value
                else:
                    beta = min(beta, value)
            if self.time_left() < self.TIMER_THRESHOLD:
                return value
        return value
    '''HLI CODE'''

    def minimax(self, game, depth, maximizing_player=True):
        """Implement the minimax search algorithm as described in the lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        '''HLI CODE'''
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()
        chosen_move = game.get_player_location(self)
        if depth == 0:
            return (self.score(game, self), chosen_move)
        moves = [move for move in game.get_legal_moves(self)]
        if maximizing_player:
            value = float('-inf')
            for move in moves:
                proposed_score = self.min_value(game.forecast_move(move), depth-1
                    , game.get_opponent(self))
                if value < proposed_score:
                    value = proposed_score
                    chosen_move = move
                if self.time_left() < self.TIMER_THRESHOLD:
                    break
        else:
            value = float('inf')
            for move in moves:
                proposed_score = self.max_value(game.forecast_move(move), depth-1
                    , game.get_opponent(self))
                if value > proposed_score:
                    value = proposed_score
                    chosen_move = move
                if self.time_left() < self.TIMER_THRESHOLD:
                    break

        return (value, chosen_move)
        '''HLI CODE'''


    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf"), maximizing_player=True):
        """Implement minimax search with alpha-beta pruning as described in the
        lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        \depth : int//
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()
        moves = [move for move in game.get_legal_moves(self)]
        # if not moves:
        #     moves = [(-1,-1)]
        # if depth == 0:
        #     move_scores = [(self.score(game.forecast_move(move), move) for move in moves)]

        #     return (max(move_scores), chosen_move)

        if maximizing_player:
            value = float('-inf')
            for move in moves:
                proposed_score = self.min_value(game.forecast_move(move), depth-1
                    , game.get_opponent(self), alpha, beta)
                if proposed_score > value:
                    value = proposed_score
                    chosen_move = move
                    alpha = value
                    # if find a win - we can stop our search because beta always set to inf, 
                    # and eval function will set value to inf if find a forced win therefore if 
                    # eval function returns inf then we know we have a win
                    if alpha >= beta:
                        break
                if self.time_left() < self.TIMER_THRESHOLD:
                    break
        # else:
        #     value = float('inf')
        #     for move in moves:
        #         proposed_score = self.max_value(game.forecast_move(move), depth-1, game.get_opponent(self), alpha, beta)
        #         if proposed_score < value:  
        #             value = proposed_score
        #             chosen_move = move
        #             beta = value
        #         if self.time_left() < self.TIMER_THRESHOLD + 8:
        #             break
        return (value, chosen_move)