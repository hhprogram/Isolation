"""This file contains all the classes you must complete for this project.

You can use the test cases in agent_test.py to help during development, and
augment the test suite with your own test cases to further test your code.

You must test your agent's strength against a set of agents with known
relative strength using tournament.py and include the results in your report.
"""
import random
from sample_players import improved_score


class Timeout(Exception):
    """Subclass base exception for code clarity."""
    pass

def calc_dist(game, player, spot):
        """HLI CODE
        helper method that calculates the sq distance between PLAYER and SPOT (note: don't bother 
            with sq rooting as actual distance doesn't matter just need an ordering and sq root 
            the sq will have same ordering)
        Returns:
            euclidean straight line distance between the 2 (absolute value - since it is the 
                sq distance)
        """
        location = game.get_player_location(player)
        # print(location, spot)
        sq_dist = (location[0] - spot[0])**2 + (location[1] - spot[1])**2
        return float(sq_dist)

def closer_blank_spots(game, player):
    """ HLI CODE
    helper method that calculates the spots closer to PLAYER than PLAYER's 
    opponent
    Args:
        GAME: the game instance
        PLAYER: the player instance which we want to calculate the closer number of blank spaces
            available
    Returns:
        list of closer blank spots to PLAYER
    """
    closer_spots = []
    opponent = game.get_opponent(player)
    blank_spots = game.get_blank_spaces()
    for blank in blank_spots:
        my_distance = calc_dist(game, player, blank)
        opponent_distance = calc_dist(game, opponent, blank)
        if my_distance < opponent_distance:
            closer_spots.append(blank)
    return closer_spots

def is_partitioned(game):
    """
    helper method to determine if the players have been partitioned
    """



def custom_heuristics(game, player, choice):
    """HLI CODE
    wrapper method that calls the particular evaluation function based on variable CHOICE. All 
    functions returns -infiinity if legal moves for PLAYER == 0
    """
    def strategy_1():
        """
        evaluation function:
            (1) Favors spots closer to opponent
            (2) Favors spots farther away from the edges that opponent is not cloasest too. (ie if 
                opponent is in top left quandrant - favor moves that put us farther away from the
                bottom and right edge)
        """
        # get all spots along the edges
        top_edge = [(row,column) for row, column in zip([0]*game.height, range(game.width))]
        left_edge = [(column, row) for row, column in top_edge]
        right_edge = [(row, column+game.width-1) for row, column in left_edge]
        bottom_edge = [(row+game.height-1, column) for row, column in top_edge]
        dist_players = calc_dist(game, player, game.get_player_location(opponent))
        edges = [top_edge, right_edge, bottom_edge, left_edge]
        opp_closest_edge_spots = []
        min_dist = float('inf')
        # loop through each edge and append the spot on each edge closest to opponent
        for edge in edges:
            closest_spot = (-1,-1)
            for spot in edge:
                dist = calc_dist(game, opponent, spot)
                if dist < min_dist:
                    min_dist = dist
                    closest_spot = spot
            opp_closest_edge_spots.append((min_dist, closest_spot))

        opp_closest_edge_spots.sort()
        # then create a list of edge spots that are the closest to OPPONENT and throw out any that 
        # are farther away then the closest
        closest_spots = [spot for dist, spot in opp_closest_edge_spots if dist==opp_closest_edge_spots[0][0]]
        # then we want to (1) be as close as possible to opponent as possible but (2) not get closer
        # to the CLOSEST_SPOTS than the opponent is
        # print(game.get_player_location(player), game.get_player_location(opponent))
        opp_dist_score = (1/dist_players)
        edge_score = 0
        for spot in closest_spots:
            spot_dist = calc_dist(game, player, spot)
            if spot_dist != 0:
                edge_score += float(1/calc_dist(game, player, spot))
        for dist, spot in opp_closest_edge_spots:
            if spot not in closest_spot:
                spot_dist = calc_dist(game, player, spot)
                if spot_dist != 0:
                    edge_score += calc_dist(game, player, spot)
        return float(opp_dist_score + edge_score + moves_difference)


    def strategy_2():
        """
        if 2nd player - favors spots that are as close as possible to mirroring the opponent. If
        not then just use strategy_3
        """
        # checking if I went second - then try the mirroring heuristic or try to get to a square 
        # that is as close to possible to mirroring
        if game.move_count % 2 == 1:
            mir_opp_pos = (game.get_player_location(opponent)[1]
                , game.get_player_location(opponent)[0])
            if game.get_player_location(player) == mir_opp_pos:
                return float('inf')
            else:
                return float(1 / calc_dist(game, player, mir_opp_pos))
        else:
            return strategy_3()
    def strategy_3():
        """
        most complex evaluation function. evaluation function Calculates:
            (1) number of blank spaces on PLAYER's side of the board (ie the part of the board that 
                PLAYER is 'closer' to than his opponent - cells equadistant considered part of 
                'wall' and not incl. in sum) plus
            (2) distance between the two players - try to minimize that
            (3) if start of game, try to be as centered as possible
            (4) difference between the legal moves I have from this game instance and the legal
                moves my opponent has at this very instant
        """
        # checking if i am first mover - if so try to get center spot or the most centered spot if
        # height and width not odd
        if game.get_blank_spaces() == game.height*game.width:
            center_spot = (int(game.height/2), int(game.width/2))
            if game.get_player_location(player) == center_spot:
                return float('inf')
            else:
                return float(1/calc_dist(game, player, center_spot))
        dist_players = calc_dist(game, player, game.get_player_location(opponent))
        return float(len(closer_blank_spots(game, player)) - dist_players + moves_difference)

    opponent = game.get_opponent(player)
    num_legal_moves = len(game.get_legal_moves(player))
    num_opp_moves = len(game.get_legal_moves(opponent))
    moves_difference = num_legal_moves - num_opp_moves
    if game.is_loser(player):
            return float('-inf')
    if game.is_winner(player):
            return float('inf')

    if choice == 1:
        return strategy_1()
    elif choice == 2:
        return strategy_2()
    elif choice == 3:
        return strategy_3()
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
    return float(custom_heuristics(game,player,choice=1))
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
        self.unique_moves = []

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
            self.unique_moves = []
            if not legal_moves:
                return (-1,-1)
            if self.method == 'minimax':
                if self.iterative:
                    # print(legal_moves)
                    # Can just do this loop because for iterative deepening we just keep going down
                    # layer by layer until we are timed out, and since we 'catch' TimeOut exception
                    # we can properly return some value
                    while True:
                        # try to optimize get_move in beginning - basically if i'm 2nd player and
                        # the 1st player plays in the center. then i create a list of moves that are
                        # the same on each 'reflection' around the center move. All i do is create
                        # a list of top left corner locations and will use that to only search these
                        # possible moves. (trying to get better performing player by being able
                        # to search deeper)
                        if len(legal_moves) == game.width*game.height-1:
                            opp_location = game.get_player_location(game.get_opponent(self))
                            opp_row_location = opp_location[0]
                            if opp_location == (int(game.height/2), int(game.width/2)):
                                self.unique_moves = []
                                for column in range(opp_location[1]+1):
                                    self.unique_moves += [(row, column) for row, column in zip(range(opp_row_location),[column]*opp_row_location)]
                                self.unique_moves.remove(opp_location)

                        value, chosen_move = self.minimax(game, depth)
                        depth += 1
                else:
                    chosen_value, chosen_move = self.minimax(game, self.search_depth)
            elif self.method == 'alphabeta':
                if self.iterative:
                    while True:
                        value, chosen_move = self.alphabeta(game, depth)
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
    def max_value(self, game, depth, player, alpha=None, beta=None, opt_moves=None):
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
            OPT_MOVES: optional arg, only used by minimax or alphabeta if 2nd player and the
                opponent's first move is in the center
        Returns:
            The value of the max value from this node -> ie the best value that MAX player can 
            obtain at this node 
        """
        if depth == 0:
            # note: i need to do self.score(game, self) and NOT self.score(game,player) -> when
            # i did this one it gave me wrong answer. This is because since max_value method
            # is returning the max utitliy for SELF then i need to return the score at the 
            # current game state in the point of view of SELF
            return self.score(game, self), game.get_player_location(player)
        if self.time_left() < self.TIMER_THRESHOLD:
            return self.score(game, self), game.get_player_location(player)
        value = float('-inf')
        chosen_move = (-1,-1)
        if opt_moves:
            moves = self.unique_moves
        # this gets the next moves for the current PLAYER - which is not necessarily SELF
        # we change this variable because legal moves of SELF vs SELF's opponent will be 
        # different
        else:
            moves = [move for move in game.get_legal_moves(player)]
        # then we loop through the 'active' player's move and forecast the gameboard with each
        # move and then since we denoted in the minimax method call that one of the player's
        # was MAXIMIZING then we want to return the max value (in the point of view of SELF)
        # from the minimum values that the other player will choose
        for move in moves:
            proposed_score = self.min_value(game.forecast_move(move), depth-1
                , game.get_opponent(player), alpha, beta)
            if value < proposed_score:
                value = proposed_score
                chosen_move = move
            # if BETA (ie beta has a value because we are doing alpha-beta pruning) then we check
            # if value > beta. If it is we just return value. We do this not necessarily to return
            # value but to basically cut the for loop short (prune) the remaining nodes. we do this
            # because if value > beta then we know (1) max player will at least want to pick VALUE
            # (2) but since this VALUE is greater than BETA - min player will never choose it and 
            # therefore the rest of the nodes are irrelevant. note, has to be >= not just strictly
            # greater because since we choose the left most value even it it is equal to beta that
            # doesn't matter as min player would pick the node that resulted in the original beta
            # value therefore all other branches can be pruned
            if beta:
                if value >= beta:
                    return value, chosen_move
                # if it isn't greater than beta, then it's possible that VALUE is a better lower
                # bound for MAX player and therefore update ALPHA when this happens. AND since VALUE
                # isn't greater than BETA this is a possible result 
                else:
                    alpha = max(alpha, value)
            if self.time_left() < self.TIMER_THRESHOLD:
                return value, chosen_move
        return value, chosen_move

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
        if depth == 0:
            return self.score(game, self)
        if self.time_left() < self.TIMER_THRESHOLD:
            return self.score(game, self)
        value = float('inf')
        moves = [move for move in game.get_legal_moves(player)]
        # print(moves, " moves")
        for move in moves:
            # print(move, depth)
            proposed_score, _ = self.max_value(game.forecast_move(move), depth-1
                , game.get_opponent(player), alpha, beta)
            value = min(proposed_score, value)
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
            (2) I had to remember to get moves from game.active_player / feed in as argument 
                for min_value method game.get_opponent.active_player as I assumed SELF always was
                starting when this method got called - not true
        """
        '''HLI CODE'''
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        if maximizing_player:
            if self.unique_moves:
                value, chosen_move = self.max_value(game,depth, self, opt_moves=self.unique_moves)
            else:
                value, chosen_move = self.max_value(game, depth, self)
        # else:
        #     for move in moves:
        #         proposed_score = self.max_value(game.forecast_move(move), depth
        #             , game.get_opponent(active))
        #         if proposed_score < value:
        #             value = proposed_score
        #             chosen_move = move

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
            (2) I had to remember to get moves from game.active_player / feed in as argument 
                for min_value method game.get_opponent.active_player as I assumed SELF always was
                starting when this method got called - not true
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()
        if maximizing_player:
            if self.unique_moves:
                value, chosen_move = self.max_value(game,depth, self, alpha=alpha, beta=beta
                    , opt_moves=self.unique_moves)
            else:
                value, chosen_move = self.max_value(game, depth, self, alpha, beta)

        return (value, chosen_move)