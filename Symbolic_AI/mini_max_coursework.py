import time


class Game:
    def __init__(self):
        self.initialize_game()

    def initialize_game(self):

        ''' To initialise the game we need the n value, the m value, and the k value. Put
        the values in below'''

        self.m = 3
        self.n = 3
        self.k = 3
        # Create the grid using list comprehension, which represents positions on the board:
        self.grid = [[' ']*self.n for i in range(self.m)]

        # Define which value is the player. For Tic Tac Toe we have 'O' and 'X' so I use these.
        # Also, player 'X' always goes first.
        self.player_turn = 'X'

    def draw_board(self):
        ''' Draw and print the board (initially empty).'''
        for i in range(self.m):
            for j in range(self.n):
                print('{}|'.format(self.grid[i][j]), end=" ")
            print()
        print()

    def is_valid(self,pos_x,pos_y):
        ''' Is a position valid? Has to be within the board dimensions and
        the position can't have been placed before'''

        if pos_x<0 or pos_x>(self.m-1) or pos_y<0 or pos_y>(self.n-1):
            return False
        elif self.grid[pos_x][pos_y]!= ' ':
            return False
        else:
            return True

    def is_terminal(self):

        '''This was the most time consuming part of the code. To define when the game ends
        We have to take into account all the possible terminal states of the m x n grid.
        The terminal state is when there are 'k' 'X's or 'O's in a row, column, or along
        a diagonal. So we have to take each of these cases into consideration. I do this
        below. To tell if there are k placements in a row I use a method involving a commulative
        sum. '''

        ##### Vertical #####
        for i in range(self.n):
            for j in range(self.m):
                if self.grid[j][i] != ' ':
                    sum = 0
                    k = 0
                    while k+1+j<=(self.m-1) and k < self.k:
                        if self.grid[j+k+1][i] == self.grid[j][i]:
                            sum+=1
                        k+=1
                    if sum == self.k-1:
                        return self.grid[j][i]

        ###### Horizontal #######

        for i in range(self.m):
            for j in range(self.n):
                if self.grid[i][j] != ' ':
                    sum = 0
                    k = 0
                    while k+1+j<=(self.n-1) and k < self.k:
                        if self.grid[i][j+k+1] == self.grid[i][j]:
                            sum+=1
                        k+=1
                    if sum == self.k - 1:
                        return self.grid[i][j]

        ######## Diagonals ########

        '''Diagonal from top left to bottom right direction

        # For this I try and do it iteratively. The idea is I take a position on the board
        # and see the diagonals using a while loop and the value for k. Then, if
        # the commulative sum is equal to k - 1 (i don't include original placement) we know that the game is won.'''

        for i in range(self.m):
            for j in range(self.n):
                if self.grid[i][j] != ' ':
                    sum = 0
                    k = 0
                    while k+1+j <= (self.n-1) and k+i+1 <= (self.m-1) and k < self.k:
                        if self.grid[i+k+1][j+k+1] == self.grid[i][j]:
                            sum += 1
                        k+=1
                    if sum == self.k - 1:
                        return self.grid[i][j]

        '''Other Diagonal '''

        for i in range(self.m):
            for j in range(self.n):
                if self.grid[i][j] != ' ':
                    sum = 0
                    k = 0
                    while j - k - 1 >= 0 and k+i+1 <= (self.m-1) and k < self.k:
                        if self.grid[i+k+1][j-k-1] == self.grid[i][j]:
                            sum += 1
                        k+=1
                    if sum == self.k - 1:
                        return self.grid[i][j]

        '''If there are any positions left so the game isn't over - want to continue game'''

        for i in range(self.m):
            for j in range(self.n):
                if self.grid[i][j] == ' ':
                    return None


        '''If all positions are taken up and no one has won then we know the game is tied.'''

        return 'Game Tied'

    ############### Min_Max No pruning ###################

    def max(self):
        '''Below is the version of max function without alpha-beta pruning.
        First I used this to calculate the times taken when not using alpha-beta pruning'''

        # Initalise max value that is smaller than any value we will get after.
        max_value = -2

        # Initalise positions on board

        pos_x = None
        pos_y = None

        # Set outcomes when the board is terminal.

        terminal_outcome = self.is_terminal()

        if terminal_outcome == 'X':
            return (-1,0,0)
        elif terminal_outcome == 'O':
            return (1,0,0)
        elif terminal_outcome == 'Game Tied':
            return (0,0,0)

        # Compute min_max algorithm for max

        for i in range(self.m):
            for j in range(self.n):
                if self.grid[i][j] == ' ':
                    self.grid[i][j] = 'O'
                    (m,min_i,min_j) = self.min()

                    if m > max_value:
                        max_value = m
                        #print(m)
                        pos_x = i
                        pos_y = j

                    self.grid[i][j] = ' '

        return(max_value, pos_x, pos_y)

    def min(self):

        '''Below is the version of min function without alpha-beta pruning.
        First I used this to calculate the times taken when not using alpha-beta pruning '''

        # Initialise a min value that is bigger than any 'terminal_outcome' value later.
        min_value = 2

        # Initialise positions on the board

        qpos_x = None
        qpos_y = None

        # Set outcomes when the board is terminal.

        terminal_outcome = self.is_terminal()

        if terminal_outcome == 'X':
            return (-1,0,0)
        elif terminal_outcome == 'O':
            return (1,0,0)
        elif terminal_outcome == 'Game Tied':
            return (0,0,0)

        # Compute min_max algorithm for min

        for i in range(self.m):
            for j in range(self.n):
                if self.grid[i][j] == ' ':
                    self.grid[i][j] = 'X'
                    (m,max_i,max_j) = self.max()
                    if m < min_value:
                        min_value = m
                        #print(min_value)
                        qpos_x = i
                        qpos_y = j

                    self.grid[i][j] = ' '

        return (min_value,qpos_x,qpos_y)

    def play_min_max_no_pruning(self):

        ''' Function to play the game without alpha-beta pruning'''

        while True:
            # Draw the board
            self.draw_board()
            # Determine if the game is finished or not.
            self.terminal_outcome = self.is_terminal()

            if self.terminal_outcome != None:
                if self.terminal_outcome == 'X':
                    print('X is the Winner')
                elif self.terminal_outcome == 'O':
                    print('O is the Winner')
                elif self.terminal_outcome == 'Game Tied':
                    print('It is a Tie')

                # If game is won: re-initialise and return

                self.initialize_game()
                return

            # When it is player X's turn (i.e. the human with suggestions based on 'min'):

            if self.player_turn == 'X':
                # infinite loop until break so non valid moves you can play again
                while True:
                    # Take the time for the algorithm to compute the moves
                    start_time = time.time()
                    (m, qpos_x, qpos_y) = self.min()
                    #print(m)
                    # end the time
                    end_time = time.time()
                    # print the total time taken
                    time_taken = float(end_time) - float(start_time)
                    print(f'Time taken is {time_taken}')
                    print(f'Take the move X = {qpos_x}, Y = {qpos_y}')

                    pos_x = int(input('Put in the X (row) coordinate: '))
                    pos_y = int(input('Put in the Y (column) coordinate: '))

                    (qpos_x,qpos_y) = (pos_x,pos_y)

                    if self.is_valid(pos_x,pos_y):
                        self.grid[pos_x][pos_y] = 'X'
                        # change who's turn it is
                        self.player_turn = 'O'
                        # Break the while loop
                        break
                    else:
                        # If move is not valid we need to print something and try again
                        print('The move is not valid')

            # When it is player O's turn (i.e. the max AI)

            else:
                (m,pos_x,pos_y) = self.max()
                self.grid[pos_x][pos_y] = 'O'
                #print(m)
                self.player_turn = 'X'

    ################# Min_Max with alpha-beta pruning ##################

    ''' Below is the alpha-beta pruning Min and Max functions. The code is pretty much
    the same as min-max without pruning, however at the end of both the max and min we have that there
    are two 'if' functions which implement the alpha-beta pruning.'''

    def max_alpha_beta_pruning(self, alpha, beta):

        # Initialise a max value that is smaller than any 'terminal_outcome' value.

        max_value = -2

        # Initialise positions on board.

        pos_x = None
        pos_y = None

        terminal_outcome = self.is_terminal()

        # Set the min and max returns depending on the terminal state of the board.
        # Value of -1 if min wins, value of 1 if max wins, and 0 for a draw.

        if terminal_outcome == 'X':
            return (-1,0,0)
        elif terminal_outcome == 'O':
            return (1,0,0)
        elif terminal_outcome == 'Game Tied':
            return (0,0,0)

        # Implement min_max algorithm for max

        for i in range(self.m):
            for j in range(self.n):
                if self.grid[i][j] == ' ':
                    self.grid[i][j] = 'O'
                    (m,min_i,min_j) = self.min_alpha_beta_pruning(alpha,beta)

                    if m > max_value:
                        max_value = m
                        #print(m)
                        pos_x = i
                        pos_y = j

                    self.grid[i][j] = ' '

                    # Implement alpha beta pruning for max

                    if max_value >= beta:
                        #print(f'we have pruned {max_value},{beta}')
                        # prune if condition satisfied
                        return(max_value, pos_x, pos_y)

                    if max_value > alpha:
                        #print(f'alpha value is {alpha}')
                        # change alpha value
                        alpha = max_value

        return (max_value, pos_x, pos_y)

    def min_alpha_beta_pruning(self, alpha, beta):

        # Initialise a min value that is larger than any 'terminal_outcome' value.

        min_value = 2

        qpos_x = None
        qpos_y = None

        terminal_outcome = self.is_terminal()

        # Set the min and max returns based on terminal state of the board.
        # Value of -1 if min wins, value if 1 if max wins, value of 0 if draw.

        if terminal_outcome == 'X':
            return (-1,0,0)
        elif terminal_outcome == 'O':
            return (1,0,0)
        elif terminal_outcome == 'Game Tied':
            return (0,0,0)

        # Implement min_max algorithm for min

        for i in range(self.m):
            for j in range(self.n):
                if self.grid[i][j] == ' ':
                    self.grid[i][j] = 'X'
                    # start with alpha, beta = -2,2... change as we go
                    (m,max_i,max_j) = self.max_alpha_beta_pruning(alpha,beta)
                    if m < min_value:
                        min_value = m
                        #print(min_value)
                        qpos_x = i
                        qpos_y = j

                    self.grid[i][j] = ' '

                    # Implement alpha-beta pruning for min

                    if min_value <= alpha:
                        # prune if condition satisfied
                        #print(f'we have pruned {min_value}, {alpha}')
                        return(min_value, qpos_x, qpos_y)

                    if min_value < beta:
                        #print(f'beta value: {beta}')
                        # change beta value if no pruning
                        beta = min_value

        return (min_value,qpos_x,qpos_y)


    def play_alpha_beta_pruning(self):

        ''' Function to play the game with alpha-beta pruning'''

        while True:
            # Draw the board
            self.draw_board()
            # Determine if the game is finished or not.
            self.terminal_outcome = self.is_terminal()

            if self.terminal_outcome != None:
                if self.terminal_outcome == 'X':
                    print('X is the Winner')
                elif self.terminal_outcome == 'O':
                    print('O is the Winner')
                elif self.terminal_outcome == 'Game Tied':
                    print('It is a Tie')

                # If game won re-initialise and return

                self.initialize_game()
                return

            # When it is player X's turn (i.e. the human with suggestions based on 'min'):

            if self.player_turn == 'X':
                # infinite loop until break (so if not valid moves can do again)
                while True:
                    # Take the time for the algorithm to compute the moves
                    start_time = time.time()
                    (m, qpos_x, qpos_y) = self.min_alpha_beta_pruning(-2,2)
                    #print(m)
                    # end the time
                    end_time = time.time()
                    # print the total time taken
                    time_taken = float(end_time) - float(start_time)
                    print(f'Time taken is {time_taken}')
                    print(f'Take the move X = {qpos_x}, Y = {qpos_y}')

                    pos_x = int(input('Put in the X (row) coordinate: '))
                    pos_y = int(input('Put in the Y (column) coordinate: '))

                    (qpos_x,qpos_y) = (pos_x,pos_y)

                    if self.is_valid(pos_x,pos_y):
                        self.grid[pos_x][pos_y] = 'X'
                        # change who's turn it is
                        self.player_turn = 'O'
                        break
                    else:
                        print('The move is not valid')

            # When it is player O's turn (i.e. the max AI)

            else:
                (m,pos_x,pos_y) = self.max_alpha_beta_pruning(-2,2)
                self.grid[pos_x][pos_y] = 'O'
                #print(m)
                self.player_turn = 'X'

def main():
    # Play the game
    g = Game()
    # To play with no pruning
    #g.play_min_max_no_pruning()
    # To play with pruning
    g.play_alpha_beta_pruning()

if __name__ == '__main__':
    main()
