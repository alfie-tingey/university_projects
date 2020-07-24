from typing import List, Tuple
from battleship.ship import Ship
import random

OFFSET_UPPER_CASE_CHAR_CONVERSION = 64

class Board(object):
    """
    Class representing the board of the player. Interface between the player and its ships.
    """
    SIZE_X = 10  # length of the rectangular board, along the x axis
    SIZE_Y = 10  # length of the rectangular board, along the y axis

    # dict: length -> number of ships of that length
    DICT_NUMBER_SHIPS_PER_LENGTH = {1: 1,
                                    2: 1,
                                    3: 1,
                                    4: 1,
                                    5: 1}

    def __init__(self,
                 list_ships: List[Ship]):
        """
        :param list_ships: list of ships for the board.
        :raise ValueError if the list of ships is in contradiction with Board.DICT_NUMBER_SHIPS_PER_LENGTH.
        :raise ValueError if there are some ships that are too close from each other
        """

        self.list_ships = list_ships
        self.set_coordinates_previous_shots = set()

        if not self.lengths_of_ships_correct():
            total_number_of_ships = sum(self.DICT_NUMBER_SHIPS_PER_LENGTH.values())

            error_message = f"There should be {total_number_of_ships} ships in total:\n"

            for length_ship, number_ships in self.DICT_NUMBER_SHIPS_PER_LENGTH.items():
                error_message += f" - {number_ships} of length {length_ship}\n"

            raise ValueError(error_message)

        if self.are_some_ships_too_close_from_each_other():
            raise ValueError("There are some ships that are too close from each other.")

    def has_no_ships_left(self) -> bool:
        """
        :return: True if and only if all the ships on the board have sunk.
        """
        # TODO
        self.ships_sunk = ([ship for ship in self.list_ships if ship.has_sunk()])
        return len(self.ships_sunk) == sum(self.DICT_NUMBER_SHIPS_PER_LENGTH.values())

    def is_attacked_at(self, coord_x: int, coord_y: int) -> Tuple[bool, bool]:
        """
        The board receives an attack at the position (coord_x, coord_y).
        - if there is no ship at that position -> nothing happens
        - if there is a ship at that position -> it is damaged at that coordinate

        :param coord_x: integer representing the projection of a coordinate on the x-axis
        :param coord_y: integer representing the projection of a coordinate on the y-axis
        :return: a tuple of bool variables (is_ship_hit, has_ship_sunk) where:
                    - is_ship_hit is True if and only if the attack was performed at a set of coordinates where an
                    opponent's ship is.
                    - has_ship_sunk is True if and only if that attack made the ship sink.
        """

        is_ship_hit, has_ship_sunk = False, False
        for ship in self.list_ships:
            ship.gets_damage_at(coord_x,coord_y)
            if not ship.is_damaged_at(coord_x,coord_y):
                self.set_coordinates_previous_shots.add((coord_x,coord_y))
            elif ship.is_damaged_at(coord_x,coord_y):
               is_ship_hit = True
               if ship.has_sunk():
                   has_ship_sunk = True
               break
        return((is_ship_hit,has_ship_sunk))


    def print_board_with_ships_positions(self) -> None:
        array_board = [[' ' for _ in range(self.SIZE_X)] for _ in range(self.SIZE_Y)]

        for x_shot, y_shot in self.set_coordinates_previous_shots:
            array_board[y_shot - 1][x_shot - 1] = 'O'

        for ship in self.list_ships:
            if ship.has_sunk():
                for x_ship, y_ship in ship.set_all_coordinates:
                    array_board[y_ship - 1][x_ship - 1] = '$'
                continue

            for x_ship, y_ship in ship.set_all_coordinates:
                array_board[y_ship - 1][x_ship - 1] = 'S'

            for x_ship, y_ship in ship.set_coordinates_damages:
                array_board[y_ship - 1][x_ship - 1] = 'X'

        board_str = self._get_board_string_from_array_chars(array_board)

        print(board_str)

    def print_board_without_ships_positions(self) -> None:
        array_board = [[' ' for _ in range(self.SIZE_X)] for _ in range(self.SIZE_Y)]

        for x_shot, y_shot in self.set_coordinates_previous_shots:
            array_board[y_shot - 1][x_shot - 1] = 'O'

        for ship in self.list_ships:
            if ship.has_sunk():
                for x_ship, y_ship in ship.set_all_coordinates:
                    array_board[y_ship - 1][x_ship - 1] = '$'
                continue

            for x_ship, y_ship in ship.set_coordinates_damages:
                array_board[y_ship - 1][x_ship - 1] = 'X'

        board_str = self._get_board_string_from_array_chars(array_board)

        print(board_str)

    def _get_board_string_from_array_chars(self, array_board: List[List[str]]) -> str:
        list_lines = []

        array_first_line = [chr(code + OFFSET_UPPER_CASE_CHAR_CONVERSION) for code in range(1, self.SIZE_X + 1)]
        first_line = ' ' * 6 + (' ' * 5).join(array_first_line) + ' \n'

        for index_line, array_line in enumerate(array_board, 1):
            number_spaces_before_line = 2 - len(str(index_line))
            space_before_line = number_spaces_before_line * ' '
            list_lines.append(f'{space_before_line}{index_line} |  ' + '  |  '.join(array_line) + '  |\n')

        line_dashes = '   ' + '-' * 6 * self.SIZE_X + '-\n'

        board_str = first_line + line_dashes + line_dashes.join(list_lines) + line_dashes

        return board_str

    def lengths_of_ships_correct(self) -> bool:
        """
        :return: True if and only if there is the right number of ships of each length, according to
        Board.DICT_NUMBER_SHIPS_PER_LENGTH
        """
        # TODO
        self.lengths_of_ships = []
        for ship in self.list_ships:
            self.lengths_of_ships.append(int(ship.length()))

        correct_ships = []
        for ship_length in self.DICT_NUMBER_SHIPS_PER_LENGTH:
            number_of_ships = self.DICT_NUMBER_SHIPS_PER_LENGTH[ship_length]
            for x in range(number_of_ships):
                correct_ships.append(int(ship_length))

        self.lengths_of_ships.sort()
        return correct_ships == self.lengths_of_ships

    def are_some_ships_too_close_from_each_other(self) -> bool:
        """
        :return: True if and only if there are at least 2 ships on the board that are near each other.
        """
        # TODO
        ships_are_near = False
        for i in range(len(self.list_ships)):
            for j in range(len(self.list_ships)):
                if i == j:
                    continue
                else:
                    ship = self.list_ships[i]
                    other_ship = self.list_ships[j]
                    if ship.is_near_ship(other_ship):
                        ships_are_near = True
                        break
                    else:
                        ships_are_near = False

        return ships_are_near

class BoardAutomatic(Board):
    def __init__(self):
        super().__init__(list_ships=self.generate_ships_automatically())

    def create_ship_of_certain_length(self, length):
        """Here I create chips depending on the length given. Then I make sure the ships
        are either horizonal or vertical"""

        orientation = random.choice(['horizontal','vertical'])

        if orientation == 'horizontal':
            self.x_random_start = random.randint(1,self.SIZE_X-length+1)
            self.y_random_start = random.randint(1,self.SIZE_Y)
            self.x_random_end = self.x_random_start + length - 1
            self.y_random_end = self.y_random_start
        else:
            self.x_random_start = random.randint(1,self.SIZE_X)
            self.y_random_start = random.randint(1,self.SIZE_Y - length+1)
            self.x_random_end = self.x_random_start
            self.y_random_end = self.y_random_start + length - 1

        self.coord_start = self.x_random_start,self.y_random_start
        self.coord_end = self.x_random_end, self.y_random_end
        self.ship = Ship(coord_start = self.coord_start, coord_end = self.coord_end)

        return self.ship

    def is_near_random_coordinate(self, coord_x: int, coord_y: int) -> bool:
        return self.x_random_start - 1 <= coord_x <= self.x_random_end + 1 \
               and self.y_random_start - 1 <= coord_y <= self.y_random_end + 1

    def is_near_random_ship(self, other_ship: 'Ship') -> bool:
        """
        :param other_ship: other object of class Ship
        :return: True if and only if there is a coordinate of other_ship that is near this ship.
        """
        ship_near = False
        if other_ship.x_start == other_ship.x_end:
            for i in range(int(other_ship.y_end - other_ship.y_start + 1)):
                if self.is_near_random_coordinate(other_ship.x_start, other_ship.y_start + i):
                    ship_near = True
        elif other_ship.y_start == other_ship.y_end:
            for i in range(int(other_ship.x_end - other_ship.x_start + 1)):
                if self.is_near_random_coordinate(other_ship.x_start + i, other_ship.y_start):
                    ship_near = True

        return ship_near


    def generate_ships_automatically(self) -> List[Ship]:
        """
        :return: A list of automatically (randomly) generated ships for the board
        """
        # TODO

        automatic_ships = list(self.DICT_NUMBER_SHIPS_PER_LENGTH.items())
        self.list_ships_auto = []
        for i in range(len(automatic_ships)):
            j = 1
            while j <= automatic_ships[i][1]:
                new_ship = self.create_ship_of_certain_length(automatic_ships[i][0])
                while True:
                    if any(self.is_near_random_ship(ships) for ships in self.list_ships_auto):
                        new_ship = self.create_ship_of_certain_length(automatic_ships[i][0])
                    else:
                        break
                self.list_ships_auto.append(new_ship)
                j = j+1

        #print(self.list_ships_auto)
        return self.list_ships_auto

if __name__ == '__main__':
    # SANDBOX for you to play and test your functions
    list_ships = [
        Ship(coord_start=(1, 1), coord_end=(1, 1)),
        Ship(coord_start=(3, 3), coord_end=(3, 4)),
        Ship(coord_start=(5, 3), coord_end=(5, 5)),
        Ship(coord_start=(7, 1), coord_end=(7, 4)),
        Ship(coord_start=(9, 1), coord_end=(9, 5)),
    ]

    boardA = BoardAutomatic() #added
    list_ships = boardA.generate_ships_automatically() #added

    board = Board(list_ships)
    board.print_board_with_ships_positions()
    board.print_board_without_ships_positions()
    print(board.is_attacked_at(5, 4),
          board.is_attacked_at(10, 9), board.is_attacked_at(5,3), board.is_attacked_at(1,1), board.is_attacked_at(5,5),board.is_attacked_at(10,10))
    print(board.set_coordinates_previous_shots)
    print(board.lengths_of_ships_correct())
    print(board.are_some_ships_too_close_from_each_other())

    # Testing for automatic board placement of ships
    print(list_ships)
