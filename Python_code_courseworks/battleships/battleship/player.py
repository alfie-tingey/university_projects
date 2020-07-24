import random
from typing import Tuple


# All lines below used to be battleship. .... Ask in tutorial
from battleship.board import Board, BoardAutomatic
from battleship.ship import Ship
from battleship.convert import get_tuple_coordinates_from_str, get_str_coordinates_from_tuple


class Player(object):
    """
    Class representing the player
    - chooses where to perform an attack
    """
    index_player = 0

    def __init__(self,
                 board: Board,
                 name_player: str = None,
                 ):
        Player.index_player += 1

        self.board = board

        if name_player is None:
            self.name_player = "player_" + str(self.index_player)
        else:
            self.name_player = name_player

    def __str__(self):
        return self.name_player

    def attacks(self,
                opponent) -> Tuple[bool, bool]:
        """
        :param opponent: object of class Player representing the person to attack
        :return: a tuple of bool variables (is_ship_hit, has_ship_sunk) where:
                    - is_ship_hit is True if and only if the attack was performed at a set of coordinates where an
                    opponent's ship is.
                    - has_ship_sunk is True if and only if that attack made the ship sink.
        """

        assert isinstance(opponent, Player)

        print(f"Here is the current state of {opponent}'s board before {self}'s attack:\n")
        opponent.print_board_without_ships()

        coord_x, coord_y = self.select_coordinates_to_attack(opponent)

        print(f"{self} attacks {opponent} "
              f"at position {get_str_coordinates_from_tuple(coord_x, coord_y)}")

        is_ship_hit, has_ship_sunk = opponent.is_attacked_at(coord_x, coord_y)

        if has_ship_sunk:
            print(f"\nA ship of {opponent} HAS SUNK. {self} can play another time.")
        elif is_ship_hit:
            print(f"\nA ship of {opponent} HAS BEEN HIT. {self} can play another time.")
        else:
            print("\nMissed".upper())

        return is_ship_hit, has_ship_sunk

    def is_attacked_at(self,
                       coord_x: int,
                       coord_y: int
                       ) -> Tuple[bool, bool]:
        """
        :param coord_x: integer representing the projection of a coordinate on the x-axis
        :param coord_y: integer representing the projection of a coordinate on the y-axis
        :return: a tuple of bool variables (is_ship_hit, has_ship_sunk) where:
                    - is_ship_hit is True if and only if the attack was performed at a set of coordinates where a
                    ship is (on the board owned by the player).
                    - has_ship_sunk is True if and only if that attack made the ship sink.
        """
        # TODO
        return self.board.is_attacked_at(coord_x,coord_y)

    def select_coordinates_to_attack(self, opponent) -> Tuple[int, int]:
        """
        Abstract method, for choosing where to perform the attack
        :param opponent: object of class Player representing the player under attack
        :return: a tuple of coordinates (coord_x, coord_y) at which the next attack will be performed
        """
        raise NotImplementedError

    def has_lost(self) -> bool:
        """
        :return: True if and only if all the ships of the player have sunk
        """
        # TODO
        return self.board.has_no_ships_left()

    def print_board_with_ships(self):
        self.board.print_board_with_ships_positions()

    def print_board_without_ships(self):
        self.board.print_board_without_ships_positions()


class PlayerUser(Player):
    """
    Player representing a user playing manually
    """

    def select_coordinates_to_attack(self, opponent: Player) -> Tuple[int, int]:
        """
        Overrides the abstract method of the parent class.
        :param opponent: object of class Player representing the player under attack
        :return: a tuple of coordinates (coord_x, coord_y) at which the next attack will be performed
        """
        print(f"It is now {self}'s turn.")

        while True:
            try:
                coord_str = input('coordinates target = ')
                coord_x, coord_y = get_tuple_coordinates_from_str(coord_str)
                return coord_x, coord_y
            except ValueError as value_error:
                print(value_error)


class PlayerAutomatic(Player):
    """
    Player playing automatically using a strategy.
    """

    def __init__(self, name_player: str = None):
        board = BoardAutomatic()
        super().__init__(board, name_player)
        self.name = name_player
        self.set_positions_previously_attacked = set()
        self.last_hit_coord = None
        self.list_ships_opponent_previously_sunk = []
        self.positions_to_attack = []\

    def select_coordinates_to_attack(self, opponent: Player) -> tuple:
        """
        Overrides the abstract method of the parent class.
        :param opponent: object of class Player representing the player under attack
        :return: a tuple of coordinates (coord_x, coord_y) at which the next attack will be performed
        """
        """Explanation: The way I have implemented the AI player is that firstly I take a random coordinate
        to attack. Then, once a random attack hits a ship the AI keeps track of all possible nearby positions
        that the ship could take up considering that the ship is either horizontal or vertical (in positions_to_attack).
        Then,the AI either attacks above, below, to the left, or to the right of the damaged coordinate. once
        the AI knows the second position of damage, it then knows the orientation of the ship and only attacks
        in that direction. Once the ship is sunk the AI loses the knowledge of the surrounding positions on the ship
        and attacks at random again. Additionally, any locations that are off the board are removed from possible
        positions to attack by the 'remove_invalid' function. It is definitely better than random but not that great."""

        if len(self.positions_to_attack) == 0:
            position_to_attack = self.select_random_coordinates_to_attack()
            coord_x = position_to_attack[0]
            coord_y = position_to_attack[1]

            attack = opponent.is_attacked_at(coord_x,coord_y) # Used to be self
            if attack[1]:
                # go random
                return position_to_attack

            if attack[0]:
                above = (coord_x, coord_y+1)
                left = (coord_x-1, coord_y)
                below = (coord_x, coord_y-1)
                right = (coord_x+1, coord_y)
                self.positions_to_attack += [above, below, left, right]
                # hit
                self.last_hit_coord = position_to_attack
                self.set_positions_previously_attacked.add(position_to_attack)
                self.positions_to_attack = self.remove_invalid()
                return position_to_attack
            else:
                # miss
                self.set_positions_previously_attacked.add(position_to_attack)
                return position_to_attack
        else:
            position_to_attack = self.positions_to_attack.pop(0)
            coord_x = position_to_attack[0]
            coord_y = position_to_attack[1]
            attack = opponent.is_attacked_at(coord_x,coord_y)
            if attack[1]:
                self.positions_to_attack = []
                return position_to_attack
            if attack[0]:
                # hit
                if position_to_attack[0] == self.last_hit_coord[0]:
                    # vertical
                    self.positions_to_attack = [
                        position for position in self.positions_to_attack
                        if position[0] == position_to_attack[0]
                    ]
                    difference = position_to_attack[1] - self.last_hit_coord[1]
                    self.positions_to_attack.append( (position_to_attack[0], position_to_attack[1]+difference) )
                    self.last_hit_coord = position_to_attack
                    self.set_positions_previously_attacked.add(position_to_attack)
                    self.positions_to_attack = self.remove_invalid()
                    return position_to_attack
                elif position_to_attack[1] == self.last_hit_coord[1]:
                    # horizontal
                    self.positions_to_attack = [
                        position for position in self.positions_to_attack
                        if position[1] == position_to_attack[1]
                    ]
                    difference = position_to_attack[0] - self.last_hit_coord[0]
                    self.positions_to_attack.append( (position_to_attack[0]+difference, position_to_attack[1]) )
                    self.last_hit_coord = position_to_attack
                    self.set_positions_previously_attacked.add(position_to_attack)
                    self.positions_to_attack = self.remove_invalid()
                    return position_to_attack
            else:
                # miss
                self.set_positions_previously_attacked.add(position_to_attack)
                return position_to_attack

    def remove_invalid(self) -> list:
        valid_positions = []
        for position in self.positions_to_attack:
            # check x coord is valid
            if (position[0] >= 0 and position[0] <= 9) and (position[1] >= 0 and position[1] <= 9):
                valid_positions.append(position)
        return valid_positions

    def select_random_coordinates_to_attack(self) -> tuple:
        has_position_been_previously_attacked = True
        is_position_near_previously_sunk_ship = True
        coord_random = None

        while has_position_been_previously_attacked or is_position_near_previously_sunk_ship:
            coord_random = self._get_random_coordinates()

            has_position_been_previously_attacked = coord_random in self.set_positions_previously_attacked
            is_position_near_previously_sunk_ship = self._is_position_near_previously_sunk_ship(coord_random)

        return coord_random

    def _get_random_coordinates(self) -> tuple:
        coord_random_x = random.randint(1, self.board.SIZE_X)
        coord_random_y = random.randint(1, self.board.SIZE_Y)

        coord_random = (coord_random_x, coord_random_y)

        return coord_random

    def _is_position_near_previously_sunk_ship(self, coord: tuple) -> bool:
        for ship_opponent in self.list_ships_opponent_previously_sunk:  # type: Ship
            if ship_opponent.has_sunk() and ship_opponent.is_near_coordinate(*coord):
                return True
        return False

class PlayerRandom(Player):
    def __init__(self, name_player: str = None):
        board = BoardAutomatic()
        self.set_positions_previously_attacked = set()
        self.last_attack_coord = None
        self.list_ships_opponent_previously_sunk = []

        super().__init__(board, name_player)

    def select_coordinates_to_attack(self, opponent: Player) -> tuple:
        position_to_attack = self.select_random_coordinates_to_attack()

        self.set_positions_previously_attacked.add(position_to_attack)
        self.last_attack_coord = position_to_attack
        return position_to_attack

    def select_random_coordinates_to_attack(self) -> tuple:
        has_position_been_previously_attacked = True
        is_position_near_previously_sunk_ship = True
        coord_random = None

        while has_position_been_previously_attacked or is_position_near_previously_sunk_ship:
            coord_random = self._get_random_coordinates()

            has_position_been_previously_attacked = coord_random in self.set_positions_previously_attacked
            is_position_near_previously_sunk_ship = self._is_position_near_previously_sunk_ship(coord_random)

        return coord_random

    def _get_random_coordinates(self) -> tuple:
        coord_random_x = random.randint(1, self.board.SIZE_X)
        coord_random_y = random.randint(1, self.board.SIZE_Y)

        coord_random = (coord_random_x, coord_random_y)

        return coord_random

    def _is_position_near_previously_sunk_ship(self, coord: tuple) -> bool:
        for ship_opponent in self.list_ships_opponent_previously_sunk:  # type: Ship
            if ship_opponent.has_sunk() and ship_opponent.is_near_coordinate(*coord):
                return True
        return False

if __name__ == '__main__':
    # SANDBOX for you to play and test your functions

    list_ships = [
        Ship(coord_start=(1, 1), coord_end=(1, 1)),
        Ship(coord_start=(3, 3), coord_end=(3, 4)),
        Ship(coord_start=(5, 3), coord_end=(5, 5)),
        Ship(coord_start=(7, 1), coord_end=(7, 4)),
        Ship(coord_start=(9, 3), coord_end=(9, 7)),
    ]

    board = Board(list_ships)
    player = PlayerUser(board)
    print(player.is_attacked_at(5, 4))
    print(player.is_attacked_at(5, 3))
    print(player.is_attacked_at(5, 5))
    print(player.is_attacked_at(1, 1))
    print(player.is_attacked_at(3, 3))
    print(player.is_attacked_at(3, 4))
    print(player.is_attacked_at(7, 1))
    print(player.is_attacked_at(7, 2))
    print(player.is_attacked_at(7, 3))
    print(player.is_attacked_at(7, 4))
    print(player.is_attacked_at(9, 3))
    print(player.is_attacked_at(9, 4))
    print(player.is_attacked_at(9, 5))
    print(player.is_attacked_at(9, 6))
    print(player.is_attacked_at(9, 7))
    print(player.has_lost())
