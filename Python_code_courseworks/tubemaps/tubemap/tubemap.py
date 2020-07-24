from typing import Set, List
import json


class StationDoesNotExist(Exception):
    def __init__(self, name_element):
        self.message = f"The station '{name_element}' does not exist in the Loaded tube map."
        super(StationDoesNotExist, self).__init__(self.message)


class TubeMap(object):
    """
    This class has two main attributes:
    - graph_tube_map
    - set_zones_per_station

    The attribute graph_tube_map should have the following form:

    {
        "station_A": {
            "neighbour_station_1": [
                {
                    "line": "name of a line between station_A and neighbour_station_1",
                    "time": "time it takes in minutes to go from station_A to neighbour_station_1 WITH THAT line"
                },
                {
                    "line": "name of ANOTHER line between station_A and neighbour_station_1",
                    "time": "time it takes in minutes to go from station_A to neighbour_station_1 with that OTHER line"
                }
            ],
            "neighbour_station_2": [
                {
                    "line": "name of line between station_A and neighbour_station_2",
                    "time": "time it takes in minutes to go from station_A to neighbour_station_2"
                }
            ]
        }

        "station_B": {
            ...
        }

        ...

    }

        Also, for instance:
        self.graph_tube_map['Hammersmith'] should be equal to:
        {
            'Barons Court': [
                {'line': 'District Line', 'time': 1},
                {'line': 'Piccadilly Line', 'time': 2}
            ],
            'Ravenscourt Park': [
                {'line': 'District Line', 'time': 2}
            ],
            'Goldhawk Road': [
                {'line': 'Hammersmith & City Line', 'time': 2}
            ],
            'Turnham Green': [
                {'line': 'Piccadilly Line', 'time': 2}
            ]
        }

    The attribute set_zones_per_station should have the following form:
    {
        station_1: {zone_a},
        station_2: {zone_a, zone_b},
        ...
    }
    For example, with the London tube map,
    self.set_zones_per_station["Turnham Green"] == {2, 3}
    """

    TIME = "time"
    LINE = "line"
    ZONES = "zones"
    CONNECTIONS = "connections"

    def __init__(self):
        self.graph_tube_map = dict()
        self.set_zones_per_station = dict()

    def import_tube_map_from_json(self, file_path: str) -> None:
        """
        Import the tube map information from a JSON file.
        During that import, the two following attributes should be updated:
        - graph_tube_map
        - set_zones_per_station

        :param file_path: relative or absolute path to the json file containing all the information about the
        tube map graph to import
        """
        with open(file_path,'r') as f:
            self.london_data = json.load(f)

        self.station1_dict = {}
        for connection in self.london_data["connections"]:
            for station in self.london_data["stations"]:
                if connection['station1'] == station['id']:
                    station1_name = {connection['station1']:station['name']}
                    self.station1_dict.update(station1_name)

        self.station2_dict = {}
        for connection in self.london_data["connections"]:
            for station in self.london_data["stations"]:
                if connection['station2'] == station['id']:
                    station2_name = {connection['station2']:station['name']}
                    self.station2_dict.update(station2_name)

        self.line_dict = {}
        for line in self.london_data["lines"]:
            for connection in self.london_data["connections"]:
                if connection['line'] == line['line']:
                    line_name = {connection['line']:line['name']}
                    self.line_dict.update(line_name)

        connection_number = []
        more_than_one_line = []
        for connection in self.london_data['connections']:
            if connection['station1'] in connection_number and not (connection['station1'],connection['station2']) in more_than_one_line:
                dict2 = [{'line':self.line_dict[connection['line']],'time':connection['time']}]
                self.graph_tube_map[self.station1_dict[connection['station1']]][self.station2_dict[connection['station2']]] = dict2
                more_than_one_line.append((connection['station1'],connection['station2']))
            elif connection['station1'] in connection_number and (connection['station1'],connection['station2']) in more_than_one_line:
                dict2 = {'line':self.line_dict[connection['line']],'time':connection['time']}
                self.graph_tube_map[self.station1_dict[connection['station1']]][self.station2_dict[connection['station2']]].append(dict2)
            else:
                london_data_dict = {self.station1_dict[connection['station1']]:{self.station2_dict[connection['station2']]:[{'line':self.line_dict[connection['line']],'time':connection['time']}]}}
                connection_number.append(connection['station1'])
                more_than_one_line.append((connection['station1'],connection['station2']))
                self.graph_tube_map.update(london_data_dict)

        for connection in self.london_data['connections']:
            if connection['station2'] in connection_number and not (connection['station2'],connection['station1']) in more_than_one_line:
                dict2 = [{'line':self.line_dict[connection['line']],'time':connection['time']}]
                self.graph_tube_map[self.station2_dict[connection['station2']]][self.station1_dict[connection['station1']]] = dict2
                more_than_one_line.append((connection['station2'],connection['station1']))
            elif connection['station2'] in connection_number and (connection['station2'],connection['station1']) in more_than_one_line:
                dict2 = {'line':self.line_dict[connection['line']],'time':connection['time']}
                self.graph_tube_map[self.station2_dict[connection['station2']]][self.station1_dict[connection['station1']]].append(dict2)
            else:
                london_data_dict = {self.station2_dict[connection['station2']]:{self.station1_dict[connection['station1']]:[{'line':self.line_dict[connection['line']],'time':connection['time']}]}}
                connection_number.append(connection['station2'])
                more_than_one_line.append((connection['station2'],connection['station1']))
                self.graph_tube_map.update(london_data_dict)

        for station in self.london_data['stations']:
            station['zone'] = float(station['zone'])
            if (station['zone']*2)//2 == -(-(station['zone']*2)//2):
                zones_dict = {station['name']:{int(station['zone'])}}
                self.set_zones_per_station.update(zones_dict)
            else:
                zones_dict = {station['name']:{(int(station['zone']*2)//2),int(-(-(station['zone']*2)//2))}}
                self.set_zones_per_station.update(zones_dict)


    def get_fastest_path_between(self, station_start: str, station_end: str) -> List[str]:
        """
        Implementation of Dijkstra algorithm to find the fastest path from station_start to station_end

        for instance: get_fastest_path_between('Stockwell', 'South Kensington') should return the list:
        ['Stockwell', 'Vauxhall', 'Pimlico', 'Victoria', 'Sloane Square', 'South Kensington']

        See here for more information: https://en.wikipedia.org/wiki/Dijkstra's_algorithm#Pseudocode

        :param station_start: name of the station at the beginning of the journey
        :param station_end: name of the station at the end of the journey
        :return: An ordered list representing the successive stations in the fastest path.
        :raise StationDoesNotExist if the station is not in the loaded tube map
        """
        self.import_tube_map_from_json("data_tubemap/london.json")

        shortest_path = {station_start:(None, 0)}
        current_station = station_start
        stations_visited = set()

        if not station_start in self.graph_tube_map.keys():
            raise StationDoesNotExist(station_start)
        elif not station_end in self.graph_tube_map.keys():
            raise StationDoesNotExist(station_end)

        while current_station != station_end:
            stations_visited.add(current_station)
            #print(stations_visited)
            next_stations = []
            for station in self.graph_tube_map[current_station]:
                next_stations.append(station)
            #print(next_stations)
            time_taken_to_next_station = shortest_path[current_station][1]
            #print(time_taken_to_next_station)

            for next_station in next_stations:
                time_taken = int(self.graph_tube_map[current_station][next_station][0]['time']) + time_taken_to_next_station # Have to make it so we take shortest time not just next time (sometimes more than one line)
                if next_station not in shortest_path:
                    shortest_path[next_station] = (current_station,time_taken)
                else:
                    current_shortest_time = shortest_path[next_station][1]
                    if int(current_shortest_time) > int(time_taken):
                        shortest_path[next_station] = (current_station,time_taken)

            next_destinations = {station: shortest_path[station] for station in shortest_path if station not in stations_visited}

            if not next_destinations:
                return "Route not Possible"
            current_station = min(next_destinations, key = lambda k:next_destinations[k][1])

        path = []
        while current_station is not None:
            path.append(current_station)
            next_station = shortest_path[current_station][0]
            current_station = next_station

        path = path[::-1]
        return path


    def get_set_lines_for_station(self, name_station: str) -> Set[str]:
        """
        :param name_station: name of a station in the tube map. (e.g. 'Hammersmith')
        :return: set of the names of the lines on which the station is found.
        :raise StationDoesNotExist if the station is not in the loaded tube map
        """

        if not name_station in self.graph_tube_map.keys():
            raise StationDoesNotExist(name_station)

        self.import_tube_map_from_json("data_tubemap/london.json")
        dict_name_station = self.graph_tube_map[name_station]
        self.set_stations = list()
        lines_on_station = dict()
        for station in dict_name_station:
            for i in range(len(dict_name_station[station])):
                lines_on_station.update(dict_name_station[station][i])
                self.set_stations.append(lines_on_station['line'])
        self.set_stations = set(self.set_stations)
        return self.set_stations

    def get_set_all_stations_on_line(self, line: str) -> Set[str]:
        """
        :param line: name of a metro line (e.g. 'Victoria Line')
        :return: the set of all the stations on that line.
        """
        self.import_tube_map_from_json("data_tubemap/london.json")
        self.dict_lines_to_station = dict()
        lines = set()
        for station in self.graph_tube_map.keys():
            for line1 in self.get_set_lines_for_station(station):
                if line1 in lines:
                    self.dict_lines_to_station[line1].add(station)
                else:
                    self.dict_lines_to_station.update({line1:{station}})
                    lines.add(line1)

        return self.dict_lines_to_station[line]

if __name__ == '__main__':
    tube_map = TubeMap()
    tube_map.import_tube_map_from_json("data_tubemap/london.json")
    print(tube_map.graph_tube_map['Pimlico'])
    print(tube_map.set_zones_per_station['Turnham Green'])
    print(tube_map.get_set_lines_for_station('Victoria'))
    print(tube_map.get_set_all_stations_on_line("Piccadilly Line"))
    print(tube_map.get_fastest_path_between("Elephant & Castle", "Wanstead"))
    # print(tube_map.set_zones_per_station["Turnham Green"])
