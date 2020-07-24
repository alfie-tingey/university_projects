import datetime
from typing import List

from tubemap.tubemap import TubeMap


class JourneyNotValid(Exception):
    pass


class Journey(object):
    PEAK_TIMES = [
        (datetime.time(6, 30), datetime.time(9, 30)),
        (datetime.time(17), datetime.time(20)),
    ]

    MAXIMUM_AMOUNT_STATIONS_SHORT_JOURNEYS = 10

    def __init__(self,
                 time_start: datetime.time,
                 time_end: datetime.time,
                 date: datetime.date,
                 list_successive_stations: List[str],
                 tube_map: TubeMap,
                 ):
        """
        :param time_start: the time at which the journey started
        :param time_end: the time at which the journey ended
        :param date: the date when that journey was performed
        :param list_successive_stations: list of the successive stations
        """

        if time_end < time_start:
            raise JourneyNotValid("The time the journey ended cannot be higher than the time at which it started.")

        self.time_start = time_start
        self.time_end = time_end

        self.date = date

        if not self.is_list_successive_stations_valid(list_successive_stations, tube_map):
            raise JourneyNotValid(f"The provided list of successive stations {list_successive_stations}"
                                  f" is not valid according to the provided tube map.")

        self.list_successive_stations = list_successive_stations

    def compute_price(self) -> float:
        """
        :return: the price of the journey depending on different parameters:
                    - if the journey has been done on peak or not
                    - if the journey is a long one (with strictly more than 10 stations) or not
        """
        if self.is_on_peak() and self.is_long_journey():
            price = 3.80
            return float('{0:.2f}'.format(price))
        elif self.is_on_peak() and not self.is_long_journey():
            price = 2.90
            return float(2.90)
        elif not self.is_on_peak() and self.is_long_journey():
            price = 2.60
            return float('{0:.2f}'.format(price))
        elif not self.is_on_peak() and not self.is_long_journey():
            price = 1.90
            return float('{0:.2f}'.format(price))

    def is_on_peak(self) -> bool:
        """
        :return: True if and only if the journey starts or ends during peak times.
        """
        on_peak = False
        if self.PEAK_TIMES[0][0] <= self.time_start <= self.PEAK_TIMES[0][1] \
        or self.PEAK_TIMES[0][0] <= self.time_end <= self.PEAK_TIMES[0][1]:
            on_peak = True
        elif self.PEAK_TIMES[1][0] <= self.time_start <= self.PEAK_TIMES[0][1] \
        or self.PEAK_TIMES[1][0] <= self.time_end <= self.PEAK_TIMES[1][1]:
            on_peak = True

        return on_peak


    def is_long_journey(self) -> bool:
        """
        :return: True if and only if the journey is not a short journey.
        """
        return len(self.list_successive_stations) > 10

    @staticmethod
    def is_list_successive_stations_valid(successive_stations_list: List[str],
                                          tube_map: TubeMap,
                                          ) -> bool:
        """
        :param successive_stations_list: list of successive stations to describe a journey
        :param tube_map: the tube map in which the stations are.
        :return: True if and only if:
            - All the stations in the successive_stations_list are indeed in the tube map provided
            - All the pairs of successive stations in successive_stations_list are indeed connected in the graph.
        """
        tube_map = TubeMap()
        tube_map.import_tube_map_from_json("data_tubemap/london.json")
        tube_map.graph_tube_map
        valid = False

        for station in successive_stations_list:
            if station in tube_map.graph_tube_map.keys():
                valid = True
            else:
                return False

        for i in range(len(successive_stations_list)-1):
            if successive_stations_list[i+1] in tube_map.graph_tube_map[successive_stations_list[i]].keys():
                valid = True
            else:
                return False

        return valid

    def __eq__(self, other_journey: 'Journey'):
        return self.date == other_journey.date \
               and self.time_start == other_journey.time_start \
               and self.time_end == other_journey.time_end \
               and self.list_successive_stations == other_journey.list_successive_stations


if __name__ == '__main__':
    tube_map = TubeMap()
    tube_map.import_tube_map_from_json("data_tubemap/london.json")
    journey = Journey(time_start=datetime.time(hour=12, minute=15),
                      time_end=datetime.time(hour=12, minute=30),
                      date=datetime.date(year=2019, month=10, day=30),
                      list_successive_stations=['Stockwell', 'Vauxhall', 'Pimlico',
                                                'Victoria','Sloane Square', 'South Kensington'],
                      tube_map=tube_map)

    print(journey.compute_price())
