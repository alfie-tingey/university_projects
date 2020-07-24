import datetime
from typing import Dict, List, Union

from tubemap.journey import Journey
from tubemap.tubemap import TubeMap


class User:
    MAX_PRICE_PER_DAY_WITH_PEAK = 8.80
    MAX_PRICE_PER_DAY_WITHOUT_PEAK = 7.00

    def __init__(self, name: str, tube_map: TubeMap):
        """

        self.journeys_per_date has the following form:
        {
            date_1: [
                journey_1_a,
                journey_1_b,
            ],
            date_2: [
                journey_2_a,
            ],
        }
        where date_1 and date_2 are of type datetime.date,
        and journey_1_a, journey_1_b and journey_2_a are of type Journey.

        :param name: full name of the user
        :param tube_map: entire tube map that is used
        """

        self.name = name
        self.tube_map = tube_map

        self.journeys_per_date = dict()  # type: Dict[datetime.date, List[Journey]]

    def register_journey(self,
                         time_start: datetime.time,
                         time_end: datetime.time,
                         date: datetime.date,
                         list_successive_stations: List[str],
                         ) -> Union[Journey, None]:
        """
        Register a journey in self.journeys_per_date
        :param time_start: the time at which the journey started
        :param time_end: the time at which the journey ended
        :param date: the date when that journey was performed
        :param list_successive_stations: list of the successive stations
        :return: - if no JourneyNotValid exception is raised when creating the Journey, then the newly registered
        journey is added in self.journeys_per_date.
                 - if a JourneyNotValid is raised during the creation of the Journey, then return None.
        """
        from tubemap.journey import JourneyNotValid
        try:
            journeys = Journey(time_start,
                          time_end,
                          date,
                          list_successive_stations,
                          self.tube_map)
        except JourneyNotValid:
            return None

        if date in self.journeys_per_date:
            self.journeys_per_date[date].append(journeys)
        else:
            self.date_dict = {date:[journeys]}
            self.journeys_per_date.update(self.date_dict)

    def compute_price_per_day(self, date: datetime.date) -> float:
        """
        :param date: day at which we want to calculate the price
        :return: Total amount of money spent in the tube by the user at that day,
        if the user did not use the tube on that date, then the function should return 0.
        """
        if not date in self.journeys_per_date.keys():
            return 0

        price = sum([float(journey.compute_price()) for journey in self.journeys_per_date[date]])
        on_peak = any([journey.is_on_peak() for journey in self.journeys_per_date[date]])
        if price > 8.80 and on_peak:
            price = 8.80
        elif price > 7.00 and not on_peak:
            price = 7.00
        return float('{0:.2f}'.format(price))


if __name__ == '__main__':
    tube_map = TubeMap()
    tube_map.import_tube_map_from_json("data_tubemap/london.json")

    user = User("Bob", tube_map)

    # A journey on the 30/10/2019 off peak
    user.register_journey(time_start=datetime.time(hour=12, minute=15),
                          time_end=datetime.time(hour=12, minute=30),
                          date=datetime.date(year=2019, month=10, day=30),
                          list_successive_stations=['Stockwell', 'Vauxhall', 'Pimlico',
                                                    'Victoria', 'Sloane Square', 'South Kensington'], )

    #Another journey on the 30/10/2019 on peak
    user.register_journey(time_start=datetime.time(hour=18, minute=15),
                          time_end=datetime.time(hour=18, minute=30),
                          date=datetime.date(year=2019, month=10, day=30),
                          list_successive_stations=['Stockwell', 'Vauxhall', 'Pimlico',
                                                    'Victoria', 'Sloane Square', 'South Kensington'], )

    # Trying to add an Invalid journey (the function should return None in that case)
    user.register_journey(time_start=datetime.time(hour=18, minute=15),
                          time_end=datetime.time(hour=18, minute=30),
                          date=datetime.date(year=2019, month=10, day=30),
                          list_successive_stations=['Stockwell', 'Vauxhall', 'Pimlico',
                                                    'Victoria', 'South Kensington'], )

    print(user.compute_price_per_day(date=datetime.date(year=2019, month=10, day=30)))

    # Adding more journeys to reach the maximum price per day
    user.register_journey(time_start=datetime.time(hour=8, minute=15),
                          time_end=datetime.time(hour=8, minute=30),
                          date=datetime.date(year=2019, month=10, day=30),
                          list_successive_stations=['Stockwell', 'Vauxhall', 'Pimlico',
                                                    'Victoria', 'Sloane Square', 'South Kensington'], )

    user.register_journey(time_start=datetime.time(hour=10, minute=15),
                          time_end=datetime.time(hour=10, minute=30),
                          date=datetime.date(year=2019, month=10, day=30),
                          list_successive_stations=['Stockwell', 'Vauxhall', 'Pimlico',
                                                    'Victoria', 'Sloane Square', 'South Kensington'], )

    print(user.compute_price_per_day(date=datetime.date(year=2019, month=10, day=30)))
