from datetime import timedelta

class Movie:
    def __init__(self, dates):
        self.__date = dates
        
    def schedule(self):
        for d in self.__date:
            current_date = d[0]
            delta = timedelta(days=1)
            while (current_date <= d[1]):
                yield current_date
                current_date += delta
