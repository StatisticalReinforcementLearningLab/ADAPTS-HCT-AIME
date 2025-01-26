class TrialTime:
    # Class variables
    WeekLength = 7
    DayLength = 2

    def __init__(self, week: int = 0, day: int = 0, timeOfDay: int = 0):
        self.week: int = week
        self.day: int = day
        self.timeOfDay: int = timeOfDay  # 0 = morning, 1 = night
        self.time: int = self.day * 2 + self.week * 14 + self.timeOfDay

    def __str__(self):
        if self.timeOfDay == 0:
            timeOfDayStr = "Morning"
        else:
            timeOfDayStr = "Night"
        return f"Week: {self.week}, Day: {self.day}, Time of Day: {timeOfDayStr}"

    def __sub__(self, other):
        if isinstance(other, TrialTime):
            return TrialTime(
                week=self.week - other.week,
                day=self.day - other.day,
                timeOfDay=self.timeOfDay - other.timeOfDay,
            )
        else:
            raise TypeError("Can only subtract TrialTime from another TrialTime")

    def getNextTime(self):
        self.timeOfDay += 1
        if self.timeOfDay >= self.DayLength:
            self.timeOfDay = 0
            self.day += 1
            if self.day >= self.WeekLength:
                self.day = 0
                self.week += 1
    def isNewDay(self):
        return self.timeOfDay == 0

    def isNewWeek(self):
        return self.day == 0 and self.timeOfDay == 0

    def getAbsoluteTime(self):
        return self.week * self.WeekLength * self.DayLength + self.day * self.DayLength + self.timeOfDay


if __name__ == "__main__":
    trial_time = TrialTime()
    for i in range(20):
        print(trial_time)
        trial_time.getNextTime()
