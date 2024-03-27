
import pandas as pd
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime


@dataclass
class Columns:
    Date: str = field(default='dateComponents', init=False)
    EnergyBurned: str = field(default='activeEnergyBurned', init=False)
    EnergyBurnedGoal: str = field(default='activeEnergyBurnedGoal', init=False)
    EnergyBurnedUnit: str = field(default='activeEnergyBurnedUnit', init=False)
    ExerciseTime: str = field(default='appleExerciseTime', init=False)
    ExerciseTimeGoal: str = field(default='activeExerciseTimeGoal', init=False)
    StandHours: str = field(default='activeStandHours', init=False)
    StandHoursGoal: str = field(default='appleStandHoursGoal', init=False)
    Weekday: str = field(default='Weekday', init=False)


class DataLoader:
    def __init__(self, filepath: str) -> None:
        self._filepath = filepath

    def load(self) -> None:
        self._df = pd.read_csv(self._filepath, parse_dates=[Columns.Date], encoding='utf-8')
        self._df[Columns.Weekday] = self._df[Columns.Date].dt.dayofweek

    def get(self, column: str | None, time_from: datetime | None, time_to: datetime | None) \
            -> pd.DataFrame:
        _time_from = pd.to_datetime(time_from)
        _time_to = pd.to_datetime(time_to)
        get_all = (column is None) or (column not in self._df.columns)
        if _time_from is None and _time_to is None:
            tmp_df = self._df
        elif _time_from is None:
            tmp_df = self._df[self._df[Columns.Date] < _time_to]
        elif _time_to is None:
            tmp_df = self._df[self._df[Columns.Date] >= _time_from]
        else:
            tmp_df = self._df[
                (self._df[Columns.Date] >= _time_from) * (self._df[Columns.Date] < _time_to)]
        # return dataframe
        if get_all:
            return tmp_df
        else:
            return tmp_df[column]
