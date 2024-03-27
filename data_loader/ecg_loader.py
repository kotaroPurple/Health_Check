
import numpy as np
import pandas as pd
from dataclasses import dataclass
from dataclasses import field


@dataclass
class EcgColumns:
    Frame: str = field(default='frame', init=False)
    Time: str = field(default='time', init=False)
    Value: str = field(default='value', init=False)


class EcgLoader:
    def __init__(self, filepath: str) -> None:
        self._filepath = filepath

    def load(self) -> None:
        # load the file
        data = np.loadtxt(self._filepath, delimiter=',', comments='#')
        self._hz = data[0]
        data = data[1:]
        # numpy to dataframe
        frames = np.arange(len(data))
        times = frames.astype(np.float64) / self._hz
        self._df = pd.DataFrame({
            EcgColumns.Frame: frames,
            EcgColumns.Time: times,
            EcgColumns.Value: data})

    def get_sampling_rate(self) -> float:
        return self._hz

    def get(self) -> pd.DataFrame:
        return self._df
