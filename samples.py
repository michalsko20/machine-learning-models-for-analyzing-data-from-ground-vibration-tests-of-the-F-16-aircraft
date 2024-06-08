import os
import csv

from enum import Enum
from dataclasses import dataclass
from numpy.typing import NDArray
import numpy as np
from sklearn.model_selection import train_test_split
from numpy.typing import NDArray


FOLDER_PATH = "BenchmarkData"


class StimulationType(Enum):
    SINE_SWEEP = "sine_sweep"
    MULTISINE_FULL_FREQUENCY_GRID = "multisine_full_frequency_grid"
    MULTISINE_RANDOM_FREQUENCY_GRID = "multisine_random_frequency_grif"


@dataclass
class DataSample:
    force: float
    voltage: float
    acceleration_1: float
    acceleration_2: float
    acceleration_3: float
    stimulation_type: StimulationType

    @property
    def X(self) -> NDArray:
        return np.array(
            [
                self.force,
                self.voltage,
            ]
        )

    @property
    def y(self) -> NDArray:
        return np.array([self.acceleration_1, self.acceleration_2, self.acceleration_3])


class LineToDataSampleError(Exception):
    pass


def determine_stimulation_type(file_name: str) -> StimulationType:
    if ("SineSw") in file_name:
        return StimulationType.SINE_SWEEP
    if ("FullMSine") in file_name:
        return StimulationType.MULTISINE_FULL_FREQUENCY_GRID
    if ("SpecialOddMSine") in file_name:
        return StimulationType.MULTISINE_RANDOM_FREQUENCY_GRID

    raise Exception(f"Failed to determine simulation type for {file_name}")


def _csv_line_to_data_sample(
    line: list[str], stimulation_type: StimulationType
) -> DataSample:
    if "Force" in line:
        raise LineToDataSampleError("Detected header row, skipping")

    if len(line) == 0:
        raise LineToDataSampleError("Invalid line")

    data_as_floats = [float(measurement) for measurement in line[:5]]

    return DataSample(
        *data_as_floats,
        stimulation_type=stimulation_type,
    )


def _csv_file_to_data_samples(
    stimulation_type: StimulationType = None,
) -> list[DataSample]:
    data_samples = []

    for data_set_filename in os.listdir(FOLDER_PATH):
        filepath = os.path.join(FOLDER_PATH, data_set_filename)

        if "csv" not in data_set_filename:
            continue

        file_stimulation_type = determine_stimulation_type(data_set_filename)

        if stimulation_type == StimulationType.MULTISINE_RANDOM_FREQUENCY_GRID:
            continue

        if stimulation_type != None and stimulation_type != file_stimulation_type:
            continue

        with open(filepath, newline="") as data_set_file:
            reader = csv.reader(data_set_file)

            for row in reader:
                try:
                    data_sample = _csv_line_to_data_sample(row, file_stimulation_type)
                    data_samples.append(data_sample)
                except LineToDataSampleError:
                    continue

    return data_samples


def get_data_samples(stimulation_type: StimulationType = None) -> list[DataSample]:
    if stimulation_type == StimulationType.MULTISINE_RANDOM_FREQUENCY_GRID:
        raise Exception("Not supported stimulation type")

    return _csv_file_to_data_samples(stimulation_type)
