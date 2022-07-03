import unittest
import numpy as np

import prepareDataUsingChunks
import pandas as pd
from util import csv_columns as columns, configLoader as loader, constants
from statistic import statistic_methods as sm


class Tests(unittest.TestCase):
    def test_numpy_zeros(self):
        very_small_value = constants.ACF_REPLACE_ZERO_VALUES
        array = np.array([0, very_small_value, 0.1])
        updated = np.where(array == 0.0, 123, array)

        assert updated[0] == 123
        assert updated[1] == very_small_value
        assert updated[2] == 0.1

    def test_unique_indexes(self):
        """
        Adapt this test if necessary. If there are too many duplicate indexes it may indicate that the recording time
        or index creation is vaulty.

        :return:
        """
        app_settings = loader.loadConfigAndParseToAppSettings()

        test_fail = False

        for year in app_settings.student_years:
            subjects = loader.loadConfigFolderAndParseToSubjects(year)

            containers = [
                prepareDataUsingChunks.prepareDataForEvaluation(subject)
                for subject in subjects.subjects
            ]

            for container in containers:
                distance_sensor_df = pd.read_csv(
                    container.pathToSensorDataDistance,
                    header=0,
                    usecols=[columns.DISTANCE_SENSOR_COLUMN_INDEX, columns.TIMESTAMP_COLUMN],
                    index_col=columns.DISTANCE_SENSOR_COLUMN_INDEX,
                )

                if not distance_sensor_df.index.is_unique:
                    series = pd.value_counts(distance_sensor_df.index.values.tolist())
                    seriesWithDuplicateIndixes = series.where(series > 1).dropna()

                    if seriesWithDuplicateIndixes.count() > 5:
                        print(
                            seriesWithDuplicateIndixes.count(),
                            container.id,
                            year,
                        )

                    if seriesWithDuplicateIndixes.count() > 5:
                        print("ERROR list", seriesWithDuplicateIndixes.index.tolist())
                        test_fail = True

        assert not test_fail

    def test_shift_array(self):
        values = np.array([127, np.inf, 129, 130, np.inf, 131], dtype=np.float64)

        assert np.array_equal(values, sm.multiply_lag_shift(values, 0))

        update_values = sm.multiply_lag_shift(values, 1)
        # [127, np.inf, 129, 130, np.inf, 131]
        # [np.inf, 1, 1, np.inf, 131, 1]
        assert np.array_equal(update_values, np.array([np.inf, np.inf, 129, np.inf, np.inf, 131], dtype=np.float64))

        update_values = sm.multiply_lag_shift(values, 2)
        assert np.array_equal(update_values, np.array([127, np.inf, np.inf, 130, np.inf, 131], dtype=np.float64))

if __name__ == "__main__":
    unittest.main()
