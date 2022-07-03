import numpy as np
import pandas as pd
from util import validations, csv_columns as columns


class Container:
    def __init__(
            self,
            subject_id: str,
            unmodified_data_container: pd.DataFrame,
            filtered_data_container: pd.DataFrame,
            data_column_name: str,
    ):
        self.id = validations.checkStringNotEmpty(subject_id)
        self.__data_container = validations.check_is_dataframe(unmodified_data_container)
        self.__filtered_data_container = validations.check_is_dataframe(filtered_data_container)
        self.data_column_name = validations.checkStringNotEmpty(data_column_name)

    # getter method
    def get_data_container(self):
        return self.__data_container.copy()

    # getter method
    def get_filtered_data_container(self):
        return self.__filtered_data_container.copy()


class LidarContainer(Container):
    def __init__(
            self,
            subject_id: str,
            unmodified_data_container: pd.DataFrame,
            filtered_data_container: pd.DataFrame,
    ):
        super().__init__(
            subject_id,
            unmodified_data_container,
            filtered_data_container,
            columns.SENSOR_COLUMN_DISTANCE,
        )


class VergenceContainer(Container):
    def __init__(
            self,
            subject_id: str,
            unmodified_data_container: pd.DataFrame,
            filtered_data_container: pd.DataFrame,
    ):
        super().__init__(
            subject_id,
            unmodified_data_container,
            filtered_data_container,
            columns.EYE_TRACKING_COLUMN_VERGENCE_ANGLE,
        )


class FigureResult:
    def __init__(
            self,
            subject_id: str,
            function,
            figure_file_name: str = None,
            figure_path: str = None,
    ):
        self.id = validations.checkStringNotEmpty(subject_id)
        self.function = function
        self.figure_file_name = validations.none_or_check(
            figure_file_name, validations.checkStringNotEmpty
        )
        self.figure_path = figure_path


class ContainerResult:
    def __init__(
            self,
            subject_id: str,
            data: pd.DataFrame = None,
            figure_function=None,
            figure_file_name: str = None,
            figure_path: str = None,
            acf_differences: np.ndarray = None,
    ):
        self.id = validations.checkStringNotEmpty(subject_id)
        self.data = validations.none_or_check(data, validations.check_is_dataframe)
        self.figure_function = figure_function
        self.figure_file_name = validations.none_or_check(
            figure_file_name, validations.checkStringNotEmpty
        )
        self.acf_differences = validations.none_or_check(
            acf_differences, validations.checkContainsNoNotNanNumpyValues
        )
        self.figure_path = figure_path


class LidarContainerResult(ContainerResult):
    def __init__(
            self,
            subject_id: str,
            data: pd.DataFrame,
            figure_function,
            figure_file_name: str = None,
            figure_path: str = None,
            acf_differences: np.ndarray = None,
    ):
        super().__init__(
            subject_id,
            validations.checkNotNone(data),
            figure_function,
            figure_file_name,
            figure_path,
            acf_differences,
        )


class VergenceContainerResult(ContainerResult):
    def __init__(
            self,
            subject_id: str,
            data: pd.DataFrame,
            figure_function=None,
            figure_file_name: str = None,
            acf_differences: np.ndarray = None,
    ):
        super().__init__(
            subject_id,
            validations.checkNotNone(data),
            figure_function,
            figure_file_name,
            None,
            acf_differences,
        )


class AcfSettings:
    def __init__(
            self,
            strip_invalid_data: bool,
            lags: int,
            delta_time: float,
            test: bool = False
    ):
        self.strip_invalid_data = validations.checkNotNone(strip_invalid_data)
        assert lags >= 0
        self.lags = lags
        assert delta_time >= 0.0
        self.delta_time = delta_time
        self.test = test
