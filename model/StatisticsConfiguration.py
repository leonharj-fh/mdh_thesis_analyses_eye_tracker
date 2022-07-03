import pandas as pd
from util import validations


class Settings:
    def __init__(self, data_container: pd.DataFrame, strip_invalid_data: bool, inverse_distance: bool,
                 plot_acf_data: bool, plot_folder_name: str):
        """

        :param strip_invalid_data:
        :param inverse_distance:
        :param plot_acf_data:
        """
        self.data_container = validations.checkNotNone(data_container)
        self.strip_invalid_data = strip_invalid_data
        self.inverse_distance = inverse_distance
        self.plot_acf_data = plot_acf_data
        self.plot_folder_name = plot_folder_name
