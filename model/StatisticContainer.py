from util import validations


class StatisticContainer:
    def __init__(
        self, acf_statistic_file: str = None, acf_inverse_statistic_file: str = None
    ):
        self.acf_statistic_file = validations.none_or_check(
            acf_statistic_file, validations.checkFileExists
        )
        self.acf_inverse_statistic_file = validations.none_or_check(
            acf_inverse_statistic_file, validations.checkFileExists
        )


class LidarStatisticContainer(StatisticContainer):
    def __init__(
        self, acf_statistic_file: str = None, acf_inverse_statistic_file: str = None
    ):
        super().__init__(acf_statistic_file, acf_inverse_statistic_file)
