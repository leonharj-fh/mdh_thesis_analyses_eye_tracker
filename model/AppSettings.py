from util import validations


class AppSettings:
    def __init__(
        self,
        student_years: [],
        commonOutputPath: str,
        createStatistic: bool = True,
        generateLidarStat: bool = True,
        generateVergenceStat: bool = True,
    ):
        """

        :param configLoader:
        :param enableAuthentication:
        """

        self.createStatistic = validations.checkNotNone(createStatistic)
        self.student_years = validations.checkAtLeastOneElement(student_years)
        self.commonOutputPath = validations.checkStringNotEmpty(commonOutputPath)
        self.generateLidarStatistics = validations.checkNotNone(generateLidarStat)
        self.generateVergenceStatStatistics = validations.checkNotNone(generateVergenceStat)


def from_dict(dikt: dict) -> ():
    """Returns the dict as a model

    :param dikt: A dict.
    :type: dict

    """
    validations.checkNotNone(dikt)
    createStatistics = dikt.get("createStatistic", True)
    student_years = validations.checkAtLeastOneElement(dikt["student_year"])
    generateLidarStatistics = dikt.get("generateLidarStat", True)
    generateVergenceStatistics = dikt.get("generateVergenceStat", True)
    commonOutputPath = dikt.get("commonOutputPath")
    return AppSettings(
        student_years,
        commonOutputPath,
        createStatistics,
        generateLidarStat=generateLidarStatistics,
        generateVergenceStat=generateVergenceStatistics,
    )
