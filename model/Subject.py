from util import validations


class Subject:
    def __init__(
            self,
            id: str,
            inputPath: str,
            outputPath: str,
            sensorDataFilename: str,
            eyetrackingDataFilename: str,
    ):  # noqa: E501

        """
        :param id: TODO
        :param inputPath: TODO
        :param outputPath: TODO
        :param sensorDataFilename: TODO
        :param eyetrackingDataFilename: TODO
        """

        self.id = validations.checkStringNotEmpty(id)
        self.inputPath = validations.checkStringNotEmpty(inputPath)
        self.outputPath = validations.checkStringNotEmpty(outputPath)
        self.sensorDataFilename = validations.checkStringNotEmpty(sensorDataFilename)
        self.eyetrackingDataFilename = validations.checkStringNotEmpty(eyetrackingDataFilename)


def parse(subject: dict) -> Subject:
    return Subject(
        subject["id"],
        subject["inputPath"],
        subject["outputPath"],
        subject["sensorDataFilename"],
        subject["eyetrackingDataFilename"],
    )
