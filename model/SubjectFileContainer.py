from util import validations


class SubjectFileContainer:
    def __init__(
            self,
            id: str,
            pathToSensorData: str,
            pathToEyeRecordsData: str,
            subjectFolder: str,
            pathToTempfile: str,
            pathToEyeDataWithReference: str,
            pathToSensorDataRbg: str,
            pathToSensorDataLight: str,
            pathToSensorDataDistance: str,
            pathToOutputFolder: str,
    ):  # noqa: E501

        """
        :param id:
        :param pathToSensorData:
        :param pathToEyeRecordsData:
        :param subjectFolder:
        :param pathToTempfile:
        :param pathToEyeDataWithReference:
        :param pathToSensorDataRbg:
        :param pathToSensorDataLight:
        :param pathToSensorDataDistance:
        """

        self.id = validations.checkStringNotEmpty(id)
        self.pathToInputSensorData = validations.checkStringNotEmpty(pathToSensorData)
        self.pathToInputEyeRecordsData = validations.checkStringNotEmpty(pathToEyeRecordsData)
        self.subjectFolder = validations.checkStringNotEmpty(subjectFolder)
        self.pathToTempfile = validations.checkStringNotEmpty(pathToTempfile)
        self.pathToEyeDataWithReference = validations.checkStringNotEmpty(
            pathToEyeDataWithReference
        )
        self.pathToSensorDataRbg = validations.checkStringNotEmpty(pathToSensorDataRbg)
        self.pathToSensorDataLight = validations.checkStringNotEmpty(pathToSensorDataLight)
        self.pathToSensorDataDistance = validations.checkStringNotEmpty(
            pathToSensorDataDistance
        )
        self.pathToOutputFolder = validations.checkStringNotEmpty(
            pathToOutputFolder
        )
