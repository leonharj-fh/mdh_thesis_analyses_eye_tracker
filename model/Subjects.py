from util import validations
from model import Subject


class Subjects:
    def __init__(self, subjects: tuple):
        """
        :param subjects: TODO
        """

        assert all(type(e) is Subject.Subject for e in subjects)
        self.subjects = validations.checkAtLeastOneElement(subjects)


def from_dict(dikt: dict) -> ():
    """Returns the dict as a model

    :param dikt: A dict.
    :type: dict

    """

    subjects = validations.checkNotNone(dikt["subjects"])
    return tuple(map(lambda x: Subject.parse(x), subjects))


def from_dict_to_subjects(dikt: dict) -> Subjects:
    """Returns the dict as a model

    :param dikt: A dict.
    :type: dict

    """

    subjectIds = validations.checkAtLeastOneElement(dikt["subjectIds"])

    inputPath = validations.checkStringNotEmpty(dikt["inputPath"])
    outputPath = validations.checkStringNotEmpty(dikt["outputPath"])
    sensorDataFilename = validations.checkStringNotEmpty(dikt["sensorDataFilename"])
    eyetrackingDataFilename = validations.checkStringNotEmpty(dikt["eyetrackingDataFilename"])
    subjectIds = validations.checkAtLeastOneElement(subjectIds)

    subjects = tuple(
        map(lambda id: Subject.Subject(id, inputPath, outputPath, sensorDataFilename, eyetrackingDataFilename),
            subjectIds))

    return Subjects(subjects)