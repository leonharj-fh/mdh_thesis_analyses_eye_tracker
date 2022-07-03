from util import validations


class StudentSettings:
    def __init__(self, loaded_yaml: dict):
        self.__student_lab_recordings = StudentLabRecordings(loaded_yaml)
        self.merge_same_student_recordings = validations.checkNotNone(loaded_yaml["mergeSameStudentRecordings"])

    def is_lab_recording(self, year, student_uuid) -> bool:
        return self.__student_lab_recordings.is_lab_recording(year, student_uuid)

    def get_ids_for_year(self, year: int):
        return self.__student_lab_recordings.get_ids_for_year(year)


class StudentLabRecordings:
    def __init__(self, loaded_yaml: dict):
        recordings = validations.checkNotNone(loaded_yaml)["recordings"]

        students_lab_recording = {}
        for recording in recordings:
            year = validations.checkValueGreaterEqualsZero(recording["year"])
            ids = validations.checkAtLeastOneElement(recording["ids"])
            students_lab_recording[year] = ids

        self.__students_lab_recording = students_lab_recording

    def get_ids_for_year(self, year: int):
        return self.__students_lab_recording[year]

    def is_lab_recording(self, year: int, student_uuid: str) -> bool:

        return (
            year in self.__students_lab_recording
            and student_uuid in self.__students_lab_recording[year]
        )
