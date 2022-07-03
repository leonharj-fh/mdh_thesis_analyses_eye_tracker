import unittest
from util import configLoader as loader
import os
import re

from jsonschema import validate


class Tests(unittest.TestCase):
    def test_validate_config_files(self):
        yaml = loader.loadConfig(loader.appFileName)
        schema = loader.getConfigSchema(loader.appSchemaFileName)
        validate(yaml, schema)

        yaml = loader.loadConfig(loader.studentSettingsFileName)
        schema = loader.getConfigSchema(loader.studentSettingsSchemaFileName)
        validate(yaml, schema)

        for values in loader.configFolderNameDict.values():
            yaml = loader.loadConfig(values)
            schema = loader.getConfigSchema(loader.configFolderSchema)
            validate(yaml, schema)

    def test_validate_folder_exists(self):
        for year in loader.configFolderNameDict.keys():
            s = loader.loadConfigFolderAndParseToSubjects(year)

            for subject in s.subjects:
                assert os.path.exists(subject.inputPath)
                assert os.path.exists(
                    os.path.join(
                        subject.inputPath, subject.id, subject.sensorDataFilename
                    )
                )
                assert os.path.exists(
                    os.path.join(
                        subject.inputPath, subject.id, subject.eyetrackingDataFilename
                    )
                )

    def test_lab_student(self):
        lab_subjects_per_year = loader.loadConfigAndParseStudentSettings()

        assert isinstance(lab_subjects_per_year.merge_same_student_recordings, bool)

        assert lab_subjects_per_year.is_lab_recording(
            2019, "e6b99af1-630f-4eed-8177-dbfe7f3c0ede"
        )
        assert not lab_subjects_per_year.is_lab_recording(
            2019, "e6b99af1-630f-4eed-8177-dbfe7f3c0"
        )

        assert lab_subjects_per_year.is_lab_recording(
            2018, "d6a4f8cc-4308-41fb-ac14-4a7c7ee0ee03"
        )

    def test_lab_student_all(self):
        student_settings = loader.loadConfigAndParseStudentSettings()

        for year in loader.configFolderNameDict.keys():
            s = loader.loadConfigFolderAndParseToSubjects(year)

            ids = list(
                map(
                    lambda subject: re.search("\\[(.*)\\]", subject.id).group(1),
                    s.subjects,
                ),
            )

            lab_students = student_settings.get_ids_for_year(year)
            for id in lab_students:
                assert id in ids


if __name__ == "__main__":
    unittest.main()
