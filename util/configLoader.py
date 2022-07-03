import yaml
import os
from util import validations
from pathlib import Path
from model import Subjects
from model import AppSettings
from model import StudentSettings


configFolderNameDict = {2018: "configFolder_2018.yaml", 2019: "configFolder_2019.yaml"}
configFolderSchema = "configFolder.schema.yaml"

appFileName = "app.yaml"
appSchemaFileName = "app.schema.yaml"

studentSettingsFileName = "studentSettings.yaml"
studentSettingsSchemaFileName = "studentSettings.schema.yaml"

refractionErrorFiles = {
    2018: "2018_refraction_error.csv",
    2019: "2019_refraction_error.csv",
}


def getRefractionErrorFile(student_year: int) -> str:
    assert student_year in refractionErrorFiles
    file = os.path.join(__getDataFolder(), refractionErrorFiles.get(student_year))
    validations.checkFileExists(file)
    return file


def loadConfigFolderAndParseToSubjects(student_year: int) -> Subjects:
    assert student_year in configFolderNameDict
    return Subjects.from_dict_to_subjects(
        loadConfig(configFolderNameDict.get(student_year))
    )


def loadConfigAndParseStudentSettings() -> StudentSettings:
    return StudentSettings.StudentSettings(loadConfig(studentSettingsFileName))


def loadConfigAndParseToAppSettings() -> AppSettings:
    return AppSettings.from_dict(loadConfig(appFileName))


def loadConfig(file):
    return getFileAsYaml(os.path.join(__getConfigFolder(), file))


def getConfigSchema(file):
    return getFileAsYaml(os.path.join(__getSchemaFolder(), file))


def __getDataFolder():
    return os.path.join(Path(__file__).parent, "../data")


def __getConfigFolder():
    return os.path.join(Path(__file__).parent, "../config")


def __getSchemaFolder():
    return os.path.join(Path(__file__).parent, "../schemas")


def getFileAsYaml(fileName):
    with open(fileName, "r", encoding="UTF-8") as stream:
        try:
            return yaml.load(stream, Loader=yaml.SafeLoader)
        except yaml.YAMLError as exception:
            raise exception
