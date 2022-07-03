import prepareDataUsingChunks
import model.Subjects as Subjects
import model.SubjectFileContainer as SubjectFileContainer
import os.path as path
from util import configLoader
from statistic import statistics
from util import validations, constants as const

# import subprocess
# print(subprocess.check_output(["git", "describe", "--always"]).strip().decode())


def prepare_and_load_subjects(_subjects: Subjects) -> [SubjectFileContainer]:
    return [
        prepareDataUsingChunks.prepareDataForEvaluation(subject)
        for subject in _subjects.subjects
    ]


if __name__ == "__main__":
    app_settings = configLoader.loadConfigAndParseToAppSettings()
    student_settings = configLoader.loadConfigAndParseStudentSettings()

    validations.createDirIfNotExists(
        path.join(app_settings.commonOutputPath, const.SUB_FOLDER_THESIS)
    )

    yearToStudentContainer = {}
    for year in app_settings.student_years:
        subjects = configLoader.loadConfigFolderAndParseToSubjects(year)
        containers = prepare_and_load_subjects(validations.checkNotNone(subjects))
        print("All student data prepared for year", year)

        yearToStudentContainer[year] = containers

        if app_settings.createStatistic:
            statistics.generate_data_and_statistics(containers, app_settings)

    if len(yearToStudentContainer.keys()) >= 2:
        statistics.test_for_autocorrelation(yearToStudentContainer)
        statistics.generate_statistics_comparison(
            yearToStudentContainer, app_settings, student_settings
        )
        statistics.generate_statistics_comparison_statistical_tests(
            yearToStudentContainer, app_settings, student_settings
        )
