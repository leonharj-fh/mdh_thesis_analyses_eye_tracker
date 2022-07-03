import pandas as pd
import os
import numpy as np
from util import constants as const
from util import validations
import model.DataContainer as dc

from timeit import default_timer as timer
from datetime import timedelta

def calculate_inverse_values(values: np.ndarray) -> np.ndarray:
    return np.where(values != np.NaN, np.reciprocal(np.true_divide(values, 1000)), values)

def get_values_for_acf(
    container: dc.Container, settings: dc.AcfSettings
) -> np.ndarray:
    filtered_data_container = container.get_filtered_data_container()

    series = (
        filtered_data_container[container.data_column_name]
        .replace([np.inf], np.nan)
        .dropna()
        if settings.strip_invalid_data
        else filtered_data_container[container.data_column_name]
    )

    values_for_acf = calculate_inverse_values(series.to_numpy())

    # this method replaces all NaN values with inf. Important in the next step
    # if Nan values are not stripped before
    values_for_acf = np.nan_to_num(values_for_acf, nan=np.inf)

    return validations.checkContainsNoNotNanNumpyValues(values_for_acf)




def __write_figure(container: dc.ContainerResult, output_path: str):
    folder = (
        os.path.join(output_path, const.IMAGES_FOLDER, container.figure_path)
        if container.figure_path is not None
        else os.path.join(output_path, const.IMAGES_FOLDER)
    )
    validations.createDirIfNotExists(folder)
    file_name = os.path.join(folder, container.id + "-" + container.figure_file_name)
    container.figure_function(file_name)


def is_instance_LidarContainerResult(object):
    return isinstance(object, dc.LidarContainerResult)


def is_instance_ContainerResult(object):
    return isinstance(object, dc.ContainerResult)


def __get_data(container) -> pd.DataFrame:
    if is_instance_ContainerResult(container):
        return container.data
    elif is_instance_LidarContainerResult(container):
        return container.data
    return container


def perform_on_students_only_fig(
    output_path: str,
    subject_containers: [dc.Container],
    function,
) -> None:
    validations.checkStringNotEmpty(output_path)
    validations.checkAtLeastOneElement(subject_containers)

    start = timer()
    results = np.array([function(subjects) for subjects in subject_containers])
    print("Finished figure function execution:", timedelta(seconds=timer() - start))

    start = timer()
    for result in results:
        if result is None:
            continue
        folder = (
            os.path.join(output_path, const.IMAGES_FOLDER, result.figure_path)
            if result.figure_path is not None
            else os.path.join(output_path, const.IMAGES_FOLDER)
        )
        validations.createDirIfNotExists(folder)
        file_name = os.path.join(folder, result.id + "-" + result.figure_file_name)
        result.function(file_name)

    print("Finished creating figures.", timedelta(seconds=timer() - start))


# TODO merge with __perform_on_students()
def perform_on_students_basic(
    subject_containers: [dc.Container],
    function,
) -> pd.DataFrame:
    validations.checkAtLeastOneElement(subject_containers)
    validations.checkNotNone(function)

    start = timer()
    results = np.array([function(subjects) for subjects in subject_containers])
    print("Finished function execution:", timedelta(seconds=timer() - start))

    mergedDf = None
    for result in results:
        if mergedDf is None:
            mergedDf = __get_data(result)
        else:
            mergedDf = pd.concat([mergedDf, __get_data(result)], axis=1)

    return mergedDf.transpose().convert_dtypes()


def perform_on_students(
    outputPath: str,
    file_name: str,
    subject_containers: [dc.Container],
    function,
    sortIndex=False,
) -> str:
    validations.checkStringNotEmpty(outputPath)
    validations.checkStringNotEmpty(file_name)
    validations.checkAtLeastOneElement(subject_containers)
    validations.checkNotNone(function)

    start = timer()
    results = np.array([function(subjects) for subjects in subject_containers])
    print("Finished function execution:", timedelta(seconds=timer() - start))

    mergedDf = None
    for result in results:
        if mergedDf is None:
            mergedDf = __get_data(result)
        else:
            mergedDf = pd.concat([mergedDf, __get_data(result)], axis=1)
        if is_instance_ContainerResult(result) and result.figure_function is not None:
            __write_figure(result, outputPath)

    statisticFolder = os.path.join(outputPath, const.SUB_FOLDER_STATISTICS)
    validations.createDirIfNotExists(statisticFolder)

    outputFile = os.path.join(statisticFolder, file_name)
    with open(outputFile, "w") as out:
        if sortIndex:
            mergedDf.sort_index(inplace=True)
        mergedDf.transpose().to_csv(out)
        print("Finished file:", file_name)

    return outputFile