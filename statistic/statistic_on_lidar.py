import pandas as pd
import os
import numpy as np
import plot
import model.DataContainer as dc

from util import constants as const, csv_columns as columns, filters, validations
from statistic import statistic_methods as sm, statistic_common as sc

from timeit import default_timer as timer
from datetime import timedelta

from model import SubjectFileContainer

# np.seterr('raise') use for debugging.
np.seterr("raise")
pd.options.display.float_format = "{:.5f}".format

DECIMAL_POINTS_ACF = 5

STRIP_INVALID_DATA_TRUE = True
STRIP_INVALID_DATA_FALSE = False

INVERSE_DISTANCE_TRUE = True


def get_acf_inverse_statistic_pairwise_basic_lidar(
    container: dc.LidarContainer,
) -> dc.LidarContainerResult:
    ACF_LIDAR_LAGS = 2
    return __get_acf_lidar_statistics_pairwise(
        container,
        dc.AcfSettings(
            STRIP_INVALID_DATA_FALSE,
            ACF_LIDAR_LAGS,
            const.LIDAR_DELTA_TIME,
        ),
    )


def get_acf_inverse_statistics_pairwise_lidar(
    container: dc.LidarContainer,
) -> dc.LidarContainerResult:
    return __get_acf_lidar_statistics_pairwise(
        container,
        dc.AcfSettings(
            STRIP_INVALID_DATA_FALSE,
            const.ACF_LIDAR_LEGS,
            const.LIDAR_DELTA_TIME,
        ),
    )


def get_acf_inverse_pairwise_strip_invalid_data_for_test(
    container: dc.LidarContainer,
) -> dc.LidarContainerResult:
    return __get_acf_lidar_statistics_pairwise(
        container,
        dc.AcfSettings(
            STRIP_INVALID_DATA_TRUE,
            const.ACF_LIDAR_LEGS,
            const.LIDAR_DELTA_TIME,
            test=True,
        ),
    )

def __get_lidar_plot(container: dc.LidarContainer) -> dc.FigureResult:
    """
    Method is only for explaining autocorrelation in the thesis for one student
    :param container:
    :return:
    """
    if(container.id != "student-5-[e3dcc4ee-3201-4a7b-81d9-509a0aa6d26d]-export"):
        return

    filtered_data_df = container.get_filtered_data_container()

    filtered_data_df[container.data_column_name] = sc.calculate_inverse_values(
        filtered_data_df[container.data_column_name].to_numpy()
    )
    number_elements = 80 # equals to 8 seconds
    first_datapoints = filtered_data_df.dropna().head(number_elements)

    function = lambda folder: plot.plot_lidar_distance_thesis_plot(
        first_datapoints[columns.SENSOR_COLUMN_DISTANCE],
        number_elements,
        folder,
    )

    return dc.FigureResult(
        container.id,
        function,
        "acf_to_compare.{}".format(plot.IMAGE_FORMAT),
        "thesis_image",
    )


def __getLidarStatistics(container: dc.LidarContainer) -> dc.ContainerResult:
    distanceSensorDf = container.get_data_container()

    newDataframe = distanceSensorDf.describe()

    invalidValuesPercent = (
        (
            distanceSensorDf.where(
                distanceSensorDf[columns.SENSOR_COLUMN_DISTANCE]
                > const.MAX_SENSOR_DISTANCE
            )[columns.SENSOR_COLUMN_DISTANCE].count()
            / distanceSensorDf.count()
        )
        * 100
    ).values[0]

    newData = pd.DataFrame(
        {columns.SENSOR_COLUMN_DISTANCE: round(invalidValuesPercent, 3)},
        index=["Percentage of invalid values"],
    )
    newDataframe = pd.concat([newDataframe, newData])

    newColumnNames = {
        columns.SENSOR_COLUMN_DISTANCE: container.id
        + "-"
        + columns.SENSOR_COLUMN_DISTANCE,
    }
    newDataframe.rename(columns=newColumnNames, inplace=True)

    return dc.ContainerResult(container.id, data=newDataframe)


def get_lidar_min_distance_statistic(
    container: dc.LidarContainer,
) -> dc.ContainerResult:
    distanceSensorDf = container.get_data_container()

    dfMinDistanceRange = distanceSensorDf.where(
        filters.getInvalidLidardDistanceLowerEnd(distanceSensorDf)
    )

    dfCountDistanceRange = pd.DataFrame(
        {columns.SENSOR_COLUMN_DISTANCE: dfMinDistanceRange.count().values[0]},
        # XXX hack integer value is used for sorting
        index=[99999999],
    )

    dfValueCount = pd.DataFrame(
        dfMinDistanceRange[columns.SENSOR_COLUMN_DISTANCE].value_counts()
    )

    newDataframe = pd.concat([dfValueCount, dfCountDistanceRange])

    newColumnNames = {
        columns.SENSOR_COLUMN_DISTANCE: container.id
        + "-"
        + "count less equals {}mm ".format(const.MIN_SENSOR_DISTANCE_100)
        + columns.SENSOR_COLUMN_DISTANCE,
    }
    newDataframe.rename(columns=newColumnNames, inplace=True)

    return dc.ContainerResult(container.id, newDataframe.convert_dtypes())


def __get_valid_lidar_statistics(
    container: dc.LidarContainer,
):
    distanceSensorDf = container.get_filtered_data_container()

    count_values = distanceSensorDf[container.data_column_name].count()

    distanceSensorDf[container.data_column_name] = sc.calculate_inverse_values(
        distanceSensorDf[container.data_column_name].to_numpy()
    )
    # just checking that the inverse distance didn't touch values.
    assert count_values == distanceSensorDf[container.data_column_name].count()

    distanceSensorDf.dropna(inplace=True)

    function = lambda folder: plot.plot_lidar_distance_histogram(
        distanceSensorDf, folder
    )

    newDataframe = distanceSensorDf.describe()

    totalDistances = container.get_data_container().count().values[0]

    newData = pd.DataFrame(
        {columns.SENSOR_COLUMN_DISTANCE: totalDistances},
        index=["Number of all distances"],
    )

    newDataframe = pd.concat([newDataframe, newData])

    newColumnNames = {
        columns.SENSOR_COLUMN_DISTANCE: container.id
        + "-"
        + "valid distances {}m <= x <= {}(m) ".format(
            const.MIN_SENSOR_DISTANCE_100, const.MAX_DISTANCE_UPPER_LIMIT
        )
        + columns.SENSOR_COLUMN_DISTANCE,
    }
    newDataframe.rename(columns=newColumnNames, inplace=True)

    return dc.LidarContainerResult(
        container.id,
        newDataframe,
        function,
        "lidar_histogram.{}".format(plot.IMAGE_FORMAT),
        const.LIDAR_PLOT_FOLDER,
    )


def mask_distance_sensor_data(df: pd.DataFrame) -> pd.DataFrame:
    # mark data lower 100 mm as invalid therefore nan.
    filtered_data = df.mask(
        filters.getInvalidLidardDistanceLowerEnd(df, const.MIN_SENSOR_DISTANCE_100),
        np.nan,
    )

    # only mark values with inf which are above measurements
    filtered_data[columns.SENSOR_COLUMN_DISTANCE] = filtered_data[
        columns.SENSOR_COLUMN_DISTANCE
    ].mask(
        filters.getInvalidLidardDistanceUpperEnd(filtered_data),
        float("inf"),
    )

    return filtered_data


@DeprecationWarning
def mask_distance_sensor_data_old(
    df: pd.DataFrame, fill_with: np.float = float("inf")
) -> pd.DataFrame:
    # mark values < 100mm and values >1800mm as infinity
    return df.mask(
        filters.getInvalidLidardDistanceLowerEnd(df, const.MIN_SENSOR_DISTANCE_100)
        | filters.getInvalidLidardDistanceUpperEnd(df)
        | filters.getInvalidLidarDistanceNan(df),
        fill_with,
    )


def __get_acf_lidar_plot_title(settings: dc.AcfSettings):
    title_distance = "inverse LIDAR distance"

    # title_filter = (
    #    "removed invalids before calculation"
    #    if settings.strip_invalid_data
    #    else " included invalids in calculation".format()
    # )

    # return "ACF plot - {} - {} <br><sup>Valid values: ({}(mm) >= x <= {}(mm))</sup>".format(
    #    title_distance,
    #    title_filter,
    #    const.MIN_SENSOR_DISTANCE_100,
    #    const.MAX_DISTANCE_UPPER_LIMIT,
    # )

    return "ACF plot - {}".format(
        title_distance,
    )


def get_lidar_container(
    fileContainer: SubjectFileContainer,
    load_all_columns: bool = False,
    remove_duplicate_index: bool = True,
) -> dc.LidarContainer:
    """

    :param fileContainer:
    :param load_all_columns: whether to load all column in the original file. Otherwise, only a subset is loaded.
    :param remove_duplicate_index: whether to keep duplicate indexes.
    :return:
    """
    columns_to_use = (
        columns.DISTANCE_SENSOR_COLUMNS_ALL
        if load_all_columns
        else [columns.DISTANCE_SENSOR_COLUMN_INDEX, columns.SENSOR_COLUMN_DISTANCE]
    )

    distance_sensor_df = pd.read_csv(
        fileContainer.pathToSensorDataDistance,
        header=0,
        usecols=columns_to_use,
        index_col=columns.DISTANCE_SENSOR_COLUMN_INDEX,
    )

    if remove_duplicate_index and not distance_sensor_df.index.is_unique:
        """
        Why can a duplicate index occur?
        LIDAR sensor records at 100ms
        If a recording gets delayed (e.g. network delay) so that two recordings have nearly the same timestamp both get
        the same index. A test covers this. If this occurs, the latest entry is taken, assuming that
        the latest is closed for the index.
        We talking here about less than 5 entries per file.

        remove_duplicate_index can be ignored as well. but its important for ACF
        """
        # just keep the last value
        distance_sensor_df = distance_sensor_df[
            ~distance_sensor_df.index.duplicated(keep="last")
        ]
        assert distance_sensor_df.index.is_unique

    filtered_df = mask_distance_sensor_data(distance_sensor_df)

    assert filtered_df[columns.SENSOR_COLUMN_DISTANCE].min() >= 100

    return dc.LidarContainer(fileContainer.id, distance_sensor_df, filtered_df)


def __get_filtered_lidar_data_simone(
    container: dc.LidarContainer,
) -> dc.ContainerResult:
    filtered_data_container = container.get_filtered_data_container().dropna()

    filtered_data_container[columns.DISTANCE_SENSOR_INVERSE_COLUMN] = pd.Series(
        sc.calculate_inverse_values(
            filtered_data_container[container.data_column_name].to_numpy()
        ),
        index=filtered_data_container.index,
    )

    return dc.ContainerResult(container.id, filtered_data_container.convert_dtypes())


def __get_acf_lidar_statistics_pairwise(
    container: dc.LidarContainer, settings: dc.AcfSettings
) -> dc.LidarContainerResult:
    values_for_acf = sc.get_values_for_acf(container, settings)

    correlationValuesTotal = np.array(
        [
            sm.acf_pairwise_calculation(values_for_acf, lag)
            for lag in range(settings.lags)
        ]
    )

    correlation_transformed = correlationValuesTotal

    acfDf = pd.DataFrame(
        correlation_transformed,
        index=np.arange(
            0.0, len(correlation_transformed) * settings.delta_time, settings.delta_time
        ),
        columns=[container.id],
    )

    if settings.test:
        assert settings.strip_invalid_data

        acf_differences = acf_test_against_original(
            values_for_acf, correlation_transformed
        )

        data = pd.DataFrame({container.id: acf_differences})

        # row 0 is has value 0 - checked before - remove row
        data = data.drop([0]).describe()

        return dc.LidarContainerResult(container.id, data, None)

    x_intercept_first_slope = np.round(
        validations.calculateFirstSlope(
            acfDf[container.id].index[1],  # 0.1
            acfDf[container.id].iloc[1],  # value y[1]
            acfDf[container.id].iloc[0],  # value y[0] = 1
        ),
        DECIMAL_POINTS_ACF,
    )
    delta_per_time_shift = 1 - acfDf[container.id].iloc[1]
    y = 1
    slopes = [y]
    for x in np.arange(0, x_intercept_first_slope, settings.delta_time):
        y -= delta_per_time_shift
        slopes.append(y)

    plot_file_name = "acfInversePairwise"

    plot_file_name += "_stripedInvalids" if settings.strip_invalid_data else "_fulldata"
    plot_file_name += ".{}".format(plot.IMAGE_FORMAT)

    yaxis_title = "ACF"
    # XXX call method at a different position
    funct = lambda file_path: plot.plotAcfByArray(
        acfDf[container.id],
        title=__get_acf_lidar_plot_title(settings),
        file_path=file_path,
        yaxis_title=yaxis_title,
        slope_array=slopes,
        slope_time=round(x_intercept_first_slope, 2),
    )

    time_lag = pd.DataFrame(
        {container.id: x_intercept_first_slope},
        index=[columns.ACF_COLUMN_INTERCEPT_SLOPE],
    )
    data = pd.concat([time_lag, acfDf])

    return dc.LidarContainerResult(container.id, data, funct, plot_file_name)


def acf_test_against_original(
    data_for_acf: np.ndarray, calculated_acf_data: np.ndarray
) -> np.ndarray:
    # XXX cannot cope with NAN values
    acf_original_method = sm.acf_by_statistic_library(
        data_for_acf, len(calculated_acf_data) - 1
    ).T
    # error_to_mean_in_percent = calculate_in_percent(correlationValuesTotal.T[1])
    # error_to_mean = np.divide(calculated_acf_data.T[1], calculated_acf_data.T[0])
    difference = np.subtract(calculated_acf_data, acf_original_method)

    assert difference[0] == 0.0

    return difference


def calculate_in_percent(values) -> []:
    validations.checkAtLeastOneElement(values)
    transformed_values = []

    for i in range(len(values)):
        transformed_values.append((values[i] / values[0]))

    return np.array(transformed_values)


def process_lidar_data(
    subjectFileContainers: [SubjectFileContainer],
) -> None:
    """
    See process_lidar_data_acf (similar method)
    :param subjectFileContainers: Container holding student file information.
    """
    validations.checkAtLeastOneElement(subjectFileContainers)

    outputPath = subjectFileContainers[0].pathToOutputFolder

    start = timer()
    subject_lidar_containers = np.array(
        [
            get_lidar_container(c, remove_duplicate_index=False)
            for c in subjectFileContainers
        ]
    )
    print("Loading lidar data took:", timedelta(seconds=timer() - start))

    sc.perform_on_students(
        outputPath,
        "filteredLidarInverseDistancesStatistics.csv",
        subject_lidar_containers,
        __get_valid_lidar_statistics,
    )

    sc.perform_on_students(
        outputPath,
        "lidarMinDistanceStatistics.csv",
        subject_lidar_containers,
        get_lidar_min_distance_statistic,
        sortIndex=True,
    )

    sc.perform_on_students_only_fig(outputPath, subject_lidar_containers, __get_lidar_plot)


def process_lidar_data_acf(
    subjectFileContainers: [SubjectFileContainer],
) -> None:
    """
    This method differs from "process_lidar_data" only that duplicate index entries are removed.
    :param subjectFileContainers: Container holding student file information.
    """
    validations.checkAtLeastOneElement(subjectFileContainers)

    outputPath = subjectFileContainers[0].pathToOutputFolder

    start = timer()
    subject_lidar_containers = np.array(
        [get_lidar_container(c) for c in subjectFileContainers]
    )
    print("Loading lidar data took:", timedelta(seconds=timer() - start))

    sc.perform_on_students(
        outputPath,
        "acfInverseStatisticsPairwise.csv",
        subject_lidar_containers,
        get_acf_inverse_statistics_pairwise_lidar,
    )

    sc.perform_on_students(
        outputPath,
        "test_against_original_method.csv",
        subject_lidar_containers,
        get_acf_inverse_pairwise_strip_invalid_data_for_test,
    )

    # XXX re-enable if needed methods
    return

    # TODO statistic seems useless.
    # TODO plot LIDAR distance over time
    # one folder per statistic
    __perform_on_students(
        outputPath,
        "lidarDistancesStatistics.csv",
        subject_lidar_containers,
        __getLidarStatistics,
    )


def filter_lidar_data_for_simone(subjectFileContainers: [SubjectFileContainer]):
    validations.checkAtLeastOneElement(subjectFileContainers)

    outputPath = subjectFileContainers[0].pathToOutputFolder

    subject_lidar_containers = np.array(
        [
            get_lidar_container(c, load_all_columns=True, remove_duplicate_index=False)
            for c in subjectFileContainers
        ]
    )

    function = lambda container: __get_filtered_lidar_data_simone(
        container,
    )

    __perform_on_student_for_simone(
        function, outputPath, subject_lidar_containers, "_sensorsDistanceFiltered.csv"
    )


def __perform_on_student_for_simone(
    function, outputPath, subject_lidar_containers, file_suffix: str
):
    results = np.array([function(subjects) for subjects in subject_lidar_containers])
    for result in results:
        statisticFolder = os.path.join(outputPath, const.SUB_FOLDER_FILTERED_DATE)
        validations.createDirIfNotExists(statisticFolder)

        file_name = result.id + file_suffix
        outputFile = os.path.join(statisticFolder, file_name)
        with open(outputFile, "w") as out:
            result.data.to_csv(out)
            print("Finished file:", file_name)
