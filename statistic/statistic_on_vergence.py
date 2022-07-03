import pandas as pd
import numpy as np
from util import constants as const, csv_columns as columns, filters
from util import validations
import plot
import model.DataContainer as dc
from statistic import statistic_methods as sm
from statistic import statistic_common as sc

from timeit import default_timer as timer
from datetime import timedelta

from model import SubjectFileContainer

# np.seterr('raise') use for debugging.
np.seterr("raise")
pd.options.display.float_format = "{:.5f}".format

SUB_FOLDER_STATISTICS = "statistics"
SUB_FOLDER_FILTERED_DATE = "filteredData"
DECIMAL_POINTS_ACF = 5

STRIP_INVALID_DATA_TRUE = True
STRIP_INVALID_DATA_FALSE = False

INVERSE_DISTANCE_TRUE = True

# TODO WIP

def __get_vergence_container(
    fileContainer: SubjectFileContainer,
) -> dc.VergenceContainer:
    vergence_df = pd.read_csv(
        fileContainer.pathToEyeDataWithReference,
        header=0,
        usecols=[
            columns.EYE_TRACKING_COLUMN_FRAME_INDEX,
            columns.EYE_TRACKING_COLUMN_VERGENCE_ANGLE,
            columns.EYE_TRACKING_COLUMN_VERGENCE_ERROR,
            columns.EYE_TRACKING_DISTANCE_BETWEEN_POINTS,
        ],
        low_memory=False,
        index_col=columns.EYE_TRACKING_COLUMN_FRAME_INDEX,
        na_values="-",
    )

    assert vergence_df.index.is_unique

    filtered_df = mask_eye_vergence_data(vergence_df)
    validations.checkContainsNoNotNanValues(filtered_df)

    return dc.VergenceContainer(fileContainer.id, vergence_df, filtered_df)


def __get_vergence_container_basic(
    fileContainer: SubjectFileContainer,
) -> dc.VergenceContainer:
    vergence_df = pd.read_csv(
        fileContainer.pathToEyeDataWithReference,
        header=0,
        usecols=[
            columns.EYE_TRACKING_COLUMN_FRAME_INDEX,
            columns.EYE_TRACKING_COLUMN_VERGENCE_ANGLE,
            columns.EYE_TRACKING_COLUMN_VERGENCE_ERROR,
            columns.EYE_TRACKING_COLUMN_GAZE_POINT_Z,
            columns.EYE_TRACKING_COLUMN_ZERO_POINT_Z,
        ],
        low_memory=False,
        index_col=columns.EYE_TRACKING_COLUMN_FRAME_INDEX,
        na_values="-",
    )

    assert vergence_df.index.is_unique

    filtered_df = vergence_df.dropna()
    validations.checkContainsNoNotNanValues(filtered_df)

    return dc.VergenceContainer(fileContainer.id, vergence_df, filtered_df)


def __get_vergence_angle_statistic(
    container: dc.VergenceContainer,
) -> dc.ContainerResult:
    eye_tracking_vergence_df = container.get_filtered_data_container()

    eye_tracking_vergence_df["distance_z"] = (
        eye_tracking_vergence_df[columns.EYE_TRACKING_COLUMN_GAZE_POINT_Z]
        - eye_tracking_vergence_df[columns.EYE_TRACKING_COLUMN_ZERO_POINT_Z]
    )

    negative_eye_distances = (
        eye_tracking_vergence_df.where(eye_tracking_vergence_df["distance_z"] < 0)
        .count()
        .values[0]
    )

    number_of_entries = pd.DataFrame(
        {container.id: eye_tracking_vergence_df.count().values[0]},
        index=["Number of entries"],
    )

    countWithoutNegativeAngles = pd.DataFrame(
        {container.id: negative_eye_distances},
        index=["Count of negative gaze z distances"],
    )

    df = pd.concat([number_of_entries, countWithoutNegativeAngles])

    return dc.ContainerResult(container.id, df)


def process_vergence_data_basic(subjectFileContainers: [SubjectFileContainer]):
    outputPath = subjectFileContainers[0].pathToOutputFolder

    start = timer()
    subject_vergence_containers = np.array(
        [__get_vergence_container_basic(c) for c in subjectFileContainers]
    )
    print("Loading vergence data took:", timedelta(seconds=timer() - start))

    sc.perform_on_students(
        outputPath,
        "gaze_distance_z_statistic.csv",
        subject_vergence_containers,
        __get_vergence_angle_statistic,
    )


# TODO this method was implemented but not used therefore the output is not tested for validity
# TODO additionally filters have to be checked whether they still are valid.
def process_vergence_data(subjectFileContainers: [SubjectFileContainer]):
    validations.checkAtLeastOneElement(subjectFileContainers)

    raise Exception("Redesign, refactor, implement this method to your needs")
    outputPath = subjectFileContainers[0].pathToOutputFolder

    start = timer()
    subject_vergence_containers = np.array(
        [__get_vergence_container(c) for c in subjectFileContainers]
    )
    print("Loading vergence data took:", timedelta(seconds=timer() - start))

    sc.perform_on_students(
        outputPath,
        "acfVergencePairwise.csv",
        subject_vergence_containers,
        get_acf_vergence_statistics_pairwise,
    )
    sc.perform_on_students(
        outputPath,
        "vergenceAngleStatistics.csv",
        subject_vergence_containers,
        __get_vergence_angle_statistic,
    )

    return None
    __perform_on_students_only_fig(
        outputPath,
        subject_vergence_containers,
        __get_vergence_angle_to_reciprocal_distance,
    )


def __get_vergence_angle_to_reciprocal_distance(
    container: dc.LidarContainer,
) -> dc.FigureResult:
    filtered_df = container.get_filtered_data_container()

    filtered_df[columns.EYE_TRACKING_DISTANCE_BETWEEN_POINTS] = filtered_df[
        columns.EYE_TRACKING_DISTANCE_BETWEEN_POINTS
    ].rdiv(1)

    file_name = "vergence_angel_to_distance." + plot.IMAGE_FORMAT
    title = "Vergence angle to reciprocal distance between gaze points <br><sup>Valid values: ({}radians >= x <= {}radians)</sup>".format(
        const.FILTER_VERGENCE_ANGLE_LOWER_END,
        const.FILTER_VERGENCE_ANGLE_UPPER_END,
    )

    function = lambda output_path: plot.plotVerganceAngleToReciprocalDistance(
        filtered_df,
        columns.EYE_TRACKING_DISTANCE_BETWEEN_POINTS,
        columns.EYE_TRACKING_COLUMN_VERGENCE_ANGLE,
        title,
        output_path,
    )

    return dc.FigureResult(
        container.id,
        function,
        file_name,
        const.VERGENCE_PLOT_FOLDER,
    )


def get_acf_vergence_statistics_pairwise(
    container: dc.VergenceContainer,
):
    lags = const.VIDEO_FPS * 3  # equals 15 seconds
    # TODO enable
    # lags = const.VIDEO_FPS * 15  # equals 15 seconds

    return __get_acf_vergence_statistics_pairwise(
        container,
        dc.AcfSettings(
            STRIP_INVALID_DATA_FALSE,
            lags,
            const.VERGENCE_DELTA_TIME,
        ),
    )


def __get_acf_vergence_plot_title(settings: dc.AcfSettings):
    title_vergence = "inverse vergence angle"

    title_filter = (
        "removed invalids before calculation"
        if settings.strip_invalid_data
        else " included invalids in calculation".format()
    )

    return "ACF plot - {} - {} <br><sup>Valid values: ({}radians >= x <= {}radians)</sup>".format(
        title_vergence,
        title_filter,
        const.FILTER_VERGENCE_ANGLE_LOWER_END,
        const.FILTER_VERGENCE_ANGLE_UPPER_END,
    )


def __get_vergence_angle_statistic(
    container: dc.VergenceContainer,
) -> dc.VergenceContainer:
    eye_tracking_vergence_df = container.get_data_container()

    countTotalSize = pd.DataFrame(
        {
            columns.EYE_TRACKING_COLUMN_VERGENCE_ANGLE: eye_tracking_vergence_df[
                columns.EYE_TRACKING_COLUMN_VERGENCE_ANGLE
            ].count()
        },
        index=["Total count of values"],
    )

    dfWithoutNegativeAngles = eye_tracking_vergence_df.where(
        ~filters.getInvalidNegativVergenceAngle(eye_tracking_vergence_df)
    )

    countWithoutNegativeAngles = pd.DataFrame(
        {
            columns.EYE_TRACKING_COLUMN_VERGENCE_ANGLE: dfWithoutNegativeAngles.count().values[
                0
            ]
        },
        index=["Count of positive vergence angle"],
    )

    sPositiveAngleAndFilteredOutAngleErrors = dfWithoutNegativeAngles.where(
        ~filters.getInvalidVergenceErrorCriteria(dfWithoutNegativeAngles)
    )

    countPositiveValidVergenceAngle = pd.DataFrame(
        {
            columns.EYE_TRACKING_COLUMN_VERGENCE_ANGLE: sPositiveAngleAndFilteredOutAngleErrors.count().values[
                0
            ]
        },
        index=["Count positive angles and (vergence angle >= vergence error)"],
    )

    countAnglesUpperEndAndFilterOutAngleErrors = pd.DataFrame(
        {
            columns.EYE_TRACKING_COLUMN_VERGENCE_ANGLE: sPositiveAngleAndFilteredOutAngleErrors.where(
                filters.getInvalidVergenceAngleUpperEnd(
                    sPositiveAngleAndFilteredOutAngleErrors,
                    const.FILTER_VERGENCE_ANGLE_UPPER_END,
                )
            )
            .count()
            .values[0]
        },
        index=[
            "Count vergence angle greater equals {} (unit rad) and (vergence angle >= vergence error)".format(
                const.FILTER_VERGENCE_ANGLE_UPPER_END
            )
        ],
    )

    """
        get datapoins which are >= 0 and <= lower end (const.FILTER_VERGENCE_ANGLE_LOWER_END)
    """
    countAnglesLowerEndAndFilterOutAngleErrors = pd.DataFrame(
        {
            columns.EYE_TRACKING_COLUMN_VERGENCE_ANGLE: sPositiveAngleAndFilteredOutAngleErrors.where(
                filters.getInvalidVergenceAngleLowerEnd(
                    sPositiveAngleAndFilteredOutAngleErrors
                )
            )
            .count()
            .values[0]
        },
        index=[
            "Count vergence angle less than {} (unit rad) and (vergence angle >= vergence error)".format(
                const.FILTER_VERGENCE_ANGLE_LOWER_END
            )
        ],
    )

    countValidAngles = pd.DataFrame(
        {
            columns.EYE_TRACKING_COLUMN_VERGENCE_ANGLE: filters.filterByValidVergence(
                eye_tracking_vergence_df
            )
            .count()
            .values[0]
        },
        index=["Count of 'valid' values"],
    )

    descriptiveDf = (
        sPositiveAngleAndFilteredOutAngleErrors[
            [columns.EYE_TRACKING_COLUMN_VERGENCE_ANGLE]
        ]
        .describe()
        .applymap(lambda x: f"{x:0.3f}")
    )

    concatDataframe = pd.concat(
        [
            countTotalSize,
            countWithoutNegativeAngles,
            countPositiveValidVergenceAngle,
            countValidAngles,
            countAnglesLowerEndAndFilterOutAngleErrors,
            countAnglesUpperEndAndFilterOutAngleErrors,
            descriptiveDf,
        ]
    )

    newColumnNames = {
        columns.EYE_TRACKING_COLUMN_VERGENCE_ANGLE: container.id
        + "-"
        + columns.EYE_TRACKING_COLUMN_VERGENCE_ANGLE,
    }

    return dc.VergenceContainerResult(
        container.id, concatDataframe.rename(columns=newColumnNames)
    )


def __get_acf_vergence_statistics_pairwise(
    container: dc.VergenceContainer,
    settings: dc.AcfSettings,
):
    start = timer()
    values_for_acf = sc.get_values_for_acf(container, settings)

    correlationValuesTotal = np.array(
        [
            sm.acf_pairwise_calculation(values_for_acf, lag)
            for lag in range(settings.lags)
        ]
    )
    print(
        "Vergence calculating ACF took:",
        container.id,
        timedelta(seconds=timer() - start),
    )

    correlation_transformed = correlationValuesTotal

    acfDf = pd.DataFrame(
        correlation_transformed,
        index=np.round(
            np.arange(0.0, settings.lags * settings.delta_time, settings.delta_time), 3
        ),
        columns=[container.id],
    )

    x_incercept_first_slope = np.round(
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
    for x in np.arange(0, x_incercept_first_slope, settings.delta_time):
        y -= delta_per_time_shift
        slopes.append(y)

    plot_file_name = "vergence_acf." + plot.IMAGE_FORMAT

    yaxis_title = "ACF"
    # XXX call method at a different position
    fig_function = lambda file_path: plot.plotAcfByArray(
        acfDf[container.id],
        title=__get_acf_vergence_plot_title(settings),
        file_path=file_path,
        yaxis_title=yaxis_title,
        slope_array=slopes,
        slope_time=x_incercept_first_slope,
    )

    time_lag = pd.DataFrame(
        {container.id: x_incercept_first_slope},
        index=[columns.ACF_COLUMN_INTERCEPT_SLOPE],
    )
    data = pd.concat([time_lag, acfDf])

    acf_differences = None
    if False:  # settings.strip_invalid_data:
        # XXX method cannot cope with NAN values
        acf_differences = acf_test_against_original(
            values_for_acf, correlation_transformed, settings.lags
        )
        acf_differences = None if not settings.strip_invalid_data else acf_differences

        # TODO map to dataframe
        # pd.DataFrame

    print("ACF Vergence took:", container.id, timedelta(seconds=timer() - start))
    return dc.VergenceContainerResult(
        container.id, data, fig_function, plot_file_name, None
    )


# TODO discuss which filters should be applied on the data.
def mask_eye_vergence_data(
    df: pd.DataFrame, fill_with: np.float = float("inf")
) -> pd.DataFrame:
    return df.mask(
        filters.getInvalidVergenceErrorCriteria(df)
        | filters.getInvalidVergenceAngleLowerEnd(df)
        | filters.getInvalidVergenceAngleUpperEnd(df)
        | filters.getInvalidVergenceAngleNan(df),
        fill_with,
    )
