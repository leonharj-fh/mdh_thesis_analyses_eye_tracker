import pandas as pd
import os
import numpy as np
import plot
import model.AppSettings as ap
import model.StudentSettings as student
import re

from util import csv_columns as columns, configLoader, constants as const, validations
from itertools import chain
from model import SubjectFileContainer
from statistic import (
    statistic_methods as sm,
    statistic_common as sc,
    statistic_on_vergence as sv,
    statistic_on_lidar as sl,
)

from timeit import default_timer as timer
from datetime import timedelta

from statsmodels.stats.stattools import durbin_watson


# np.seterr('raise') use for debugging.
np.seterr("raise")
pd.options.display.float_format = "{:.5f}".format


COLUMN_IS_LAB_RECORDING = "is_lab_recording"
COLUMN_STUDENT_INDEX = "student_index"


def compare_lidar_gaze_distance_list(subjectFileContainers: [SubjectFileContainer]):
    compare_lidar_gaze_distance(subjectFileContainers[0])


# XXX not tested,
# XXX not used
def compare_lidar_gaze_distance(fileContainer: SubjectFileContainer):
    """
    TODO under construction
    :param fileContainer:
    :return:
    """
    eyeTrackingDf = pd.read_csv(
        fileContainer.pathToEyeDataWithReference,
        header=0,
        usecols=[
            columns.TIMESTAMP_COLUMN,
            columns.EYE_TRACKING_COLUMN_DISTANCE_SENSOR_REFERENCE,
            columns.EYE_TRACKING_DISTANCE_BETWEEN_POINTS,
            columns.EYE_TRACKING_COLUMN_VERGENCE_ANGLE,
            columns.EYE_TRACKING_COLUMN_VERGENCE_ERROR,
        ],
        na_values="-",
    )

    COLUMN_TIME_DIFFERENCE = "diff"

    eyeTrackingDf[COLUMN_TIME_DIFFERENCE] = np.abs(
        eyeTrackingDf[columns.TIMESTAMP_COLUMN]
        - eyeTrackingDf[columns.EYE_TRACKING_COLUMN_DISTANCE_SENSOR_REFERENCE]
    )

    groupByResult = eyeTrackingDf.sort_values(
        by=[
            columns.EYE_TRACKING_COLUMN_DISTANCE_SENSOR_REFERENCE,
            COLUMN_TIME_DIFFERENCE,
        ]
    ).groupby(columns.EYE_TRACKING_COLUMN_DISTANCE_SENSOR_REFERENCE)

    start = timer()
    first_value_per_key = [v[0] for v in groupByResult.groups.values()]

    # different method would be eyeTrackingDf[eyeTrackingDf.index.isin(first_value_per_key)]
    filtered_df = eyeTrackingDf.merge(
        pd.DataFrame(index=first_value_per_key),
        left_index=True,
        right_index=True,
        how="inner",
    )
    print("Merge keys took:", timedelta(seconds=timer() - start))

    assert len(first_value_per_key) == len(filtered_df.index.to_list())

    distanceSensorDf = pd.read_csv(
        fileContainer.pathToSensorDataDistance,
        header=0,
        usecols=[columns.TIMESTAMP_COLUMN, columns.SENSOR_COLUMN_DISTANCE],
        index_col=columns.TIMESTAMP_COLUMN,
    )

    joinedData = pd.merge(
        eyeTrackingDf,
        distanceSensorDf,
        left_on=columns.EYE_TRACKING_COLUMN_DISTANCE_SENSOR_REFERENCE,
        right_index=True,
        how="inner",
    )

# TODO refactor, redesign, rework
def __get_and_prepare_dataframes_for_statistical_analyses(
    yearToSubjectFileContainers: dict, student_settings: student.StudentSettings
) -> dict:
    acfInverseDataDict = {}

    for year, containers in yearToSubjectFileContainers.items():
        validations.checkAtLeastOneElement(containers)
        start = timer()
        subject_lidar_containers = np.array(
            [sl.get_lidar_container(c) for c in containers]
        )
        print("Loading lidar data took:", timedelta(seconds=timer() - start))

        data_df = sc.perform_on_students_basic(
            subject_lidar_containers, sl.get_acf_inverse_statistic_pairwise_basic_lidar
        )

        data_df[COLUMN_IS_LAB_RECORDING] = list(
            map(
                lambda x: student_settings.is_lab_recording(
                    year, re.search("\\[(.*)\\]", x).group(1)
                ),
                data_df.index.to_list(),
            ),
        )

        # check that at least one recoding in lab and in a classroom
        assert data_df[COLUMN_IS_LAB_RECORDING].values.any()
        assert not data_df[COLUMN_IS_LAB_RECORDING].values.all()



        data_df[COLUMN_STUDENT_INDEX] = list(
            map(
                lambda x: int(re.search("student-([0-9]+).*-\\[", x).group(1)),
                data_df.index.to_list(),
            ),
        )

        data_df["slopeValue"] = (1 - data_df[sm.const.LIDAR_DELTA_TIME]) / -sm.const.LIDAR_DELTA_TIME

        refraction_error_students_df = pd.read_csv(
            configLoader.getRefractionErrorFile(year),
            index_col=[0],
            skipinitialspace=True,
        ).convert_dtypes()

        refraction_error_students_df["myopic"] = (
            refraction_error_students_df[columns.HEADER_REFRACTION_ERROR_RIGHT_EYE]
            + refraction_error_students_df[columns.HEADER_REFRACTION_ERROR_LEFT_EYE]
        ) < 0  # refraction error is negative

        joined_data = pd.merge(
            data_df,
            refraction_error_students_df,
            left_on=COLUMN_STUDENT_INDEX,
            right_index=True,
        )

        joined_data["refract_error"] = np.true_divide(
            np.add(
                joined_data[columns.HEADER_REFRACTION_ERROR_RIGHT_EYE].to_numpy(),
                joined_data[columns.HEADER_REFRACTION_ERROR_LEFT_EYE].to_numpy(),
            ),
            2,
        )

        if len(joined_data) != len(data_df):
            raise Exception("Error mismatch of refraction error indexes.")

        acfInverseDataDict[year] = joined_data.copy()

    return acfInverseDataDict


# TODO refactor, redesign, ...
def generate_statistics_comparison_statistical_tests(
    yearToSubjectFileContainers: dict,
    app_settings: ap.AppSettings,
    student_settings: student.StudentSettings,
) -> None:
    validations.checkAtLeastOneElement(yearToSubjectFileContainers)

    delta_time_seconds = sl.const.LIDAR_DELTA_TIME
    acfInverseDataDfDict = __get_and_prepare_dataframes_for_statistical_analyses(
        yearToSubjectFileContainers, student_settings
    )

    assert 2018 in acfInverseDataDfDict
    assert 2019 in acfInverseDataDfDict

    acfInverseDataDict = {}
    for year in acfInverseDataDfDict.keys():
        df = acfInverseDataDfDict[year]
        df = df.where(~df[COLUMN_IS_LAB_RECORDING]).dropna()

        if student_settings.merge_same_student_recordings:
            df = df.groupby(by=[COLUMN_STUDENT_INDEX]).mean(numeric_only=True)
            df["myopic"] = df["myopic"].astype(bool)

        print("############## KS-Test #####################")
        calculate_ks_emmetropic_myopic(
            df,
            delta_time_seconds,
            year,
        )

        acfInverseDataDict[year] = df["slopeValue"].to_numpy(dtype=np.float64)
        #acfInverseDataDict[year] = df[delta_time_seconds].to_numpy(dtype=np.float64)
        acfInverseDataDfDict[year] = df  # update entry

    thesis_output_folder = os.path.join(
        app_settings.commonOutputPath,
        const.SUB_FOLDER_THESIS,
    )

    output_file = os.path.join(
        thesis_output_folder,
        "cdf_school_comparison_for_ks.{}".format(plot.IMAGE_FORMAT),
    )

    plot.cdf_plot(acfInverseDataDict, output_file)

    print("############## Mean + Std of data #####################")
    print("Descriptive Statistic 2018:", pd.DataFrame(acfInverseDataDict[2018], dtype=np.float64).describe())
    print("Descriptive Statistic 2019:", pd.DataFrame(acfInverseDataDict[2019], dtype=np.float64).describe())

    print("############## KS-Test #####################")
    print(
        "Entries 2018: {}, entries 2019: {}".format(
            len(acfInverseDataDict[2018]), len(acfInverseDataDict[2019])
        )
    )
    #result = sm.ks_2samp_manuel(acfInverseDataDict[2018], acfInverseDataDict[2019])
    #print(result)
    result = sm.ks_2sampled(acfInverseDataDict[2018], acfInverseDataDict[2019])
    print("KS-Test 2018, 2019 students", result)
    print("############################################")

    result = sm.ks_2sampled(acfInverseDataDict[2018], acfInverseDataDict[2019], alternative="greater")
    print("KS-Test 2018 greater than 2019 students", result)
    print("############################################")

    print("############## Normal distribution-Test #####################")
    result = sm.shapiro_wilk_test(
        np.array(list(chain(*acfInverseDataDict.values())), dtype=np.float64)
    )
    print("Normal distributed both data sets?: ", result)

    for year in yearToSubjectFileContainers.keys():
        result = sm.lilliefors_test_norm(acfInverseDataDict[year])
        print("Normal distributed?: ", year, result)

        result = sm.shapiro_wilk_test(acfInverseDataDict[year])
        print("Normal distributed?: ", year, result)

        result = sm.normal_test(acfInverseDataDict[year])
        print("Normal distributed?: ", year, result)

        sm.qq_plot_for_data(
            acfInverseDataDict[year],
            os.path.join(
                thesis_output_folder,
                "qq_plot_distribution_{}.{}".format(year, plot.IMAGE_FORMAT),
            ),
        )


def calculate_ks_emmetropic_myopic(
    data_df: pd.DataFrame,
    delta_time_seconds: float,
    year: int,
) -> None:

    myopic_students = (
        data_df.where(data_df["myopic"])[delta_time_seconds]
        .dropna()
        .to_numpy(dtype=np.float64)
    )
    emmetropic_students = (
        data_df.where(~data_df["myopic"])[delta_time_seconds]
        .dropna()
        .to_numpy(dtype=np.float64)
    )

    result = sm.ks_2sampled(myopic_students, emmetropic_students)

    print(
        "Difference myopic and emmetropic. year: {}, result {}, myopic: {}, emmetropic: {}".format(
            year, result, len(myopic_students), len(emmetropic_students)
        )
    )

    # result = sm.ks_2sampled(myopic_students, emmetropic_students, alternative="greater")
    # print(
    #    "Difference myopic and emmetropic. year: {}, result {}, myopic: {}, emmetropic: {}".format(
    #        year, result, len(myopic_students), len(emmetropic_students)
    #    )
    # )

    # data = {"emmetropic":emmetropic_students, "mopic":myopic_students}
    # for key in data.keys():
    #    sm.qq_plot_for_data(
    #        data[key], "./"+str(key)+"" + IMAGE_FORMAT
    #    )



# this only works for LIDAR data
def generate_statistics_comparison(
    yearToSubjectFileContainers: dict,
    app_settings: ap.AppSettings,
    student_settings: student.StudentSettings,
) -> None:
    validations.checkAtLeastOneElement(yearToSubjectFileContainers)

    acfInverseDataDfDict = __get_and_prepare_dataframes_for_statistical_analyses(
        yearToSubjectFileContainers, student_settings
    )

    for year in acfInverseDataDfDict.keys():
        df = acfInverseDataDfDict[year]
        if student_settings.merge_same_student_recordings:
            df = df.groupby(by=[COLUMN_STUDENT_INDEX, COLUMN_IS_LAB_RECORDING], as_index=False).mean(numeric_only=False)
            df["myopic"] = df["myopic"].astype(bool)

        df.set_index(COLUMN_STUDENT_INDEX, inplace=True)
        # XXX could be changed to two separate columns
        # XXX plotly has to be adapted color=year, marker=is_lab_recording
        df[plot.PLOT_GROUP_NAME] = np.where(
            df[COLUMN_IS_LAB_RECORDING],
            plot.naming.get(str(year) + "_lab"),
            plot.naming.get(str(year)),
        )

        df["student_index_label"] = list(
            map(
                lambda x: plot.naming_short.get(str(year)) + "_" + str(x),
                df.index.values,
            )
        )

        df["theta"] = (
            df[columns.HEADER_REFRACTION_ERROR_RIGHT_EYE].astype(str)
            + ","
            + df[columns.HEADER_REFRACTION_ERROR_LEFT_EYE].astype(str)
        )

        acfInverseDataDfDict[year] = df


    data = pd.concat([acfInverseDataDfDict[2018], acfInverseDataDfDict[2019]], ignore_index=True)

    validations.createDirIfNotExists(app_settings.commonOutputPath)
    # https://plotly.com/python/polar-chart/
    plot.scatter_plot_slop_intersection_grouped_students(
        data,
        os.path.join(
            app_settings.commonOutputPath,
            const.SUB_FOLDER_THESIS,
            "(2)_initial_slope-comparison-grouped-students-acf_inverse.{}".format(
                plot.IMAGE_FORMAT
            ),
        ),
        #r"$m^{-1}$"  # latex syntax
    )

    plot.scatter_plot_slop_by_refraction_error(
        data,
        os.path.join(
            app_settings.commonOutputPath,
            const.SUB_FOLDER_THESIS,
            "(3)_initial_slope-comparison-refraction_error-acf_inverse.{}".format(
                plot.IMAGE_FORMAT
            ),
        ),
    )

    plot.scatter_polar_plot_slope(
        data,
        os.path.join(
            app_settings.commonOutputPath,
            const.SUB_FOLDER_THESIS,
            "(4)_initial_slope-comparison-polar_refraction_error-acf_inverse.{}".format(
                plot.IMAGE_FORMAT
            ),
        ),
        r"$ACF\ inverse\ distance\ m^{-1}$",
    )


def is_lab_recording(year, data_df, student_settings: student.StudentSettings):
    """
    Get a list which student recording was recorded in a LAB
    Regex to match UUID in student_id

    :return: List of true false values for student
    """
    return list(
        map(
            lambda student_id: student_settings.is_lab_recording(
                year, re.search("\\[(.*)\\]", student_id).group(1)
            ),
            data_df.index.to_list(),
        ),
    )


def strip_inf_values(data_frame: pd.DataFrame) -> pd.DataFrame:
    return data_frame.replace([np.inf], np.nan).dropna()


#
def test_for_autocorrelation(
    yearToSubjectFileContainers: dict, ljungbox_test: bool = True
):
    validations.checkAtLeastOneElement(yearToSubjectFileContainers)

    acfInverseDataDict = {}
    for year, containers in yearToSubjectFileContainers.items():
        validations.checkAtLeastOneElement(containers)
        start = timer()

        subject_lidar_containers = np.array(
            [sl.get_lidar_container(c) for c in containers]
        )
        print("Loading lidar data took:", timedelta(seconds=timer() - start))

        durbin_statistics = np.array(
            [
                [
                    container.id,
                    durbin_watson(
                        strip_inf_values(container.get_filtered_data_container())[
                            container.data_column_name
                        ]
                    ),
                ]
                for container in subject_lidar_containers
            ]
        )
        acfInverseDataDict[year] = durbin_statistics.T[1]

        if ljungbox_test:
            # move to unit test ?
            for container in subject_lidar_containers:
                sm.autocorr_ljungbox_assert_test(
                    strip_inf_values(container.get_filtered_data_container())[
                        container.data_column_name
                    ].to_numpy(),
                )

        acfInverseDataDict[year] = durbin_statistics.T[1]

    df_durbin_year = pd.concat(
        [
            pd.DataFrame(
                acfInverseDataDict[year], columns=[year], dtype=np.float64
            ).describe()
            for year in acfInverseDataDict.keys()
        ],
        axis=1,
    )
    s = df_durbin_year.style.format(precision=3)
    print(s.to_latex())


def generate_data_and_statistics(
    subjectFileContainers: [SubjectFileContainer], app_settings: ap.AppSettings
) -> None:
    if app_settings.generateLidarStatistics:
        sl.process_lidar_data(subjectFileContainers)
        sl.process_lidar_data_acf(subjectFileContainers)
    if app_settings.generateVergenceStatStatistics:
        sv.process_vergence_data_basic(subjectFileContainers)
    if False:
        sl.filter_lidar_data_for_simone(subjectFileContainers)
