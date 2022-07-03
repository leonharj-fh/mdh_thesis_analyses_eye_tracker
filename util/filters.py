import pandas as pd
from util import constants as const, csv_columns as columns
from util import validations


def getInvalidVergenceErrorCriteria(df: pd.DataFrame) -> bool:
    return (
            df[columns.EYE_TRACKING_COLUMN_VERGENCE_ERROR]
            > df[columns.EYE_TRACKING_COLUMN_VERGENCE_ANGLE]
    )


def getInvalidNegativVergenceAngle(df: pd.DataFrame) -> bool:
    return df[columns.EYE_TRACKING_COLUMN_VERGENCE_ANGLE] < 0


def getInvalidVergenceAngleLowerEnd(
        df: pd.DataFrame, lowerEquals: int = const.FILTER_VERGENCE_ANGLE_LOWER_END
) -> bool:
    return df[columns.EYE_TRACKING_COLUMN_VERGENCE_ANGLE] <= lowerEquals


def getInvalidVergenceAngleUpperEnd(
        df: pd.DataFrame, greaterEquals: float = const.FILTER_VERGENCE_ANGLE_UPPER_END
) -> bool:
    return df[columns.EYE_TRACKING_COLUMN_VERGENCE_ANGLE] >= greaterEquals


def filterByValidVergence(df: pd.DataFrame) -> pd.DataFrame:
    """
    :param df: Dataframe containing a vergence angle and vergence error column.
    :return: Copy of Pandas dataframe entries where XXX ...
    """
    validations.checkDataFrameContainsColumn(df, columns.EYE_TRACKING_COLUMN_VERGENCE_ANGLE)
    validations.checkDataFrameContainsColumn(df, columns.EYE_TRACKING_COLUMN_VERGENCE_ERROR)

    return df.where(
        ~getInvalidVergenceErrorCriteria(df)
        & ~getInvalidVergenceAngleUpperEnd(df)
        & ~getInvalidVergenceAngleLowerEnd(df)
    )


def getInvalidVergenceAngleNan(df: pd.DataFrame) -> bool:
    return df[columns.EYE_TRACKING_COLUMN_VERGENCE_ANGLE].isna()


def getInvalidLidarDistanceNan(df: pd.DataFrame) -> bool:
    return df[columns.SENSOR_COLUMN_DISTANCE].isna()


def getInvalidLidardDistanceUpperEnd(
        df: pd.DataFrame, maxSupportedDistance: int = const.MAX_SENSOR_DISTANCE
) -> bool:
    return df[columns.SENSOR_COLUMN_DISTANCE] > maxSupportedDistance


def getInvalidLidardDistanceLowerEnd(
        df: pd.DataFrame, minimumUsableDistance: int = const.MIN_SENSOR_DISTANCE_100
) -> bool:
    return df[columns.SENSOR_COLUMN_DISTANCE] < minimumUsableDistance


def getLidardDistanceGreatherZeroValues(df: pd.DataFrame) -> bool:
    return df[columns.SENSOR_COLUMN_DISTANCE] >= 1


def filterByValidDistance(df: pd.DataFrame) -> pd.DataFrame:
    """
    :param df:
    :return:
    """
    validations.checkDataFrameContainsColumn(df, columns.SENSOR_COLUMN_DISTANCE)

    return df.where(
        ~getInvalidLidardDistanceUpperEnd(df) & ~getInvalidLidardDistanceLowerEnd(df)
    )
