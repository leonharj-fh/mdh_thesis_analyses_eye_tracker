import numpy as np
import pandas as pd
import os


def checkIndexUnique(dataframe: pd.DataFrame):
    if not dataframe.index.is_unique:
        series = pd.value_counts(dataframe.index.values.tolist())
        seriesWithDuplicateIndixes = series.where(series > 1).dropna()
        # TODO remove duplicate indexes.
        for i in seriesWithDuplicateIndixes.index.tolist():
            values = dataframe.loc[i]
        print("Has a duplicate index:", seriesWithDuplicateIndixes.head())


def check_is_dataframe(object):
    checkNotNone(object)
    assert isinstance(object, pd.DataFrame)
    return object


def none_or_check(element, function):
    return function(element) if element is not None else element


def checkAtLeastOneElement(element):
    checkNotNone(element)
    assert len(element) > 0
    return element


def checkNotNone(element):
    assert element is not None
    return element


def checkStringNumeric(element):
    checkNotNone(element)
    assert element.isnumeric()
    return element


def checkStringNotEmpty(element):
    checkNotNone(element)
    assert element.strip()
    return element


def checkDataFrameContainsColumn(df, columnName):
    checkNotNone(df)
    checkStringNotEmpty(columnName)
    assert columnName in df.columns
    return df


def checkValueGreaterEqualsZero(value: int) -> int:
    assert value >= 0
    return value


def millisToHumanReadableFormat(millis: int) -> str:
    x = millis / 1000
    seconds = int(x % 60)
    x /= 60
    minutes = int(x % 60)
    x /= 60
    hours = int(x % 24)

    return "{:02d}h:{:02d}m:{:02d}s".format(hours, minutes, seconds)


def number_element_less_to_value(array, value):
    array = np.asarray(array)
    return (array < value).sum()


def calculateFirstSlope(deltaTime: float, c1: float, c0: float) -> float:
    # = deltaTime / (1 - y1/y0)
    return deltaTime / (1 - c1 / c0)


def calculate(x: float, c1: float, c0: float) -> float:
    # = deltaTime / (1 - y1/y0)
    return 1 - ((1 - c1) / 0.1) * x


def createDirIfNotExists(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)


def checkContainsNoNotNanValues(df: pd.DataFrame) -> pd.DataFrame:
    assert not df.isnull().values.any()
    return df


def checkContainsNoNotNanNumpyValues(a: []) -> []:
    checkNotNone(a)
    assert not np.isnan(a).any()
    return a

def checkFileExists(file_with_path):
    checkNotNone(file_with_path)
    assert os.path.exists(file_with_path)
    return file_with_path