import os

import pandas as pd
from util import csv_columns as columns
from util import validations
import numpy as np
import time
from model import Subject
from model.SubjectFileContainer import SubjectFileContainer

TEMP_FILE = "tempfile.csv"
DEFAULT_CHUNK_SIZE = 10000


def __initFileStructure(subject: Subject) -> SubjectFileContainer:
    fileNameSensorData = os.path.join(
        subject.inputPath, subject.id, subject.sensorDataFilename
    )
    fileNameEyeRecordsData = os.path.join(
        subject.inputPath, subject.id, subject.eyetrackingDataFilename
    )

    subjectFolder = os.path.join(subject.outputPath, subject.id)

    tempfile = os.path.join(subjectFolder, TEMP_FILE)
    fileNameEyeDataWithReference = os.path.join(
        subjectFolder, subject.id + "_eyetrackingWithReference.csv"
    )
    fileNameSensorDataRbg = os.path.join(subjectFolder, subject.id + "_sensorsRbg.csv")
    fileNameSensorDataLight = os.path.join(
        subjectFolder, subject.id + "_sensorsLight.csv"
    )
    fileNameSensorDataDistance = os.path.join(
        subjectFolder, subject.id + "_sensorsDistance.csv"
    )

    return SubjectFileContainer(
        subject.id,
        fileNameSensorData,
        fileNameEyeRecordsData,
        subjectFolder,
        tempfile,
        fileNameEyeDataWithReference,
        fileNameSensorDataRbg,
        fileNameSensorDataLight,
        fileNameSensorDataDistance,
        subject.outputPath,
    )


def prepareDataForEvaluation(
        subject: Subject, initialize=False
) -> SubjectFileContainer:
    validations.checkNotNone(subject)

    fileContainer = __initFileStructure(subject)
    if __doesAnyOutputFileAlreadyExist(fileContainer):
        if not initialize:
            #print("At least one output file already exists. Returning paths")
            return fileContainer
        else:
            # XXX not tested
            print("Recreating files.")
            raise Exception("not tested")

    print("Started to prepare data for subject", subject.id)
    __deleteFilesIfExist(fileContainer)

    # folder = os.path.join(pathToFiles, subjectFolder)
    if not os.path.exists(fileContainer.subjectFolder):
        os.makedirs(fileContainer.subjectFolder)

    sensorGroupDistance, sensorGroupLight, sensorGroupRgb = __processSensorData(
        fileContainer
    )

    __processEyeTrackingData(
        fileContainer, sensorGroupDistance, sensorGroupLight, sensorGroupRgb
    )

    print("Finished creating references")

    return fileContainer


def __processSensorData(fileContainer: SubjectFileContainer):
    # group by sensorNumber
    dfSensorGroup = __readSensorData(fileContainer).groupby(
        columns.SENSOR_NUMBER_COLUMN
    )
    sensorGroupRgb = (
        dfSensorGroup.get_group(0)[columns.SENSOR_COLUMN_NAMES]
            .set_axis(columns.HEADER_SENSOR_DATA0_RGB, axis=1)
            .copy()
            .reset_index()
    )
    sensorGroupLight = (
        dfSensorGroup.get_group(1)[
            columns.SENSOR_COLUMN_NAMES[: len(columns.HEADER_SENSOR_DATA1_LIGHT)]
        ]
            .set_axis(columns.HEADER_SENSOR_DATA1_LIGHT, axis=1)
            .copy()
            .reset_index()
    )
    sensorGroupDistance = (
        dfSensorGroup.get_group(2)[
            columns.SENSOR_COLUMN_NAMES[: len(columns.HEADER_SENSOR_DATA2_DISTANCE)]
        ]
            .set_axis(columns.HEADER_SENSOR_DATA2_DISTANCE, axis=1)
            .copy()
            .reset_index()
    )
    for dfElement in [
        sensorGroupRgb,
        sensorGroupLight,
        sensorGroupDistance,
    ]:
        dfElement.rename(
            columns={columns.TIMESTAMP_COLUMN: columns.TIMESTAMP_ORIGINAL_COLUMN},
            inplace=True,
            errors="raise",
        )
        dfElement[columns.TIMESTAMP_COLUMN] = dfElement[
            columns.TIMESTAMP_ORIGINAL_COLUMN
        ].astype(int)

    sensorGroupDistance = __setDistanceSensorIndex(sensorGroupDistance)

    sensorGroupRgb.to_csv(
        fileContainer.pathToSensorDataRbg, index=False, encoding="utf-8"
    )
    sensorGroupLight.to_csv(
        fileContainer.pathToSensorDataLight, index=False, encoding="utf-8"
    )
    sensorGroupDistance.to_csv(
        fileContainer.pathToSensorDataDistance, index=True, encoding="utf-8"
    )
    return sensorGroupDistance, sensorGroupLight, sensorGroupRgb


def __setDistanceSensorIndex(df: pd.DataFrame):
    meanValue = 100

    z = 0
    indexList = None

    values = df[columns.TIMESTAMP_COLUMN].to_numpy()
    with np.nditer(values, flags=["f_index"], op_flags=["readonly"]) as it:
        for timeStamp in it:
            if it.index == 0:
                indexList = [z]
                continue

            else:
                value = timeStamp - values[it.index - 1]

                diff = np.divide(
                    np.round(
                        np.divide(
                            value,
                            meanValue,
                        ),
                        decimals=1,
                    ),
                    # XXX divide by 10 to have ms in seconds (error prone)
                    10,
                )
                z = np.round(np.add(z, diff), 1)
                indexList.append(z)

    df["index"] = pd.Series(np.array(indexList) * 10, dtype=np.int)
    # df.set_index("index", inplace=True)

    indexRange = np.arange(indexList[0], indexList[len(indexList) - 1] * 10, 1)
    dfIndex = pd.DataFrame(index=indexRange, dtype=np.int)

    dfOutput = pd.merge(
        dfIndex, df, how="outer", left_index=True, right_on="index", sort=True
    )
    dfOutput["index"] = dfOutput["index"] / 10
    dfOutput.set_index("index", inplace=True)

    return dfOutput.convert_dtypes()

@DeprecationWarning
def __setDistanceSensorIndex_experimental(df: pd.DataFrame):
    # code is experimental
    meanValue = df[columns.TIMESTAMP_COLUMN].diff().mean()
    # std = df[columns.TIMESTAMP_COLUMN].diff().std()

    lowerEnd = meanValue - (meanValue * 0.3)
    upperEnd = meanValue + (meanValue * 0.3)

    indexList = None

    value_between_entries = 10
    z = 0

    values = df[columns.TIMESTAMP_COLUMN].diff().to_numpy()
    values[0] = 0
    assert len(values[np.isnan(values)]) == 0
    values = np.where((values >= lowerEnd) & (values <= upperEnd), np.inf, values)

    initialTimestamp = df[columns.TIMESTAMP_COLUMN].head(1).values[0]
    with np.nditer(values, flags=["f_index"], op_flags=["readonly"]) as it:
        for entry in it:
            if it.index == 0:
                indexList = [z]
                continue
            elif entry == np.inf:
                if z % value_between_entries != 0:
                    z = round(z, -1)
                z += value_between_entries
                indexList.append(z)
            else:
                correctedValue = np.remainder(entry - initialTimestamp, meanValue)
                if False and correctedValue > lowerEnd and correctedValue < upperEnd:
                    z += value_between_entries
                    indexList.append(z)
                else:
                    delta = entry.item(0)
                    i = value_between_entries
                    added = False
                    while delta > 0:
                        delta -= meanValue
                        i += value_between_entries
                        if delta > lowerEnd and delta < upperEnd:
                            z += i
                            indexList.append(z)
                            added = True
                            break
                    if not added:
                        correctedValue = np.round(
                            np.multiply(
                                np.divide(
                                    entry,
                                    meanValue,
                                ),
                                value_between_entries,
                            ),
                            0,
                        )

                        z += correctedValue
                        indexList.append(z)

    newIndexes = np.array(indexList, dtype=np.float64) / 100

    df["index"] = pd.Series(np.array(newIndexes))
    df.set_index("index")


def __deleteFilesIfExist(fileContainer: SubjectFileContainer) -> None:
    for file in [
        fileContainer.pathToTempfile,
        fileContainer.pathToEyeDataWithReference,
        fileContainer.pathToSensorDataDistance,
        fileContainer.pathToSensorDataRbg,
        fileContainer.pathToSensorDataLight,
    ]:
        if os.path.exists(file):
            os.remove(file)


def __addSensorReferenceColumn(
        reader: pd.io.parsers.readers.TextFileReader,
        sensorDataFrame: pd.DataFrame,
        columnName: str,
        fileName: str,
        chunkSize: int = DEFAULT_CHUNK_SIZE
) -> str:
    """

    :param reader:
    :param sensorDataFrame:
    :param columnName:
    :param fileName:
    :param chunkSize:
    :return:
    """
    print("Started long time operation. Create reference for column:", columnName)

    dataFrameToUpdate = __readDataByChunkSizeAndUpdateColumn(
        chunkSize, columnName, reader
    )
    rowCriteria = dataFrameToUpdate[columnName].isnull()

    sensorDataTimeStamps = np.array(
        sensorDataFrame[columns.TIMESTAMP_COLUMN].dropna().values.tolist()
    )
    countElements = len(sensorDataTimeStamps)

    header = True
    lastChunkWritten = False
    now = time.time()

    with np.nditer(
            sensorDataTimeStamps, flags=["f_index", "refs_ok"], op_flags=["readonly"]
    ) as it:
        for timeStamp in it:
            # calculate mean between two rows to get the closest timestamp reference
            timeStampUpdateCriteria = (
                timeStamp + (sensorDataTimeStamps[it.index + 1] - timeStamp) / 2
                if countElements > (it.index + 1)
                else timeStamp
            )
            # update rows
            rowCriteria = __updateReference(
                dataFrameToUpdate,
                rowCriteria,
                timeStamp,
                timeStampUpdateCriteria,
                columnName,
            )
            # I expect that every row is updated with a reference to fetch the next chunk
            if not rowCriteria.any():
                __writeChunkOfReferenceData(
                    fileName, columnName, dataFrameToUpdate, header
                )
                header = False

                if len(dataFrameToUpdate) < chunkSize:
                    print(
                        "Wrote last chunk size. Fetched data frame size was less than chunksize. Stopping iteration."
                    )
                    lastChunkWritten = True
                    break
                dataFrameToUpdate = __readDataByChunkSizeAndUpdateColumn(
                    chunkSize, columnName, reader
                )
                if dataFrameToUpdate is None or len(dataFrameToUpdate) == 0:
                    print("Data frame chunk returned empty row.")
                    lastChunkWritten = True
                    break

                # update rows
                rowCriteria = __updateReference(
                    dataFrameToUpdate,
                    dataFrameToUpdate[columnName].isnull(),
                    timeStamp,
                    timeStampUpdateCriteria,
                    columnName,
                )

    # only write dataframe if it has not already been written.
    if not lastChunkWritten:
        # update all remaining values which have no value set
        dataFrameToUpdate.loc[
            (dataFrameToUpdate[columnName].isnull()),
            columnName,
        ] = sensorDataTimeStamps[countElements - 1]
        # write data
        __writeChunkOfReferenceData(fileName, columnName, dataFrameToUpdate, header)
    print(
        (
            {
                "method": "adding sensor reference",
                "time": time.time() - now,
                "column": columnName,
            }
        )
    )
    return fileName


def __readDataByChunkSizeAndUpdateColumn(chunkSize, columnName, reader):
    try:
        dataFrameToUpdate = reader.get_chunk(chunkSize)
        # add new column and set with nan (float64)
        dataFrameToUpdate.loc[:, columnName] = np.nan
        return dataFrameToUpdate
    except StopIteration:
        # TODO validate if a workaround for exception exists
        # TODO read number of lines to avoid this exception. chunksize mod number of files != 0
        print("Suppressing StopIteration exception. EOF reached?")
        return None


def __writeChunkOfReferenceData(
        filename: str, columnName: str, dataFrameToUpdate: pd.DataFrame, header: bool
) -> None:
    # convert timestamp reference to integer
    dataFrameToUpdate[columnName] = dataFrameToUpdate[columnName].astype(int)
    # write Data
    dataFrameToUpdate.to_csv(
        filename, header=header, index=False, encoding="utf-8", mode="a"
    )


def __doesAnyOutputFileAlreadyExist(fileContainer: SubjectFileContainer) -> bool:
    return (
            len(
                list(
                    filter(
                        lambda file: not os.path.exists(file),
                        [
                            fileContainer.pathToEyeDataWithReference,
                            fileContainer.pathToSensorDataDistance,
                            fileContainer.pathToSensorDataRbg,
                            fileContainer.pathToSensorDataLight,
                        ],
                    )
                )
            )
            == 0
    )


def __addColumnGazeToZeroDistance(chunk: pd.DataFrame) -> None:
    chunk[
        columns.EYE_TRACKING_DISTANCE_BETWEEN_POINTS
    ] = __calculateDistanceBetweenTwoPoints(
        chunk[columns.EYE_TRACKING_COLUMN_ZERO_POINT_X],
        chunk[columns.EYE_TRACKING_COLUMN_ZERO_POINT_Y],
        chunk[columns.EYE_TRACKING_COLUMN_ZERO_POINT_Z],
        chunk[columns.EYE_TRACKING_COLUMN_GAZE_POINT_X],
        chunk[columns.EYE_TRACKING_COLUMN_GAZE_POINT_Y],
        chunk[columns.EYE_TRACKING_COLUMN_GAZE_POINT_Z],
    )


def __calculateDistanceBetweenTwoPoints(
        x1: float, y1: float, z1: float, x2: float, y2: float, z2: float
) -> float:
    return np.sqrt(np.square(x2 - x1) + np.square(y2 - y1) + np.square(z2 - z1))


def __processEyeTrackingData(
        fileContainer: SubjectFileContainer,
        sensorGroupDistance: pd.DataFrame,
        sensorGroupLight: pd.DataFrame,
        sensorGroupRgb: pd.DataFrame,
) -> None:
    __extendEyeTrackingFileWithColumns(fileContainer)

    # add a timestamp reference for each sensor data.
    for columnName, dataFrame in {
        columns.EYE_TRACKING_COLUMN_DISTANCE_SENSOR_REFERENCE: sensorGroupDistance,
        columns.EYE_TRACKING_COLUMN_RGB_SENSOR_REFERENCE: sensorGroupRgb,
        columns.EYE_TRACKING_COLUMN_LIGHT_REFERENCE: sensorGroupLight,
    }.items():
        with pd.read_csv(
                fileContainer.pathToEyeDataWithReference,
                header=0,
                names=columns.EYE_TRACKING_COLUMNS_DATA_WITH_REFERENCES,
                iterator=True,
                low_memory=False,
        ) as reader:
            __addSensorReferenceColumn(
                reader, dataFrame, columnName, fileContainer.pathToTempfile
            )
        # replace file as we need one for read and write operation
        os.replace(
            fileContainer.pathToTempfile, fileContainer.pathToEyeDataWithReference
        )

    __add_eye_data_index_gaps(fileContainer)


def __add_eye_data_index_gaps(fileContainer: SubjectFileContainer) -> None:
    # XXX could be changed to read chunks as well.
    eye_tracking_df = pd.read_csv(
        fileContainer.pathToEyeDataWithReference,
        header=0,
        names=columns.EYE_TRACKING_COLUMNS_DATA_WITH_REFERENCES,
    )

    index_list = eye_tracking_df.index.to_numpy()
    indexRange = np.arange(index_list[0], index_list[len(index_list) - 1], 1)
    dfIndex = pd.DataFrame(index=indexRange, dtype=np.int)

    dfOutput = pd.merge(
        dfIndex,
        eye_tracking_df,
        how="outer",
        left_index=True,
        right_on=columns.EYE_TRACKING_COLUMN_FRAME_INDEX,
        sort=True,
    )
    # dfOutput.set_index("index", inplace=True)

    dfOutput.convert_dtypes().to_csv(
        fileContainer.pathToTempfile,
        header=True,
        index=False,
        encoding="utf-8",
    )

    os.replace(fileContainer.pathToTempfile, fileContainer.pathToEyeDataWithReference)


def __extendEyeTrackingFileWithColumns(fileContainer: SubjectFileContainer) -> None:
    chunksize = 10 ** 5
    # append the three column
    with pd.read_csv(
            fileContainer.pathToInputEyeRecordsData,
            # XXX header given or not?
            header=None,
            names=columns.EYE_TRACKING_COLUMNS_DATA,
            chunksize=chunksize,
            na_values=["-", "Indeterminate"],
            low_memory=False,
    ) as reader:
        header = True
        for chunk in reader:
            chunk[columns.EYE_TRACKING_REFERENCE_COLUMNS] = pd.DataFrame(
                [[np.nan, np.nan, np.nan]], index=chunk.index, dtype=float
            )
            __addColumnGazeToZeroDistance(chunk)

            chunk.to_csv(
                fileContainer.pathToTempfile,
                header=header,
                index=False,
                encoding="utf-8",
                mode="a",
            )
            header = False
        os.replace(
            fileContainer.pathToTempfile, fileContainer.pathToEyeDataWithReference
        )


def __readSensorData(fileContainer: SubjectFileContainer) -> pd.DataFrame:
    return pd.read_csv(
        fileContainer.pathToInputSensorData,
        header=None,
        names=columns.HEADER_SENSOR_IMPORT_NAMES,
        low_memory=False,
    )


def __updateReference(
        df: pd.DataFrame,
        nullSeries: pd.Series,
        timeStamp: int,
        timeStampCriteria: float,
        columnName: str,
) -> pd.Series:
    df.loc[
        (nullSeries) & (df[columns.TIMESTAMP_COLUMN] < timeStampCriteria),
        columnName,
    ] = timeStamp
    return df[columnName].isnull()
