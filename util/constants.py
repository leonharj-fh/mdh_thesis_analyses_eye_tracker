# constant value of maximum distance sensor measurements
import numpy as np

INVALID_SENSOR_DISTANCE = 8000  # unit millimeter
MAX_SENSOR_DISTANCE = 1800  # unit millimeter
NEAR_SIGHT_DISTANCE = 300  # unit millimeter
NUMBER_LAGS = 1000
MIN_SENSOR_DISTANCE_100 = 100  # unit millimeter
MAX_DISTANCE_UPPER_LIMIT = float('inf')  # unit millimeter

ACF_REPLACE_ZERO_VALUES = np.reciprocal(np.power(10, 24), dtype=np.float64)



NUMBER_LAGS_IN_TIME = 15  # seconds
VIDEO_FPS = 60

VERGENCE_DELTA_TIME = 0.016  # 1000ms / VIDEO_FPS
LIDAR_DELTA_TIME = 0.1
ACF_LIDAR_LEGS = 151
# DISTANCE_SENSOR_RECORDING_INTERVAL = 100  # at which interval we get distance sensor records (unit ms)

DISTANCE_SENSOR_FPS = 10
# (1000 ms / interval (in ms) * seconds
DISTANCE_SENSOR_15_SECONDS = 15

FILTER_VERGENCE_ANGLE_UPPER_END = 1.0  # unit radian
FILTER_VERGENCE_ANGLE_LOWER_END = 0.005  # unit radian

IMAGES_FOLDER = "images"
LIDAR_PLOT_FOLDER = "lidar"
VERGENCE_PLOT_FOLDER = "vergence"
SUB_FOLDER_STATISTICS = "statistics"
SUB_FOLDER_FILTERED_DATE = "filteredData"
SUB_FOLDER_THESIS = "thesis"


