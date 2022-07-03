import numpy as np
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

from statsmodels.tsa.stattools import acf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.diagnostic import lilliefors
from util import validations
from util import constants as const


def multiply_lag_shift(values: np.ndarray, lag: int) -> np.ndarray:
    if lag == 0:
        return values

    # shift values by lag
    y = values[lag:]
    # fill up with '1' to get same array size
    y_copy = np.append(np.copy(y), np.ones(lag))
    # set all values to 1 which are not infinity
    y_copy = np.where(y_copy != np.inf, 1, np.inf)
    # multiply both array
    return np.multiply(values, y_copy)


def replaceZeroValues(
    values: np.ndarray, replace_by: np.float64 = const.ACF_REPLACE_ZERO_VALUES
):
    return np.where(values == 0.0, replace_by, values)


def acf_pairwise_calculation(values: np.ndarray, lag: int) -> float:
    """

    :param values:
    :param lag:
    :return:
    """
    validations.checkAtLeastOneElement(values)
    validations.checkValueGreaterEqualsZero(lag)
    # start = timer()
    validations.checkContainsNoNotNanNumpyValues(values)
    values = values.flatten()

    # added to avoid multiplication of infinity with 0. resulted in warn.
    values = replaceZeroValues(values)
    shifted_values = multiply_lag_shift(values.flatten(), lag)
    # TODO mean should only be the time series values???
    mean = np.mean(shifted_values[np.isfinite(shifted_values)])
    values_mean = np.copy(values) - mean

    # shift elements to calculate ACF for lag n
    values_mean_y1 = values_mean[: (len(values_mean) - lag)]
    values_mean_y2 = values_mean[lag:]

    products = np.multiply(values_mean_y1, values_mean_y2)
    # strip all infinity values and calculate mean
    lag_value = np.mean(products[np.isfinite(products)])

    assert np.isfinite([lag_value])

    # calculate the divisor mean((y-mean(y))^2)
    # shifted_values can be used here
    shifted_values_0 = shifted_values - mean
    products = np.multiply(shifted_values_0, shifted_values_0)
    # remove all infinity value and calculate mean
    zero_lag_value = np.mean(products[np.isfinite(products)])

    assert np.isfinite([zero_lag_value])
    # print("Finished lag", lag, timedelta(seconds=timer() - start))
    acf_value = lag_value / zero_lag_value

    if acf_value > 1 or acf_value < -1:
        print("WARN: ACF acf_value lower or greater 1", acf_value)
    return acf_value


@DeprecationWarning
def acf_pairwise_invalid(values: np.ndarray, lag: int) -> np.ndarray:
    """
    Am einfachsten und stabilsten ist es wohl, wenn Du aus einer Datenreihe (z.B. valider Messpaare) zuerst den Mittelwert
    ausrechnest, dann diesen Mittelwert von jeder Messung subtrahierst, und dann die Paarkorrelation davon ausrechnest.
    Dies als Alternative zum Vorgehen, wo Du die Paarkorrelation zuerst ausrechnest und danach das Quadrat des Mittelwertes subtrahierst.
    Also, vereinfacht, mean[(x_i-mean[x])^2] statt mean[x_i^2] - mean[x]^2.

    :param values:
    :param lag:
    :return:
    """
    validations.checkValueGreaterEqualsZero(lag)
    validations.checkAtLeastOneElement(values)

    # Slice the relevant sub-series based on the lag
    # y1 = values[: (len(values) - lag)]

    # added_values = np.add(y1, y2)
    # mean = np.mean(added_values[np.isfinite(added_values)])

    shifted_values = multiply_lag_shift(values, lag)

    mean = np.mean(shifted_values[np.isfinite(shifted_values)])
    # subtract mean of each element

    # values_mean = np.copy(shifted_values) - mean
    values_mean = np.copy(values) - mean
    # shift elements to calculate ACF for lag
    values_mean_y1 = values_mean[: (len(values_mean) - lag)]
    values_mean_y2 = values_mean[lag:]

    print(mean)
    print(pd.DataFrame(values_mean[np.isfinite(values_mean)]).describe())
    # assert len(values_mean[np.isfinite(values_mean)]) == len(
    #    values[np.isfinite(values)]
    # )

    # added to avoid multiplication of infinity with 0. resulted in warn.
    values_mean_y1 = np.where(
        values_mean_y1 == 0.0, const.ACF_REPLACE_ZERO_VALUES, values_mean_y1
    )
    values_mean_y2 = np.where(
        values_mean_y2 == 0.0, const.ACF_REPLACE_ZERO_VALUES, values_mean_y2
    )

    # if lag == 119:
    #    print(np.mean(products[np.isfinite(products)]))
    #    #for i in range(len(values_mean_y2)):
    #    #     products = np.mean(np.multiply(values_mean_y1, values_mean_y2))
    #
    products = np.multiply(values_mean_y1, values_mean_y2)
    print(pd.DataFrame(products[np.isfinite(products)]).describe())
    correlation_value = np.mean(products[np.isfinite(products)])
    print(correlation_value)
    print(np.sum(products[np.isfinite(products)]))
    # acf_data_to_debug = pd.concat(
    #    [
    #        pd.DataFrame(shifted_values[np.isfinite(shifted_values)]),
    #        pd.DataFrame(np.append(shifted_values[np.isfinite(shifted_values)] - mean, np.array(mean))),
    #        pd.DataFrame(products[np.isfinite(products)]),
    #    ],
    #    axis=1,
    # )

    # acf_data_to_debug.to_csv("acf_data_{}.csv".format(lag))

    return np.asarray(validations.checkContainsNoNotNanNumpyValues(correlation_value))


@DeprecationWarning
def acf_by_hand_experimental(values: np.ndarray, lag: int) -> float:
    # Slice the relevant subseries based on the lag

    shifted_values = multiply_lag_shift(values, lag)

    shifted_values_finite = shifted_values[np.isfinite(shifted_values)]
    # subtract mean of each element
    values_mean = np.copy(values) - np.mean(shifted_values_finite)
    # shift elements to calculate ACF for lag
    values_mean_y1 = values_mean[: (len(values_mean) - lag)]
    values_mean_y2 = values_mean[lag:]

    products = np.multiply(values_mean_y1, values_mean_y2)
    sum_product = np.sum(products[np.isfinite(products)])

    return np.true_divide(
        sum_product,
        ((len(shifted_values_finite) - lag) * np.var(shifted_values_finite)),
    )


# START ################ acf implementations by the following overflow comment

# https://stackoverflow.com/questions/36038927/whats-the-difference-between-pandas-acf-and-statsmodel-acf


def acf_by_hand_original(x, lag: int) -> float:
    # Slice the relevant subseries based on the lag
    y1 = x[: (len(x) - lag)]
    y2 = x[lag:]
    # Subtract the mean of the whole series x to calculate Cov
    sum_product = np.sum((y1 - np.mean(x)) * (y2 - np.mean(x)))
    # Normalize with var of whole series

    # sum_product / np.var[lag:(len(x) - lag)] ???
    return sum_product / ((len(x) - lag) * np.var(x))


def autocorr_by_hand(x, lag):
    # Slice the relevant subseries based on the lag
    y1 = x[: (len(x) - lag)]
    y2 = x[lag:]
    # Subtract the subseries means
    sum_product = np.sum((y1 - np.mean(y1)) * (y2 - np.mean(y2)))
    # Normalize with the subseries stds
    return sum_product / ((len(x) - lag) * np.std(y1) * np.std(y2))


# END ################ acf implementations by the following overflow comment


def acf_by_statistic_library(data, lags: int):
    return acf(data, nlags=lags)


def qq_plot_for_data(data: np.array, file_name) -> None:
    fig = sm.qqplot(data, line="q")

    plt.savefig(file_name)
    plt.close(fig)


def autocorr_ljungbox_assert_test(data: np.ndarray, alpha: float = 0.001):
    value = acorr_ljungbox(data, lags=[x + 1 for x in range(10)], return_df=True)
    assert not value[value["lb_pvalue"] >= alpha].values.any()


# START =============== manual Kolmogorow-Smirnow-Test


def __cdf(sample: np.ndarray, x, sort=False):
    # Sorts the sample, if unsorted
    if sort:
        sample.sort()
    # Counts how many observations are below x
    cdf = sum(sample <= x)
    # Divides by the total number of observations
    cdf = cdf / len(sample)
    return cdf


def ks_2samp_manuel(sample1: np.ndarray, sample2: np.ndarray) -> dict:
    validations.checkAtLeastOneElement(sample1)
    validations.checkAtLeastOneElement(sample2)

    # Gets all observations
    observations = np.concatenate((sample1, sample2))
    observations.sort()
    # Sorts the samples
    sample1.sort()
    sample2.sort()
    # Evaluates the KS statistic
    D_ks = []  # KS Statistic list
    for x in observations:
        cdf_sample1 = __cdf(sample=sample1, x=x)
        cdf_sample2 = __cdf(sample=sample2, x=x)
        D_ks.append(abs(cdf_sample1 - cdf_sample2))
    ks_stat = max(D_ks)
    # Calculates the P-Value based on the two-sided test
    # The P-Value comes from the KS Distribution Survival Function (SF = 1-CDF)
    m, n = float(len(sample1)), float(len(sample2))
    en = m * n / (m + n)
    p_value = stats.kstwo.sf(ks_stat, np.round(en))
    return {"ks_stat": ks_stat, "p_value": p_value}


# END =============== manual Kolmogorow-Smirnow-Test


def ks_2sampled(data1: np.ndarray, data2: np.ndarray, alternative: str="two-sided"):
    validations.checkNotNone(data1)
    validations.checkNotNone(data2)
    return stats.ks_2samp(data1, data2, alternative=alternative)


def lilliefors_test_norm(data: np.ndarray):
    return lilliefors(data, dist="norm")

def shapiro_wilk_test(data: np.ndarray):
    return stats.shapiro(data)

def normal_test(data: np.ndarray):
    return stats.normaltest(data)
