import plotly.graph_objects as go
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
import numpy as np
import plotly.io as pio
from plotly.subplots import make_subplots
from util import validations, csv_columns

PLOT_DEFAULT_WIDTH = 700
PLOT_DEFAULT_HEIGHT = 500

IMAGE_FORMAT = "svg"
pio.kaleido.scope.default_format = IMAGE_FORMAT

ACF_SHIFTED_VERSION_MS = [0.2, 1, 3]  # equals to 0.2s, 1s, and 3 seconds

naming_short = {
    "2018": "Int.",
    "2019": "local",
}

naming = {
    "2018": "international",
    "2018_lab": "international_lab",
    "2019": "local",
    "2019_lab": "local_lab",
}

symbols_map = {
    naming.get("2018"): "circle-open",
    naming.get("2018_lab"): "cross-open",
    naming.get("2019"): "circle-open",
    naming.get("2019_lab"): "cross-open",
}
# blue for the 2018 international school data and red for the 2019 local school data
color_map = {
    naming.get("2018"): "blue",
    naming.get("2018_lab"): "blue",
    naming.get("2019"): "red",
    naming.get("2019_lab"): "red",
}

PLOT_GROUP_NAME = "School group"


def plotAcfByArray(
    series: pd.Series,
    title: str,
    file_path: str,
    yaxis_title="ACF",
    slope_array=None,
    slope_time: float = None,
) -> None:
    validations.checkNotNone(series)
    validations.checkStringNotEmpty(title)
    validations.checkStringNotEmpty(file_path)

    layout = go.Layout(
        autosize=True
    )  # , width=PLOT_DEFAULT_WIDTH, height=PLOT_DEFAULT_HEIGHT)

    fig = go.Figure(layout=layout)

    if slope_array is not None:
        validations.checkNotNone(slope_time)
        fig.add_scatter(
            x=series.index,
            y=slope_array,
            name="x-axis intercept first lag {:.2f}s".format(slope_time),
            mode="lines",
            line_color="rgba(50, 56, 66, 0.8)",
        )
    # XXX hack this is done for generation a plot for the thesis in the context of explaining autocorrelation
    # XXX hack see method plot_lidar_distance
    colorSeries = pd.Series(["#1f77b4" if i not in ACF_SHIFTED_VERSION_MS else "red" for i in series.index])
    markerSize = pd.Series([6 if i not in ACF_SHIFTED_VERSION_MS else 6 for i in series.index])

    fig.add_scatter(
        x=series.index,
        y=series,
        mode="markers",
        name="ACF",
        marker_color=colorSeries,
        marker_size=markerSize,
    )

    fig.update_layout(
        title=title,
        xaxis_title=r"$T[s]$",
        yaxis_title=yaxis_title,
        font=dict(
            size=12,
        ),
    )

    # add legend inside of plot
    fig.update_layout(
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=1,
            font=dict(
                # XXX this is a hack for latex (overleaf), since latex doesn't display the font size of the legend correctly
                # XXX this "fixes" the display issue of the legend in latex
                size=16,
            ),
        )
    )

    fig.write_image(file_path)
    plt.close("all")


def plot_lidar_distance_thesis_plot(series: pd.Series, numberElements: int, file_path: str) -> None:
    """
    Method is only for explaining autocorrelation in the thesis
    """
    validations.checkNotNone(series)
    validations.checkValueGreaterEqualsZero(numberElements)
    assert numberElements == 80  # check for a hardcoded value

    subplot_title = "Signal with a delayed copy of {}s"
    fig = make_subplots(
        rows=3,
        cols=1,
        subplot_titles=(
            subplot_title.format(ACF_SHIFTED_VERSION_MS[0]),
            subplot_title.format(ACF_SHIFTED_VERSION_MS[1]),
            subplot_title.format(ACF_SHIFTED_VERSION_MS[2]),
        ),
    )

    i = 1
    for elements in ACF_SHIFTED_VERSION_MS:
        showlegend = True if i == 1 else False

        take_elements = int(numberElements - (elements * 10))

        first_data_points = series.head(int(take_elements))
        fig.append_trace(
            go.Scatter(
                x=first_data_points.index,
                y=first_data_points,
                name="original signal",
                legendgroup="original signal",
                mode="lines",
                line=dict(color="blue"),
                showlegend=showlegend,
            ),
            row=i,
            col=1,
        )

        last_data_points = pd.Series(
            series.tail(take_elements).to_numpy(),
            index=first_data_points.index,
            dtype=np.float64,
        )

        fig.append_trace(
            go.Scatter(
                x=last_data_points.index,
                y=last_data_points,
                name="delayed copy",
                legendgroup="delayed copy",
                mode="lines",
                line=dict(color="red"),
                showlegend=showlegend,
            ),
            row=i,
            col=1,
        )
        i += 1

    fig.update_layout(
        height=960,
        width=1280,
        #title_text="How data is compared for ACF",
    )

    ytitle = r"$m^{-1}$"  # latex syntax
    xtitle = r"$T[s]$"  # latex syntax
    for i in range(1, 4):
        fig.update_yaxes(title_text=ytitle, row=i, showgrid=False, col=1)
        fig.update_xaxes(title_text=xtitle, row=i, showgrid=False, col=1)

    fig.write_image(file_path)
    # fig.savefig(file_path)
    plt.close("all")


def plot_lidar_distance_histogram(df: pd.DataFrame, file_path: str) -> None:
    validations.checkNotNone(df)

    fig = px.histogram(
        df,
        x=csv_columns.SENSOR_COLUMN_DISTANCE,
        title="LIDAR inverse distance - histogram",
        log_y=True,
        width=1024,
        height=768,
    )

    fig.update_layout(
        xaxis_title=r"$distance\ m^{-1}$", yaxis_title="count (logarithmic scale)"
    )

    fig.write_image(file_path)
    # fig.savefig(file_path)
    plt.close("all")


def plotVerganceAngleToReciprocalDistance(
    data_df: pd.DataFrame, x_column: str, y_column: str, title: str, file_path: str
) -> None:
    # fig = plt.figure()
    fig = data_df.plot(
        figsize=(19.8, 12.8),
        x=x_column,
        y=y_column,
        kind="scatter",
        title=title,
        xlabel="inverse gaze vector distance",
        ylabel="radiant",
    ).get_figure()

    fig.savefig(file_path)
    plt.close(fig)


def scatter_plot_slop_by_refraction_error(
    data: [], file_name: str
) -> None:
    validations.checkNotNone(file_name)
    validations.checkNotNone(data)

    fig = px.scatter(
        data,
        x="refract_error",
        y="slopeValue",
        color=PLOT_GROUP_NAME,
        symbol=PLOT_GROUP_NAME,
        symbol_map=symbols_map,
        color_discrete_map=color_map,
        width=PLOT_DEFAULT_WIDTH,
        height=PLOT_DEFAULT_HEIGHT,
        title="Initial slope comparison international and local school"
        + "<br><sup>The closer to zero on the y-axis the lower the gaze dynamic</sup>",
    )

    fig.update_layout(
        xaxis_title="mean refraction error of [R,L] eye",
        yaxis_title=r"$s^{-1}$",
        font=dict(
            size=12,
        ),
    )

    fig.write_image(file_name)


# https://stackoverflow.com/questions/65946833/plotly-how-to-set-marker-symbol-shapes-for-multiple-traces-using-plotly-express
def scatter_plot_slop_intersection_grouped_students(
    data: pd.DataFrame, file_name: str
):
    validations.checkNotNone(file_name)
    validations.checkNotNone(data)

    data = data.sort_values(by=[PLOT_GROUP_NAME, "slopeValue"])

    fig = px.scatter(
        data,
        x="student_index_label",
        y="slopeValue",
        color=PLOT_GROUP_NAME,
        symbol=PLOT_GROUP_NAME,
        symbol_map=symbols_map,
        color_discrete_map=color_map,
        width=PLOT_DEFAULT_WIDTH,
        height=PLOT_DEFAULT_HEIGHT,
        title="Initial slope comparison international and local school"
        + "<br><sup>The closer to zero the lower the gaze dynamic</sup>",
    )

    fig.update_layout(
        xaxis_title="Student of the respective school",
        # latex syntax  r"ACF inverse distance $m^{-1}$",
        yaxis_title=r"$s^{-1}$",
        font=dict(
            size=12,
        ),
    )

    fig.write_image(file_name)


def scatter_polar_plot_slope(data: dict, file_name: str, title_prefix: str):
    validations.checkNotNone(file_name)
    validations.checkNotNone(data)

    fig = px.scatter_polar(
        data,
        r="slopeValue",
        theta="theta",
        color=PLOT_GROUP_NAME,
        symbol=PLOT_GROUP_NAME,
        symbol_map=symbols_map,
        color_discrete_map=color_map,
        width=PLOT_DEFAULT_WIDTH,
        height=PLOT_DEFAULT_HEIGHT,
    )

    fig.update_layout(
        title=title_prefix
        + " -  angular coordinate label: refraction error [R, L] eye "
        + "<br><sup>The closer to the center the lower the gaze dynamic</sup>",
        font=dict(
            size=12,
        ),
    )

    fig.update_layout(
        # XXX overthink range definition
        polar=dict(
            radialaxis_range=[0, -2.5],
        ),
    )

    fig.write_image(file_name)


def cdf_plot(acf_data_schools: dict, file_name: str) -> None:
    validations.checkNotNone(acf_data_schools)
    validations.checkStringNotEmpty(file_name)

    column_name = "initial slope"
    df_2018 = pd.DataFrame(
        acf_data_schools[2018], columns=[column_name], dtype=np.float
    )
    df_2018[PLOT_GROUP_NAME] = naming.get("2018")

    df_2019 = pd.DataFrame(
        acf_data_schools[2019], columns=[column_name], dtype=np.float
    )
    df_2019[PLOT_GROUP_NAME] = naming.get("2019")

    data = pd.concat([df_2018, df_2019], ignore_index=True)

    fig = px.ecdf(
        data.convert_dtypes(), x=column_name, color=PLOT_GROUP_NAME, #title="Cu"
    )
    fig.update_xaxes(title_text='slope value')
    fig.write_image(file_name, engine="kaleido")
