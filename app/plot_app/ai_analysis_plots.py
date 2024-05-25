""" This contains PID analysis plots """

from bokeh.io import curdoc
from bokeh.layouts import row, column

import os
import numpy as np
import itertools
import gzip
import requests
import pickle
from plotted_tables import get_heading_html
from bokeh.plotting import figure
from bokeh.palettes import Dark2_5 as palette
from bokeh.palettes import interp_palette
from bokeh.models import (
    Paragraph,
    LinearColorMapper,
    Range1d,
    Spinner,
    CheckboxGroup,
)
from plotting import CustomJSTickFormatter
import pandas as pd
from typing import TypedDict, List

AI_MODEL_ENDPOINT = os.getenv('AI_MODEL_ENDPOINT')

if not AI_MODEL_ENDPOINT:
    raise Exception('Forgot to specify AI_MODEL_ENDPOINT')

default_threshold = 0.3
gradient_palette = interp_palette(["#FFFFFF", "#FFC0CB"], 256)
color_mapper = LinearColorMapper(gradient_palette, low=0, high=1)

# these are the parameters, that were considered while training
params = {
    "vehicle_attitude": ["roll", "pitch", "yaw"],
    "vehicle_attitude_setpoint": ["roll_d", "pitch_d", "yaw_d"],
    "vehicle_local_position": ["x", "y", "z"],
    "vehicle_local_position_setpoint": ["x", "y", "z"],
    "sensor_combined": [
        "accelerometer_m_s2[0]",
        "accelerometer_m_s2[1]",
        "accelerometer_m_s2[2]",
    ],
    "vehicle_magnetometer": [
        "magnetometer_ga[0]",
        "magnetometer_ga[1]",
        "magnetometer_ga[2]",
    ],
}

figures = [
    {
        "title": "Attitude.Pitch",
        "plots": [
            {"dataset": "vehicle_attitude", "key": "pitch", "label": "Pitch"},
            {
                "dataset": "vehicle_attitude_setpoint",
                "key": "pitch_d",
                "label": "Pitch Setpoint",
            },
        ],
    },
    {
        "title": "Attitude.Roll",
        "plots": [
            {"dataset": "vehicle_attitude", "key": "roll", "label": "Roll"},
            {
                "dataset": "vehicle_attitude_setpoint",
                "key": "roll_d",
                "label": "Roll Setpoint",
            },
        ],
    },
    {
        "title": "Attitude.Yaw",
        "plots": [
            {"dataset": "vehicle_attitude", "key": "yaw", "label": "Yaw"},
            {
                "dataset": "vehicle_attitude_setpoint",
                "key": "yaw_d",
                "label": "Yaw Setpoint",
            },
        ],
    },
    {
        "title": "Position.X",
        "plots": [
            {"dataset": "vehicle_local_position", "key": "x", "label": "X"},
            {
                "dataset": "vehicle_local_position_setpoint",
                "key": "x",
                "label": "X Setpoint",
            },
        ],
    },
    {
        "title": "Position.Y",
        "plots": [
            {"dataset": "vehicle_local_position", "key": "y", "label": "Y"},
            {
                "dataset": "vehicle_local_position_setpoint",
                "key": "y",
                "label": "Y Setpoint",
            },
        ],
    },
    {
        "title": "Position.Z",
        "plots": [
            {"dataset": "vehicle_local_position", "key": "z", "label": "Z"},
            {
                "dataset": "vehicle_local_position_setpoint",
                "key": "z",
                "label": "Z Setpoint",
            },
        ],
    },
    {
        "title": "Raw Acceleration",
        "plots": [
            {
                "dataset": "sensor_combined",
                "key": "accelerometer_m_s2[0]",
                "label": "X",
            },
            {
                "dataset": "sensor_combined",
                "key": "accelerometer_m_s2[1]",
                "label": "Y",
            },
            {
                "dataset": "sensor_combined",
                "key": "accelerometer_m_s2[2]",
                "label": "Z",
            },
        ],
    },
    {
        "title": "Magnetometer",
        "plots": [
            {
                "dataset": "vehicle_magnetometer",
                "key": "magnetometer_ga[0]",
                "label": "X",
            },
            {
                "dataset": "vehicle_magnetometer",
                "key": "magnetometer_ga[1]",
                "label": "Y",
            },
            {
                "dataset": "vehicle_magnetometer",
                "key": "magnetometer_ga[2]",
                "label": "Z",
            },
        ],
    },
]


class MissionData(TypedDict):
    dataset: str
    attr: str
    timestamp: np.ndarray[np.int32]
    values: np.ndarray[np.float32]


def extract_mission_mode(ulog) -> List[MissionData]:
    # find largest mission subarray
    # 3 is mission mode
    arr = ulog.get_dataset("vehicle_status").data["nav_state"] == 3
    diff = np.diff(arr.astype(int))
    start = (np.where(diff == 1)[0] + 1).tolist()
    end = (np.where(diff == -1)[0] + 1).tolist()
    if arr[0] == 1:
        start = [0] + start  # start with True
    if arr[-1] == 1:
        end = end + [len(arr)]  # ends with True
    arg = np.subtract(end, start).argmax()
    start_time = ulog.get_dataset("vehicle_status").data["timestamp"][start[arg]]
    end_time = ulog.get_dataset("vehicle_status").data["timestamp"][end[arg] - 1]

    cols = []
    for dataset, attrs in params.items():
        try:
            data = ulog.get_dataset(dataset).data
        except (KeyError, IndexError, ValueError):
            raise KeyError(f"{dataset} not found")
        for attr in attrs:
            values = data.get(attr)
            timestamp = data["timestamp"]
            mission_idx = np.where((timestamp > start_time) & (timestamp < end_time))[0]

            if values is None:
                raise KeyError(f"{attr} not found in {dataset}")
            cols.append(
                {
                    "dataset": dataset,
                    "attr": attr,
                    "timestamp": timestamp[mission_idx],
                    "values": values[mission_idx],
                }
            )

    return cols


def add_custom_tick(model):
    model.xaxis[0].formatter = CustomJSTickFormatter(
        code="""
            //func arguments: ticks, x_range
            // assume us ticks
            var ms = Math.round(tick / 1000);
            var sec = Math.floor(ms / 1000);
            var minutes = Math.floor(sec / 60);
            var hours = Math.floor(minutes / 60);
            ms = ms % 1000;
            sec = sec % 60;
            minutes = minutes % 60;

            function pad(num, size) {
                var s = num+"";
                while (s.length < size) s = "0" + s;
                return s;
            }

            if (hours > 0) {
                var ret_val = hours + ":" + pad(minutes, 2) + ":" + pad(sec,2);
            } else {
                var ret_val = minutes + ":" + pad(sec,2);
            }
            if (x_range.end - x_range.start < 4e6) {
                ret_val = ret_val + "." + pad(ms, 3);
            }
            return ret_val;
        """,
        args={"x_range": model.x_range},
    )
    model.legend.click_policy = "hide"


def compress(C, T, t):
    assert len(T) >= len(t)
    assert len(C) == len(T)

    idx = np.searchsorted(T, t, side="right")

    start = 0
    c = np.zeros_like(t, dtype=np.float32)
    for j, i in enumerate(idx):
        if i > start:
            c[j] = C[start:i].mean()
        else:
            c[j] = C[start if start < len(C) else -1]
        start = i
    return c


def expand(c, t, T):
    assert len(T) >= len(t)
    assert len(c) == len(t)

    idx = np.searchsorted(t, T, side="left")

    C = np.zeros_like(T, dtype=np.float32)
    for j, i in enumerate(idx):
        C[j] = c[i if i < len(c) else -1]
    return C


def align(c, t, t_):
    if len(t) > len(t_):
        return compress(c, t, t_)
    else:
        return expand(c, t, t_)


def align_cols(cols: List[MissionData]) -> None:
    # vehicle_local_position.x is 10 Hz, so this should make
    # all log attributes 10 Hz
    idx = 6  # vehicle_local_position.x
    for col in cols:
        c_ = align(col["values"], col["timestamp"], cols[idx]["timestamp"])
        col["values"] = c_
        col["timestamp"] = cols[idx]["timestamp"]


def cols_to_df(cols: List[MissionData]) -> pd.DataFrame:
    return pd.DataFrame(
        np.vstack([cols[0]["timestamp"]] + [col["values"] for col in cols]).T,
        columns=["timestamp"] + [f'{col["dataset"]}.{col["attr"]}' for col in cols],
    )


def preprocess(df, window_size=30):
    df = df.iloc[: df.shape[0] // window_size * window_size]
    # fill na
    df = df.interpolate().bfill()
    # TODO: replace it with batch norm
    df = (df - df.mean(axis=0)) / (df.std(axis=0) + 1e-6)  # prevent 0/0
    return df


def extract_data(ulog):
    cols = extract_mission_mode(ulog)
    if min(map(lambda x: len(x["timestamp"]), cols)) < 20:
        raise Exception("Mission too short")
    align_cols(cols)
    df = cols_to_df(cols)
    timestamp = df.pop("timestamp").values
    df = preprocess(df)
    data = df.values
    return timestamp, data


def analyse(data):
    payload = pickle.dumps(
        {"window_size": 30, "stride": 5, "data": data.astype(np.float32)}
    )
    compressed_payload = gzip.compress(payload)
    url = AI_MODEL_ENDPOINT
    print("sending request")
    response = requests.post(
        url,
        headers={"Content-Type": "application/octet-stream"},
        data=compressed_payload,
    )
    if response.status_code == 200:
        response_data = response.json()
        losses = response_data["loss"]
    else:
        raise Exception(
            f"Failed to get valid response, Status code: {response.status_code}\n{response.text}"
        )
    return losses


def add_anomaly_mask(data, losses, threshold, figs):
    window_size = 30
    stride = 5
    mask = np.zeros(len(data))
    count = np.zeros(len(data))
    idx = 0
    for loss in losses:
        if loss > threshold:
            mask[idx : idx + window_size] = loss
            count[idx : idx + window_size] += 1
        idx += stride
    mask = (mask / (count + 1e-5)).clip(0, 1)
    bool_mask = mask.astype(bool).astype(int)

    for fig in figs:
        if fig["mask"] is not None:
            bool_mask_data = fig["bool_mask"].data_source.data
            mask_data = fig["mask"].data_source.data
            bool_mask_data["image"] = [bool_mask[None]]
            mask_data["image"] = [mask[None]]
            continue

        x0, x1, y0, y1 = fig["bounds"]
        dy = y1 - y0
        fig["bool_mask"] = fig["model"].image(
            image=[bool_mask[None]],
            color_mapper=color_mapper,
            dw=x1 - x0,
            dh=(y1 - y0) * 1.2,
            x=[x0],
            y=[y0 - dy * 0.1],
            level="underlay",
            global_alpha=0.5,
            visible=False,
        )
        fig["mask"] = fig["model"].image(
            image=[mask[None]],
            color_mapper=color_mapper,
            dw=x1 - x0,
            dh=(y1 - y0) * 1.2,
            x=[x0],
            y=[y0 - dy * 0.1],
            level="underlay",
        )


def get_ai_analysis_plots(ulog, px4_ulog, db_data, link_to_main_plots):
    plots = []
    figs = []
    alpha = 0.7
    colors = itertools.cycle(palette)

    page_intro = """
<p>
This page displays anomalies detected in the flight log by our AI model. The regions
of anomalies are highlighted in pink. The darker the shade of pink, the more severe
the anomaly. The anomaly region is not specific to a graph but to a time interval,
as all parameters and their relationships are considered when detecting anomalies.
</p>
<p>
Only the time interval corresponding to mission mode is considered for anomaly detection.
The threshold value defines the limit beyond which a time interval is classified as
anomalous. In this iteration, only a few parameters are considered for detecting
anomalies in the flight log. More parameters will be added soon.
</p>
<p>
The analysis may take a while...
</p>
    """

    curdoc().template_variables["title_html"] = (
        get_heading_html(
            ulog,
            px4_ulog,
            db_data,
            None,
            [("Open Main Plots", link_to_main_plots)],
            "AI Analysis",
        )
        + page_intro
    )
    warning = Paragraph(
        text="Provided log file isn't compatible for AI analysis",
        styles={"color": "#d5433e", "font-size": "1.2rem"},
    )

    try:
        timestamp, data = extract_data(ulog)
    except Exception:
        plots.append(column(warning))
        return plots

    try:
        losses = analyse(data)
    except Exception:
        warning.text = "Failed to analyse log file"
        plots.append(column(warning))
        return plots

    for f in figures:
        x0, x1 = timestamp[0], timestamp[-1]
        y0, y1 = float("inf"), float("-inf")
        model = figure(width=800, height=400, title=f["title"])
        for p in f["plots"]:
            x = ulog.get_dataset(p["dataset"]).data["timestamp"]
            y = ulog.get_dataset(p["dataset"]).data[p["key"]]
            y0, y1 = min(y0, min(y)), max(y1, max(y))
            model.line(
                x,
                y,
                color=next(colors),
                legend_label=p["label"],
                line_width=2,
                alpha=alpha,
            )

        dy = y1 - y0
        model.y_range = Range1d(start=y0 - dy * 0.05, end=y1 + dy * 0.05)

        add_custom_tick(model)
        figs.append(
            {
                "model": model,
                "mask": None,
                "bool_mask": None,
                "bounds": [x0, x1, y0, y1],
            }
        )

    add_anomaly_mask(data, losses, default_threshold, figs)

    def spinner_callback(attr, old, new):
        add_anomaly_mask(data, losses, new, figs)

    def toggle_callback(attr, old, new):
        active = len(new) > 0
        for fig in figs:
            fig["mask"].visible = active
            fig["bool_mask"].visible = not active

    spinner = Spinner(
        title="Threshold",
        low=0.1,
        high=0.7,
        step=0.1,
        value=default_threshold,
        width=80,
    )
    toggle = CheckboxGroup(labels=["Gradient"], active=[0])
    spinner.on_change("value", spinner_callback)
    toggle.on_change("active", toggle_callback)

    plots.append(
        column(
            row(
                toggle, spinner, styles={"align-items": "center", "margin-left": "auto"}
            ),
            *[f["model"] for f in figs],
            sizing_mode="scale_width",
        )
    )

    return plots
