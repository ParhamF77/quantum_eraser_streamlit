from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import plotly.graph_objects as go
import streamlit as st


# ---------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------

st.set_page_config(
    page_title="Quantum Information Explorer",
    page_icon="🔬",
    layout="wide",
)


# ---------------------------------------------------------------------
# Physics model
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class SimulationResult:
    position: np.ndarray
    intensity: np.ndarray
    visibility: float
    distinguishability: float
    mean_intensity: float
    maximum_intensity: float
    minimum_intensity: float


def normalize(vector: np.ndarray) -> np.ndarray:
    """Return a normalized complex vector."""
    norm = np.linalg.norm(vector)

    if norm == 0:
        raise ValueError("The polarization vector cannot have zero magnitude.")

    return vector / norm


def linear_polarization(angle_degrees: float) -> np.ndarray:
    """
    Jones vector for linear polarization at a specified angle.

    0 degrees  -> horizontal
    90 degrees -> vertical
    45 degrees -> diagonal
    """
    angle = np.deg2rad(angle_degrees)

    return np.array(
        [
            np.cos(angle),
            np.sin(angle),
        ],
        dtype=complex,
    )


def elliptical_polarization(
    angle_degrees: float,
    relative_phase_degrees: float,
) -> np.ndarray:
    """
    General normalized Jones vector.

    angle_degrees controls the relative component amplitudes.
    relative_phase_degrees controls the phase of the vertical component.
    """
    angle = np.deg2rad(angle_degrees)
    relative_phase = np.deg2rad(relative_phase_degrees)

    vector = np.array(
        [
            np.cos(angle),
            np.sin(angle) * np.exp(1j * relative_phase),
        ],
        dtype=complex,
    )

    return normalize(vector)


def analyzer_vector(angle_degrees: float) -> np.ndarray:
    """Transmission axis of an ideal linear analyzer."""
    return linear_polarization(angle_degrees)


def calculate_visibility(intensity: np.ndarray) -> float:
    """Calculate fringe visibility V = (Imax - Imin) / (Imax + Imin)."""
    maximum = float(np.max(intensity))
    minimum = float(np.min(intensity))
    denominator = maximum + minimum

    if denominator <= 1e-12:
        return 0.0

    return float((maximum - minimum) / denominator)


def simulate_interference(
    upper_state: np.ndarray,
    lower_state: np.ndarray,
    upper_amplitude: float,
    lower_amplitude: float,
    phase_offset_degrees: float,
    fringe_count: int,
    analyzer_enabled: bool,
    analyzer_angle_degrees: float,
    sample_count: int = 1200,
) -> SimulationResult:
    """
    Simulate interference between two beams.

    The two paths acquire a spatially varying relative phase. Their Jones
    vectors are summed coherently before the intensity is calculated.

    When an analyzer is enabled, each path is projected onto the analyzer
    transmission axis before recombination.
    """
    x = np.linspace(-1.0, 1.0, sample_count)

    spatial_phase = (
        2.0 * np.pi * fringe_count * x
        + np.deg2rad(phase_offset_degrees)
    )

    upper = upper_amplitude * normalize(upper_state)
    lower = lower_amplitude * normalize(lower_state)

    if analyzer_enabled:
        analyzer = analyzer_vector(analyzer_angle_degrees)

        upper_projection = np.vdot(analyzer, upper)
        lower_projection = np.vdot(analyzer, lower)

        total_scalar_field = (
            upper_projection
            + lower_projection * np.exp(1j * spatial_phase)
        )

        intensity = np.abs(total_scalar_field) ** 2

    else:
        lower_phase_factor = np.exp(1j * spatial_phase)[:, np.newaxis]

        total_vector_field = (
            upper[np.newaxis, :]
            + lower[np.newaxis, :] * lower_phase_factor
        )

        intensity = np.sum(
            np.abs(total_vector_field) ** 2,
            axis=1,
        )

    visibility = calculate_visibility(intensity)

    # For two pure marker states with equal prior probabilities,
    # D^2 + V^2 = 1. This gauge is used as a conceptual indicator.
    distinguishability = float(
        np.sqrt(max(0.0, 1.0 - visibility**2))
    )

    return SimulationResult(
        position=x,
        intensity=intensity,
        visibility=visibility,
        distinguishability=distinguishability,
        mean_intensity=float(np.mean(intensity)),
        maximum_intensity=float(np.max(intensity)),
        minimum_intensity=float(np.min(intensity)),
    )


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------

def make_interference_plot(
    result: SimulationResult,
) -> go.Figure:
    figure = go.Figure()

    figure.add_trace(
        go.Scatter(
            x=result.position,
            y=result.intensity,
            mode="lines",
            line=dict(
                color="#2952CC",
                width=3,
            ),
            fill="tozeroy",
            fillcolor="rgba(41, 82, 204, 0.12)",
            name="Detected intensity",
        )
    )

    figure.update_layout(
        title="Simulated detector pattern",
        xaxis_title="Position on screen",
        yaxis_title="Relative intensity",
        template="plotly_white",
        height=420,
        margin=dict(l=40, r=30, t=60, b=45),
        showlegend=False,
    )

    figure.update_xaxes(showgrid=False)
    figure.update_yaxes(rangemode="tozero")

    return figure


def make_polarization_plot(
    upper_angle: float,
    lower_angle: float,
    analyzer_enabled: bool,
    analyzer_angle: float,
) -> go.Figure:
    figure = go.Figure()

    def add_arrow(
        angle_degrees: float,
        color: str,
        label: str,
        width: float,
    ) -> None:
        angle = np.deg2rad(angle_degrees)
        x_end = np.cos(angle)
        y_end = np.sin(angle)

        figure.add_annotation(
            x=x_end,
            y=y_end,
            ax=0,
            ay=0,
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            text="",
            showarrow=True,
            arrowhead=3,
            arrowsize=1.2,
            arrowwidth=width,
            arrowcolor=color,
        )

        figure.add_trace(
            go.Scatter(
                x=[x_end],
                y=[y_end],
                mode="markers+text",
                marker=dict(size=8, color=color),
                text=[label],
                textposition="top center",
                name=label,
            )
        )

    add_arrow(
        angle_degrees=upper_angle,
        color="#D62728",
        label="Upper path",
        width=4,
    )

    add_arrow(
        angle_degrees=lower_angle,
        color="#1F77B4",
        label="Lower path",
        width=4,
    )

    if analyzer_enabled:
        add_arrow(
            angle_degrees=analyzer_angle,
            color="#2CA02C",
            label="Analyzer",
            width=3,
        )

    figure.update_layout(
        title="Polarization directions",
        xaxis=dict(
            range=[-1.2, 1.2],
            visible=False,
            scaleanchor="y",
            scaleratio=1,
        ),
        yaxis=dict(
            range=[-1.2, 1.2],
            visible=False,
        ),
        template="plotly_white",
        height=360,
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(orientation="h"),
    )

    return figure


# ---------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------

DEFAULT_RESPONSES = {
    "student_name": "",
    "prediction": "",
    "explanation": "",
    "reflection": "",
}


for key, value in DEFAULT_RESPONSES.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ---------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------

st.title("Quantum Information Explorer")

st.markdown(
    """
Explore how polarization can make two optical paths distinguishable,
suppress interference, and restore interference when the path information
is erased.
"""
)

st.info(
    """
This simulation uses classical Jones calculus as an analogy for the
single-photon quantum eraser. It accurately represents the polarization
optics of the classroom demonstration, but it is not a single-photon
experiment.
"""
)


# ---------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------

with st.sidebar:
    st.header("Experiment controls")

    mode = st.radio(
        "Activity mode",
        options=[
            "Guided investigation",
            "Free exploration",
        ],
    )

    st.subheader("Upper path")

    upper_angle = st.slider(
        "Upper-path polarization angle",
        min_value=0,
        max_value=180,
        value=0,
        step=5,
        help="0° is horizontal and 90° is vertical.",
    )

    upper_phase = st.slider(
        "Upper-path internal phase",
        min_value=-180,
        max_value=180,
        value=0,
        step=5,
        help="Used to create elliptical polarization states.",
    )

    upper_amplitude = st.slider(
        "Upper-path amplitude",
        min_value=0.0,
        max_value=1.5,
        value=1.0,
        step=0.05,
    )

    st.subheader("Lower path")

    lower_angle = st.slider(
        "Lower-path polarization angle",
        min_value=0,
        max_value=180,
        value=90,
        step=5,
    )

    lower_phase = st.slider(
        "Lower-path internal phase",
        min_value=-180,
        max_value=180,
        value=0,
        step=5,
    )

    lower_amplitude = st.slider(
        "Lower-path amplitude",
        min_value=0.0,
        max_value=1.5,
        value=1.0,
        step=0.05,
    )

    st.subheader("Interference")

    phase_offset = st.slider(
        "Relative phase offset",
        min_value=0,
        max_value=360,
        value=0,
        step=5,
    )

    fringe_count = st.slider(
        "Number of fringes",
        min_value=1,
        max_value=12,
        value=5,
        step=1,
    )

    st.subheader("Quantum eraser")

    analyzer_enabled = st.checkbox(
        "Insert analyzer after recombination",
        value=False,
    )

    analyzer_angle = st.slider(
        "Analyzer angle",
        min_value=0,
        max_value=180,
        value=45,
        step=5,
        disabled=not analyzer_enabled,
    )


# ---------------------------------------------------------------------
# Run simulation
# ---------------------------------------------------------------------

upper_state = elliptical_polarization(
    angle_degrees=upper_angle,
    relative_phase_degrees=upper_phase,
)

lower_state = elliptical_polarization(
    angle_degrees=lower_angle,
    relative_phase_degrees=lower_phase,
)

result = simulate_interference(
    upper_state=upper_state,
    lower_state=lower_state,
    upper_amplitude=upper_amplitude,
    lower_amplitude=lower_amplitude,
    phase_offset_degrees=phase_offset,
    fringe_count=fringe_count,
    analyzer_enabled=analyzer_enabled,
    analyzer_angle_degrees=analyzer_angle,
)


# ---------------------------------------------------------------------
# Guided investigation
# ---------------------------------------------------------------------

if mode == "Guided investigation":
    st.header("Guided investigation")

    investigation = st.selectbox(
        "Choose an investigation",
        options=[
            "1. Same polarization",
            "2. Orthogonal path markers",
            "3. Erase the path information",
            "4. Partial distinguishability",
        ],
    )

    if investigation == "1. Same polarization":
        st.markdown(
            """
**Recommended settings**

- Upper path: (0 degrees)
- Lower path: (0 degrees)
- Analyzer: off

Predict whether a high-visibility interference pattern will appear.
"""
        )

    elif investigation == "2. Orthogonal path markers":
        st.markdown(
            """
**Recommended settings**

- Upper path: (0 degrees)
- Lower path: (90 degrees)
- Analyzer: off

The two paths carry orthogonal polarization labels. Predict what happens
to the interference pattern.
"""
        )

    elif investigation == "3. Erase the path information":
        st.markdown(
            """
**Recommended settings**

- Upper path: (0 degrees)
- Lower path: (90 degrees)
- Analyzer: on
- Analyzer angle: (45 degrees)

The analyzer projects both path markers onto a common polarization axis.
Predict whether interference will be restored.
"""
        )

    else:
        st.markdown(
            """
Set the upper path to 0 degrees, then rotate the lower-path polarization
from 0 degrees toward 90 degrees.

Observe how fringe visibility changes continuously as the two paths
become more distinguishable.
"""
        )

    st.session_state["student_name"] = st.text_input(
        "Student or group name",
        value=st.session_state["student_name"],
    )

    st.session_state["prediction"] = st.text_area(
        "Prediction: What pattern do you expect?",
        value=st.session_state["prediction"],
        height=90,
    )

    st.session_state["explanation"] = st.text_area(
        "Reasoning: Why do you expect this result?",
        value=st.session_state["explanation"],
        height=100,
    )


# ---------------------------------------------------------------------
# Main display
# ---------------------------------------------------------------------

left_column, right_column = st.columns([1.45, 1.0])

with left_column:
    st.plotly_chart(
        make_interference_plot(result),
        use_container_width=True,
    )

with right_column:
    st.plotly_chart(
        make_polarization_plot(
            upper_angle=upper_angle,
            lower_angle=lower_angle,
            analyzer_enabled=analyzer_enabled,
            analyzer_angle=analyzer_angle,
        ),
        use_container_width=True,
    )


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

metric_1, metric_2, metric_3, metric_4 = st.columns(4)

metric_1.metric(
    "Fringe visibility",
    f"{result.visibility:.3f}",
)

metric_2.metric(
    "Path distinguishability",
    f"{result.distinguishability:.3f}",
)

metric_3.metric(
    "Maximum intensity",
    f"{result.maximum_intensity:.3f}",
)

metric_4.metric(
    "Mean intensity",
    f"{result.mean_intensity:.3f}",
)

st.caption(
    r"""
For ideal pure marker states and equal path probabilities, the simulation
illustrates the complementarity relation
\[
V^2 + D^2 = 1,
\]
where \(V\) is fringe visibility and \(D\) is path distinguishability.
"""
)


# ---------------------------------------------------------------------
# Conceptual interpretation
# ---------------------------------------------------------------------

st.header("Interpret the result")

if result.visibility > 0.85:
    st.success(
        """
The two detected path amplitudes are sufficiently indistinguishable to
produce strong interference.
"""
    )

elif result.visibility > 0.25:
    st.warning(
        """
Partial path distinguishability remains. The interference pattern is
visible, but its contrast is reduced.
"""
    )

else:
    st.error(
        """
The detected paths are highly distinguishable, so little or no
interference is observed.
"""
    )

if analyzer_enabled:
    st.markdown(
        f"""
The analyzer is set to **{analyzer_angle}°**. It projects both path
polarizations onto the same measurement direction. This can erase the
polarization label that distinguished the paths, although the total
transmitted intensity may decrease.
"""
    )
else:
    st.markdown(
        """
No analyzer is present. The detector remains sensitive to the full
polarization states carried by the two paths.
"""
    )


# ---------------------------------------------------------------------
# Reflection and export
# ---------------------------------------------------------------------

st.header("Student reflection")

st.session_state["reflection"] = st.text_area(
    "Explain how polarization affects path information and interference.",
    value=st.session_state["reflection"],
    height=130,
)

response_data = {
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "student_name": st.session_state["student_name"],
    "mode": mode,
    "upper_angle_degrees": upper_angle,
    "upper_internal_phase_degrees": upper_phase,
    "upper_amplitude": upper_amplitude,
    "lower_angle_degrees": lower_angle,
    "lower_internal_phase_degrees": lower_phase,
    "lower_amplitude": lower_amplitude,
    "relative_phase_offset_degrees": phase_offset,
    "fringe_count": fringe_count,
    "analyzer_enabled": analyzer_enabled,
    "analyzer_angle_degrees": (
        analyzer_angle if analyzer_enabled else None
    ),
    "fringe_visibility": result.visibility,
    "path_distinguishability": result.distinguishability,
    "prediction": st.session_state["prediction"],
    "reasoning": st.session_state["explanation"],
    "reflection": st.session_state["reflection"],
}

download_column_1, download_column_2 = st.columns(2)

with download_column_1:
    json_text = json.dumps(
        response_data,
        indent=2,
    )

    st.download_button(
        label="Download response as JSON",
        data=json_text,
        file_name="quantum_eraser_response.json",
        mime="application/json",
    )

with download_column_2:
    csv_buffer = io.StringIO()
    writer = csv.DictWriter(
        csv_buffer,
        fieldnames=response_data.keys(),
    )
    writer.writeheader()
    writer.writerow(response_data)

    st.download_button(
        label="Download response as CSV",
        data=csv_buffer.getvalue(),
        file_name="quantum_eraser_response.csv",
        mime="text/csv",
    )


# ---------------------------------------------------------------------
# Instructor notes
# ---------------------------------------------------------------------

with st.expander("Instructor notes and physics"):
    st.markdown(
        r"""
### Field model

The two path fields are represented by Jones vectors

\[
\mathbf{E}_1
\quad \text{and} \quad
\mathbf{E}_2.
\]

At screen position \(x\), the lower path acquires a relative phase
\(\phi(x)\), giving

\[
\mathbf{E}_{\mathrm{tot}}(x)
=
\mathbf{E}_1
+
e^{i\phi(x)}\mathbf{E}_2.
\]

The detected intensity is

\[
I(x)
=
\mathbf{E}_{\mathrm{tot}}^\dagger
\mathbf{E}_{\mathrm{tot}}.
\]

For equal-amplitude paths, the interference term is proportional to

\[
\langle E_1 | E_2 \rangle.
\]

If the polarization states are orthogonal,

\[
\langle H|V\rangle = 0,
\]

and the interference term vanishes.

### Eraser projection

An analyzer oriented along \(|a\rangle\) projects each path into the same
measurement channel:

\[
A_1 = \langle a|E_1\rangle,
\qquad
A_2 = \langle a|E_2\rangle.
\]

The detected signal becomes

\[
I_a(x)
=
\left|
A_1 + e^{i\phi(x)} A_2
\right|^2.
\]

For horizontal and vertical path markers with a 45 degree analyzer,

\[
\langle 45^\circ|H\rangle
=
\langle 45^\circ|V\rangle
=
\frac{1}{\sqrt{2}},
\]

so interference is restored in the selected analyzer output.

### Important limitation

This is a classical polarization-optics simulation. It reproduces the
mathematical structure of the analogy experiment but does not simulate
single-photon detection statistics, entanglement, or delayed-choice
quantum eraser experiments.
"""
    )
