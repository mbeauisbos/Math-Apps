import streamlit as st
import numpy as np
import pandas as pd
import altair as alt

st.set_page_config(page_title="2D Free Fall Trajectory Simulator", layout="wide")

if "pinned_trails" not in st.session_state:
    st.session_state.pinned_trails = []

TRAIL_COLORS = ["#f472b6", "#a78bfa", "#34d399", "#fbbf24", "#fb7185"]


def calculate_ideal(v0: float, angle_deg: float, h0: float, g: float, num_points: int = 200):
    """Analytical kinematic solution without air resistance."""
    theta = np.radians(angle_deg)
    vx0 = v0 * np.cos(theta)
    vy0 = v0 * np.sin(theta)

    discriminant = vy0**2 + 2 * g * h0
    total_time = (vy0 + np.sqrt(discriminant)) / g

    t = np.linspace(0, total_time, num_points)
    x = vx0 * t
    y = np.maximum(0.0, h0 + vy0 * t - 0.5 * g * t**2)
    vx = np.full_like(t, vx0)
    vy = vy0 - g * t
    speed = np.hypot(vx, vy)

    t_peak = max(0.0, vy0 / g)
    x_peak = vx0 * t_peak
    y_peak = h0 + vy0 * t_peak - 0.5 * g * t_peak**2

    return {
        "t": t,
        "x": x,
        "y": y,
        "vx": vx,
        "vy": vy,
        "speed": speed,
        "x_peak": float(x_peak),
        "y_peak": float(y_peak),
        "range": float(x[-1]),
        "total_time": float(total_time),
        "v_impact": float(speed[-1]),
    }


def calculate_with_drag(
    v0: float,
    angle_deg: float,
    h0: float,
    g: float,
    mass: float,
    cd: float,
    area: float,
    rho: float,
    dt: float = 0.01,
):
    """Numerical RK4 integration with quadratic aerodynamic drag: F_d = 0.5 * rho * Cd * A * v^2."""
    theta = np.radians(angle_deg)
    vx = v0 * np.cos(theta)
    vy = v0 * np.sin(theta)
    x, y, t = 0.0, h0, 0.0

    k = 0.5 * rho * cd * area / mass

    def derivatives(vx_s, vy_s):
        v = np.hypot(vx_s, vy_s)
        ax = -k * v * vx_s
        ay = -g - k * v * vy_s
        return vx_s, vy_s, ax, ay

    ts, xs, ys, vxs, vys = [t], [x], [y], [vx], [vy]

    for _ in range(100_000):
        dx1, dy1, dvx1, dvy1 = derivatives(vx, vy)
        dx2, dy2, dvx2, dvy2 = derivatives(vx + 0.5 * dt * dvx1, vy + 0.5 * dt * dvy1)
        dx3, dy3, dvx3, dvy3 = derivatives(vx + 0.5 * dt * dvx2, vy + 0.5 * dt * dvy2)
        dx4, dy4, dvx4, dvy4 = derivatives(vx + dt * dvx3, vy + dt * dvy3)

        x_next = x + (dt / 6.0) * (dx1 + 2 * dx2 + 2 * dx3 + dx4)
        y_next = y + (dt / 6.0) * (dy1 + 2 * dy2 + 2 * dy3 + dy4)
        vx_next = vx + (dt / 6.0) * (dvx1 + 2 * dvx2 + 2 * dvx3 + dvx4)
        vy_next = vy + (dt / 6.0) * (dvy1 + 2 * dvy2 + 2 * dvy3 + dvy4)
        t_next = t + dt

        if y_next <= 0.0 and t > 0:
            frac = y / (y - y_next) if (y - y_next) != 0 else 1.0
            ts.append(t + frac * dt)
            xs.append(x + frac * (x_next - x))
            ys.append(0.0)
            vxs.append(vx + frac * (vx_next - vx))
            vys.append(vy + frac * (vy_next - vy))
            break

        x, y, vx, vy, t = x_next, y_next, vx_next, vy_next, t_next
        ts.append(t)
        xs.append(x)
        ys.append(y)
        vxs.append(vx)
        vys.append(vy)

    t_arr = np.array(ts)
    x_arr = np.array(xs)
    y_arr = np.array(ys)
    vx_arr = np.array(vxs)
    vy_arr = np.array(vys)
    speed_arr = np.hypot(vx_arr, vy_arr)
    peak_idx = int(np.argmax(y_arr))

    return {
        "t": t_arr,
        "x": x_arr,
        "y": y_arr,
        "vx": vx_arr,
        "vy": vy_arr,
        "speed": speed_arr,
        "x_peak": float(x_arr[peak_idx]),
        "y_peak": float(y_arr[peak_idx]),
        "range": float(x_arr[-1]),
        "total_time": float(t_arr[-1]),
        "v_impact": float(speed_arr[-1]),
    }


st.sidebar.header("Launch Parameters")

v0 = st.sidebar.slider("Initial Velocity v₀ (m/s)", min_value=1.0, max_value=100.0, value=30.0, step=0.5)
angle = st.sidebar.slider("Launch Angle θ (deg)", min_value=-85.0, max_value=90.0, value=45.0, step=1.0)
h0 = st.sidebar.slider("Initial Height h₀ (m)", min_value=0.0, max_value=100.0, value=10.0, step=0.5)

planet_presets = {
    "Earth (9.81 m/s²)": 9.81,
    "Moon (1.62 m/s²)": 1.62,
    "Mars (3.72 m/s²)": 3.72,
    "Jupiter (24.79 m/s²)": 24.79,
    "Custom": None,
}
preset = st.sidebar.selectbox("Gravity Preset", list(planet_presets.keys()))
if planet_presets[preset] is not None:
    g = planet_presets[preset]
else:
    g = st.sidebar.slider("Custom Gravity g (m/s²)", min_value=0.5, max_value=30.0, value=9.81, step=0.01)

st.sidebar.divider()
enable_drag = st.sidebar.checkbox("Enable Air Resistance (Quadratic Drag)", value=False)

if enable_drag:
    mass = st.sidebar.number_input("Projectile Mass (kg)", min_value=0.01, max_value=1000.0, value=0.145, step=0.05)
    diameter_cm = st.sidebar.slider("Sphere Diameter (cm)", min_value=1.0, max_value=100.0, value=7.4, step=0.2)
    cd = st.sidebar.slider("Drag Coefficient (C_d)", min_value=0.05, max_value=1.50, value=0.47, step=0.01)
    rho = st.sidebar.number_input("Air Density ρ (kg/m³)", min_value=0.0, max_value=5.0, value=1.225, step=0.05)
    area = np.pi * ((diameter_cm / 100.0) / 2.0) ** 2

ideal_traj = calculate_ideal(v0, angle, h0, g)

if enable_drag:
    active_traj = calculate_with_drag(v0, angle, h0, g, mass, cd, area, rho)
    active_label = f"Active (Drag: v₀={v0}, θ={angle}°, h₀={h0}m)"
else:
    active_traj = ideal_traj
    active_label = f"Active (v₀={v0}, θ={angle}°, h₀={h0}m)"

st.sidebar.divider()
col_pin, col_clear = st.sidebar.columns(2)
if col_pin.button("Pin Curve", use_container_width=True):
    color = TRAIL_COLORS[len(st.session_state.pinned_trails) % len(TRAIL_COLORS)]
    st.session_state.pinned_trails.append(
        {
            "x": active_traj["x"],
            "y": active_traj["y"],
            "t": active_traj["t"],
            "speed": active_traj["speed"],
            "range": active_traj["range"],
            "y_peak": active_traj["y_peak"],
            "label": f"Pinned #{len(st.session_state.pinned_trails) + 1}: v₀={v0}, θ={angle}°, h₀={h0}m",
            "color": color,
        }
    )

if col_clear.button("Clear Trails", use_container_width=True):
    st.session_state.pinned_trails = []

df_export = pd.DataFrame(
    {
        "time_s": active_traj["t"],
        "x_m": active_traj["x"],
        "y_m": active_traj["y"],
        "vx_m_s": active_traj["vx"],
        "vy_m_s": active_traj["vy"],
        "speed_m_s": active_traj["speed"],
    }
)
st.sidebar.download_button(
    label="Download Trajectory CSV",
    data=df_export.to_csv(index=False).encode("utf-8"),
    file_name="trajectory_data.csv",
    mime="text/csv",
    use_container_width=True,
)

st.title("2D Free Fall Trajectory Simulator")

m1, m2, m3, m4 = st.columns(4)
m1.metric(
    "Max Range (R)",
    f"{active_traj['range']:.2f} m",
    delta=f"{active_traj['range'] - ideal_traj['range']:.2f} m vs vacuum" if enable_drag else None,
)
m2.metric(
    "Max Height (H)",
    f"{active_traj['y_peak']:.2f} m",
    delta=f"{active_traj['y_peak'] - ideal_traj['y_peak']:.2f} m vs vacuum" if enable_drag else None,
)
m3.metric(
    "Flight Time (T)",
    f"{active_traj['total_time']:.2f} s",
    delta=f"{active_traj['total_time'] - ideal_traj['total_time']:.2f} s vs vacuum" if enable_drag else None,
)
m4.metric(
    "Impact Speed (v_f)",
    f"{active_traj['v_impact']:.2f} m/s",
    delta=f"{active_traj['v_impact'] - ideal_traj['v_impact']:.2f} m/s vs vacuum" if enable_drag else None,
)

t_scrub = st.slider(
    "Inspect Projectile at Time t (s)",
    min_value=0.0,
    max_value=float(active_traj["total_time"]),
    value=0.0,
    step=max(0.01, round(active_traj["total_time"] / 100.0, 2)),
)
scrub_idx = int(np.argmin(np.abs(active_traj["t"] - t_scrub)))

frames_list = [
    pd.DataFrame(
        {
            "x": active_traj["x"],
            "y": active_traj["y"],
            "t": active_traj["t"],
            "speed": active_traj["speed"],
            "Series": active_label,
            "Style": "solid",
        }
    )
]

domain = [active_label]
range_colors = ["#38bdf8"]

if enable_drag:
    frames_list.append(
        pd.DataFrame(
            {
                "x": ideal_traj["x"],
                "y": ideal_traj["y"],
                "t": ideal_traj["t"],
                "speed": ideal_traj["speed"],
                "Series": "Ideal Vacuum Reference",
                "Style": "dashed",
            }
        )
    )
    domain.append("Ideal Vacuum Reference")
    range_colors.append("#94a3b8")

for trail in st.session_state.pinned_trails:
    frames_list.append(
        pd.DataFrame(
            {
                "x": trail["x"],
                "y": trail["y"],
                "t": trail["t"],
                "speed": trail["speed"],
                "Series": trail["label"],
                "Style": "dashed",
            }
        )
    )
    domain.append(trail["label"])
    range_colors.append(trail["color"])

plot_df = pd.concat(frames_list, ignore_index=True)

all_trajs = st.session_state.pinned_trails + [active_traj, ideal_traj]
max_bound = max(10.0, max(t["range"] for t in all_trajs) * 1.1, max(t["y_peak"] for t in all_trajs) * 1.15)

lines_chart = (
    alt.Chart(plot_df)
    .mark_line(strokeWidth=3)
    .encode(
        x=alt.X("x:Q", title="Horizontal Distance x (m)", scale=alt.Scale(domain=[0, max_bound])),
        y=alt.Y("y:Q", title="Vertical Height y (m)", scale=alt.Scale(domain=[0, max_bound])),
        color=alt.Color(
            "Series:N",
            scale=alt.Scale(domain=domain, range=range_colors),
            legend=alt.Legend(orient="top", title=None),
        ),
        strokeDash=alt.condition(
            alt.datum.Style == "dashed",
            alt.value([6, 4]),
            alt.value([1, 0]),
        ),
        tooltip=[
            alt.Tooltip("Series:N", title="Trajectory"),
            alt.Tooltip("x:Q", title="x (m)", format=".2f"),
            alt.Tooltip("y:Q", title="y (m)", format=".2f"),
            alt.Tooltip("t:Q", title="Time (s)", format=".2f"),
            alt.Tooltip("speed:Q", title="Speed (m/s)", format=".2f"),
        ],
    )
)

markers_df = pd.DataFrame(
    [
        {
            "x": float(active_traj["x"][scrub_idx]),
            "y": float(active_traj["y"][scrub_idx]),
            "Point": f"Projectile (t={active_traj['t'][scrub_idx]:.2f}s)",
            "color": "#f8fafc",
            "shape": "circle",
        },
        {
            "x": active_traj["x_peak"],
            "y": active_traj["y_peak"],
            "Point": f"Apex ({active_traj['x_peak']:.2f}m, {active_traj['y_peak']:.2f}m)",
            "color": "#fbbf24",
            "shape": "triangle-up",
        },
    ]
)

markers_chart = (
    alt.Chart(markers_df)
    .mark_point(size=140, filled=True, stroke="#38bdf8", strokeWidth=2)
    .encode(
        x="x:Q",
        y="y:Q",
        color=alt.Color("color:N", scale=None),
        shape=alt.Shape("shape:N", scale=None),
        tooltip=[
            alt.Tooltip("Point:N", title="Marker"),
            alt.Tooltip("x:Q", title="x (m)", format=".2f"),
            alt.Tooltip("y:Q", title="y (m)", format=".2f"),
        ],
    )
)

combined_chart = (lines_chart + markers_chart).properties(height=550).interactive()
st.altair_chart(combined_chart, use_container_width=True)
