import streamlit as st
import numpy as np
import pandas as pd
import altair as alt

st.set_page_config(page_title="3D Plane & Line Intersection", layout="wide")


def project_3d_to_2d(pts: np.ndarray, yaw_deg: float, pitch_deg: float) -> np.ndarray:
    """Projects (N, 3) 3D points onto a 2D camera plane rotated by yaw and pitch."""
    yaw = np.radians(yaw_deg)
    pitch = np.radians(pitch_deg)

    # Rotate around Z-axis (yaw)
    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0.0],
        [np.sin(yaw),  np.cos(yaw), 0.0],
        [0.0,          0.0,         1.0],
    ])
    # Rotate around X-axis (pitch)
    Rx = np.array([
        [1.0, 0.0,            0.0],
        [0.0, np.cos(pitch), -np.sin(pitch)],
        [0.0, np.sin(pitch),  np.cos(pitch)],
    ])

    rotated = pts @ Rz.T @ Rx.T
    # Return horizontal screen X and vertical screen Z (depth is rotated[:, 1])
    return rotated[:, [0, 2]]


# --- Sidebar Controls ---
st.sidebar.header("1. Plane Parameters")
st.sidebar.latex(r"Ax + By + Cz = D")

col_p1, col_p2 = st.sidebar.columns(2)
A = col_p1.number_input("A", value=1.0, step=0.5, format="%.2f")
B = col_p2.number_input("B", value=1.0, step=0.5, format="%.2f")
C = col_p1.number_input("C", value=2.0, step=0.5, format="%.2f")
D = col_p2.number_input("D", value=4.0, step=0.5, format="%.2f")

st.sidebar.divider()
st.sidebar.header("2. Line Parameters")
st.sidebar.latex(r"\begin{cases} x = x_0 + a k \\ y = y_0 + b k \\ z = z_0 + c k \end{cases}")

st.sidebar.markdown("**Base Point $(x_0, y_0, z_0)$**")
col_l0_1, col_l0_2, col_l0_3 = st.sidebar.columns(3)
x0 = col_l0_1.number_input("x₀", value=0.0, step=0.5, format="%.2f")
y0 = col_l0_2.number_input("y₀", value=0.0, step=0.5, format="%.2f")
z0 = col_l0_3.number_input("z₀", value=0.0, step=0.5, format="%.2f")

st.sidebar.markdown(r"**Direction Vector $\vec{d} = (a, b, c)$**")
col_d1, col_d2, col_d3 = st.sidebar.columns(3)
a = col_d1.number_input("a", value=1.0, step=0.5, format="%.2f")
b = col_d2.number_input("b", value=2.0, step=0.5, format="%.2f")
c = col_d3.number_input("c", value=1.0, step=0.5, format="%.2f")

st.sidebar.divider()
st.sidebar.header("3. 3D Camera & Display")
yaw_deg = st.sidebar.slider("Rotate Horizontal (Yaw °)", -180, 180, -35, 5)
pitch_deg = st.sidebar.slider("Rotate Vertical (Pitch °)", -85, 85, 25, 5)
k_range = st.sidebar.slider("Line Parameter Range (k)", 1.0, 20.0, 6.0, 0.5)
vec_scale = st.sidebar.slider("Vector Display Scale", 0.2, 3.0, 1.0, 0.1)

# --- Main Display ---
st.title("3D Plane and Line Intersection Explorer")

n_vec = np.array([A, B, C], dtype=float)
d_vec = np.array([a, b, c], dtype=float)
p_line0 = np.array([x0, y0, z0], dtype=float)

if np.linalg.norm(n_vec) < 1e-12:
    st.error("Invalid plane: Normal vector (A, B, C) cannot be (0, 0, 0).")
    st.stop()

if np.linalg.norm(d_vec) < 1e-12:
    st.error("Invalid line: Direction vector (a, b, c) cannot be (0, 0, 0).")
    st.stop()

denom = float(np.dot(n_vec, d_vec))
rhs = float(D - np.dot(n_vec, p_line0))

cos_theta = np.clip(abs(denom) / (np.linalg.norm(n_vec) * np.linalg.norm(d_vec)), 0.0, 1.0)
angle_deg = float(np.degrees(np.arcsin(cos_theta)))

if abs(denom) > 1e-10:
    case = "unique"
    k_star = rhs / denom
    p_int = p_line0 + k_star * d_vec
    status_title = "Single Intersection Point"
    status_detail = f"k = {k_star:.4f} → P = ({p_int[0]:.3f}, {p_int[1]:.3f}, {p_int[2]:.3f})"
elif abs(rhs) <= 1e-10:
    case = "contained"
    k_star = None
    p_int = None
    status_title = "Line Lies Inside the Plane"
    status_detail = "Infinitely many intersection points (n · d = 0 and Ax₀ + By₀ + Cz₀ = D)"
else:
    case = "parallel"
    k_star = None
    p_int = None
    status_title = "Parallel (No Intersection)"
    status_detail = "0 intersection points (n · d = 0 and Ax₀ + By₀ + Cz₀ ≠ D)"

if case == "unique":
    st.success(f"**{status_title}:** `{status_detail}`")
elif case == "contained":
    st.info(f"**{status_title}:** {status_detail}")
else:
    st.warning(f"**{status_title}:** {status_detail}")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Normal Vector n", f"({A:g}, {B:g}, {C:g})")
m2.metric("Direction Vector d", f"({a:g}, {b:g}, {c:g})")
m3.metric("Dot Product (n · d)", f"{denom:.3f}")
m4.metric("Angle (Line to Plane)", f"{angle_deg:.2f}°")

# Frame the 3D scene
if case == "unique" and abs(k_star) <= k_range * 2.5:
    scene_center = p_int
    k_min = min(-k_range, k_star - 2.0)
    k_max = max(k_range, k_star + 2.0)
else:
    scene_center = p_line0
    k_min, k_max = -k_range, k_range

# Construct orthonormal basis (u, v) on the plane around `scene_center`
n_norm_sq = np.dot(n_vec, n_vec)
p_plane_anchor = scene_center - ((np.dot(n_vec, scene_center) - D) / n_norm_sq) * n_vec
helper = np.array([1.0, 0.0, 0.0]) if abs(n_vec[0]) < 0.9 * np.sqrt(n_norm_sq) else np.array([0.0, 1.0, 0.0])
u_vec = np.cross(n_vec, helper)
u_vec /= np.linalg.norm(u_vec)
v_vec = np.cross(n_vec, u_vec)
v_vec /= np.linalg.norm(v_vec)

line_pts_3d = np.vstack([p_line0 + k_min * d_vec, p_line0 + k_max * d_vec])
half_span = max(6.0, np.linalg.norm(line_pts_3d[1] - line_pts_3d[0]) * 0.55)

# Build Plane Wireframe Grid + Boundary Polygon
segments = []
grid_vals = np.linspace(-half_span, half_span, 9)
for idx, g_val in enumerate(grid_vals):
    # Lines along u
    p_a = p_plane_anchor - half_span * u_vec + g_val * v_vec
    p_b = p_plane_anchor + half_span * u_vec + g_val * v_vec
    segments.append((f"plane_u_{idx}", "Plane Grid", p_a, p_b))
    # Lines along v
    p_c = p_plane_anchor + g_val * u_vec - half_span * v_vec
    p_d = p_plane_anchor + g_val * u_vec + half_span * v_vec
    segments.append((f"plane_v_{idx}", "Plane Grid", p_c, p_d))

# 3D Coordinate Axes through scene_center
axis_len = half_span * 0.6
for ax_name, ax_dir in [("X-Axis", [1, 0, 0]), ("Y-Axis", [0, 1, 0]), ("Z-Axis", [0, 0, 1])]:
    ax_v = np.array(ax_dir, dtype=float) * axis_len
    segments.append((ax_name, "3D Axes", scene_center - ax_v, scene_center + ax_v))

# Parametric Line
segments.append(("line_main", "Parametric Line L(k)", line_pts_3d[0], line_pts_3d[1]))

# Normal Vector n = (A, B, C)
normal_origin = p_int if case == "unique" else p_plane_anchor
normal_tip = normal_origin + n_vec * vec_scale
segments.append(("vec_n", f"Normal Vector n = ({A:g}, {B:g}, {C:g})", normal_origin, normal_tip))

# Direction Vector d = (a, b, c)
dir_tip = p_line0 + d_vec * vec_scale
segments.append(("vec_d", f"Direction Vector d = ({a:g}, {b:g}, {c:g})", p_line0, dir_tip))

# Project all segments to 2D
line_rows = []
for seg_id, category, p1, p2 in segments:
    proj = project_3d_to_2d(np.vstack([p1, p2]), yaw_deg, pitch_deg)
    for order, (pt3, pt2) in enumerate(zip([p1, p2], proj)):
        line_rows.append({
            "seg_id": seg_id,
            "Category": category,
            "order": order,
            "sx": float(pt2[0]),
            "sy": float(pt2[1]),
            "x3": float(pt3[0]),
            "y3": float(pt3[1]),
            "z3": float(pt3[2]),
        })

df_lines = pd.DataFrame(line_rows)

# Points (Base point, Vector tips, Intersection point)
point_items = [
    ("Base Point P₀", p_line0, "#f8fafc", "circle", 120),
    ("Tip of Normal n", normal_tip, "#22d3ee", "triangle-up", 150),
    ("Tip of Direction d", dir_tip, "#a3e635", "triangle-up", 150),
]
if case == "unique":
    point_items.append(("Intersection Point P", p_int, "#fbbf24", "diamond", 220))

pt_rows = []
for label, pt3, color, shape, size in point_items:
    pt2 = project_3d_to_2d(pt3.reshape(1, 3), yaw_deg, pitch_deg)[0]
    pt_rows.append({
        "Label": label,
        "sx": float(pt2[0]),
        "sy": float(pt2[1]),
        "x3": float(pt3[0]),
        "y3": float(pt3[1]),
        "z3": float(pt3[2]),
        "color": color,
        "shape": shape,
        "size": size,
    })
df_pts = pd.DataFrame(pt_rows)

# Unified square bounds for undistorted 3D projection
all_sx = np.concatenate([df_lines["sx"].values, df_pts["sx"].values])
all_sy = np.concatenate([df_lines["sy"].values, df_pts["sy"].values])
cx, cy = 0.5 * (all_sx.min() + all_sx.max()), 0.5 * (all_sy.min() + all_sy.max())
span = max(all_sx.max() - all_sx.min(), all_sy.max() - all_sy.min(), 5.0) * 0.58

n_label = f"Normal Vector n = ({A:g}, {B:g}, {C:g})"
d_label = f"Direction Vector d = ({a:g}, {b:g}, {c:g})"

domain = ["3D Axes", "Plane Grid", "Parametric Line L(k)", n_label, d_label]
colors = ["#334155", "#0284c7", "#f43f5e", "#22d3ee", "#a3e635"]

chart_lines = (
    alt.Chart(df_lines)
    .mark_line()
    .encode(
        x=alt.X("sx:Q", scale=alt.Scale(domain=[cx - span, cx + span]), axis=None),
        y=alt.Y("sy:Q", scale=alt.Scale(domain=[cy - span, cy + span]), axis=None),
        detail="seg_id:N",
        order="order:Q",
        color=alt.Color("Category:N", scale=alt.Scale(domain=domain, range=colors), legend=alt.Legend(orient="top", title=None)),
        strokeWidth=alt.condition(
            alt.FieldOneOfPredicate(field="Category", oneOf=["Parametric Line L(k)", n_label, d_label]),
            alt.value(4.0),
            alt.value(1.2),
        ),
        opacity=alt.condition(
            alt.datum.Category == "Plane Grid",
            alt.value(0.45),
            alt.value(0.95),
        ),
        tooltip=[
            alt.Tooltip("Category:N", title="Object"),
            alt.Tooltip("x3:Q", title="X", format=".2f"),
            alt.Tooltip("y3:Q", title="Y", format=".2f"),
            alt.Tooltip("z3:Q", title="Z", format=".2f"),
        ],
    )
)

chart_pts = (
    alt.Chart(df_pts)
    .mark_point(filled=True, stroke="#0f172a", strokeWidth=1.5)
    .encode(
        x="sx:Q",
        y="sy:Q",
        color=alt.Color("color:N", scale=None),
        shape=alt.Shape("shape:N", scale=None),
        size=alt.Size("size:Q", scale=None),
        tooltip=[
            alt.Tooltip("Label:N", title="Point"),
            alt.Tooltip("x3:Q", title="X", format=".3f"),
            alt.Tooltip("y3:Q", title="Y", format=".3f"),
            alt.Tooltip("z3:Q", title="Z", format=".3f"),
        ],
    )
)

st.altair_chart((chart_lines + chart_pts).properties(height=580).interactive(), use_container_width=True)

with st.expander("Show Step-by-Step Algebraic Solution"):
    st.markdown("**1. Substitute the line's parametric equations into the plane equation:**")
    st.latex(
        rf"{A:g}({x0:g} + {a:g}k) + {B:g}({y0:g} + {b:g}k) + {C:g}({z0:g} + {c:g}k) = {D:g}"
    )
    st.markdown(r"**2. Group constant terms and $k$-terms ($\vec{n}\cdot\vec{d}\,k = D - \vec{n}\cdot P_0$):**")
    st.latex(rf"({denom:g})k = {rhs:g}")
    if case == "unique":
        st.markdown("**3. Solve for $k$ and substitute back into $L(k)$:**")
        st.latex(rf"k = \frac{{{rhs:g}}}{{{denom:g}}} = {k_star:.4f}")
        st.latex(
            rf"P = \left({x0:g} + ({a:g})({k_star:.4f}),\; {y0:g} + ({b:g})({k_star:.4f}),\; {z0:g} + ({c:g})({k_star:.4f})\right) = ({p_int[0]:.4f},\; {p_int[1]:.4f},\; {p_int[2]:.4f})"
        )
    elif case == "contained":
        st.markdown(r"**3. Result:** $0 \cdot k = 0$ holds for every $k \in \mathbb{R}$, so the entire line lies on the plane.")
    else:
        st.markdown(rf"**3. Result:** $0 \cdot k = {rhs:g}$ has no solution, so the line is strictly parallel to the plane.")
