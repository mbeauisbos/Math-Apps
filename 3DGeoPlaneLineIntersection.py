import streamlit as st
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="3D Plane & Line Intersection", layout="wide")


def add_vector_arrow(fig, start, vec, color, name, scale=1.0):
    """Adds a 3D vector shaft + arrowhead cone from `start` along `vec * scale`."""
    start = np.asarray(start, dtype=float)
    vec = np.asarray(vec, dtype=float)
    norm = np.linalg.norm(vec)
    if norm < 1e-12:
        return

    scaled_vec = vec * scale
    end = start + scaled_vec
    tip_dir = scaled_vec * 0.22

    fig.add_trace(
        go.Scatter3d(
            x=[start[0], end[0]],
            y=[start[1], end[1]],
            z=[start[2], end[2]],
            mode="lines",
            line=dict(color=color, width=6),
            name=name,
            hovertemplate=(
                f"{name}<br>"
                f"Origin: ({start[0]:.2f}, {start[1]:.2f}, {start[2]:.2f})<br>"
                f"Components: ({vec[0]:.2f}, {vec[1]:.2f}, {vec[2]:.2f})<extra></extra>"
            ),
        )
    )

    fig.add_trace(
        go.Cone(
            x=[end[0]],
            y=[end[1]],
            z=[end[2]],
            u=[tip_dir[0]],
            v=[tip_dir[1]],
            w=[tip_dir[2]],
            anchor="tip",
            sizemode="raw",
            colorscale=[[0, color], [1, color]],
            showscale=False,
            name=name,
            showlegend=False,
            hoverinfo="skip",
        )
    )


def build_plane_mesh(A, B, C, D, center, half_span):
    """Generates an orthonormal basis on the plane Ax + By + Cz = D centered near `center`."""
    n = np.array([A, B, C], dtype=float)
    n_norm_sq = np.dot(n, n)
    if n_norm_sq < 1e-12:
        return None

    c0 = np.asarray(center, dtype=float)
    p0 = c0 - ((np.dot(n, c0) - D) / n_norm_sq) * n

    helper = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 * np.sqrt(n_norm_sq) else np.array([0.0, 1.0, 0.0])
    u = np.cross(n, helper)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    v /= np.linalg.norm(v)

    grid = np.linspace(-half_span, half_span, 20)
    s_grid, t_grid = np.meshgrid(grid, grid)

    X = p0[0] + s_grid * u[0] + t_grid * v[0]
    Y = p0[1] + s_grid * u[1] + t_grid * v[1]
    Z = p0[2] + s_grid * u[2] + t_grid * v[2]

    return X, Y, Z, p0


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
st.sidebar.header("3. Display Options")
k_range = st.sidebar.slider("Line Parameter Range (k)", min_value=1.0, max_value=20.0, value=6.0, step=0.5)
vec_scale = st.sidebar.slider("Vector Display Scale", min_value=0.2, max_value=3.0, value=1.0, step=0.1)

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

if case == "unique" and abs(k_star) <= k_range * 2.5:
    scene_center = p_int
    k_min = min(-k_range, k_star - 2.0)
    k_max = max(k_range, k_star + 2.0)
else:
    scene_center = p_line0
    k_min, k_max = -k_range, k_range

k_vals = np.linspace(k_min, k_max, 100)
line_pts = p_line0[:, None] + d_vec[:, None] * k_vals[None, :]

line_span = np.linalg.norm(line_pts[:, -1] - line_pts[:, 0]) * 0.55
half_span = max(6.0, line_span)

X_plane, Y_plane, Z_plane, p_plane_anchor = build_plane_mesh(A, B, C, D, scene_center, half_span)

fig = go.Figure()

fig.add_trace(
    go.Surface(
        x=X_plane,
        y=Y_plane,
        z=Z_plane,
        colorscale=[[0, "#38bdf8"], [1, "#0284c7"]],
        opacity=0.45,
        showscale=False,
        name=f"Plane: {A:g}x + {B:g}y + {C:g}z = {D:g}",
        showlegend=True,
        hovertemplate="Plane<br>x: %{x:.2f}<br>y: %{y:.2f}<br>z: %{z:.2f}<extra></extra>",
    )
)

fig.add_trace(
    go.Scatter3d(
        x=line_pts[0],
        y=line_pts[1],
        z=line_pts[2],
        mode="lines",
        line=dict(color="#f43f5e", width=5),
        name="Parametric Line L(k)",
        customdata=k_vals,
        hovertemplate="Line L(k)<br>x: %{x:.2f}<br>y: %{y:.2f}<br>z: %{z:.2f}<br>k: %{customdata:.2f}<extra></extra>",
    )
)

fig.add_trace(
    go.Scatter3d(
        x=[x0],
        y=[y0],
        z=[z0],
        mode="markers",
        marker=dict(size=6, color="#f8fafc", line=dict(color="#f43f5e", width=2)),
        name=f"Base Point P₀ ({x0:g}, {y0:g}, {z0:g})",
        hovertemplate=f"Base Point P₀ (k=0)<br>({x0:.2f}, {y0:.2f}, {z0:.2f})<extra></extra>",
    )
)

if case == "unique":
    fig.add_trace(
        go.Scatter3d(
            x=[p_int[0]],
            y=[p_int[1]],
            z=[p_int[2]],
            mode="markers+text",
            text=[f"  Intersection ({p_int[0]:.2f}, {p_int[1]:.2f}, {p_int[2]:.2f})"],
            textposition="top center",
            textfont=dict(color="#fbbf24", size=12),
            marker=dict(size=8, color="#fbbf24", symbol="diamond", line=dict(color="#0f172a", width=1.5)),
            name="Intersection Point",
            hovertemplate=(
                f"Intersection (k = {k_star:.4f})<br>"
                f"x: {p_int[0]:.4f}<br>y: {p_int[1]:.4f}<br>z: {p_int[2]:.4f}<extra></extra>"
            ),
        )
    )

normal_origin = p_int if case == "unique" else p_plane_anchor
add_vector_arrow(
    fig,
    start=normal_origin,
    vec=n_vec,
    color="#22d3ee",
    name=f"Normal Vector n = ({A:g}, {B:g}, {C:g})",
    scale=vec_scale,
)

add_vector_arrow(
    fig,
    start=p_line0,
    vec=d_vec,
    color="#a3e635",
    name=f"Direction Vector d = ({a:g}, {b:g}, {c:g})",
    scale=vec_scale,
)

fig.update_layout(
    template="plotly_dark",
    height=650,
    margin=dict(l=0, r=0, t=30, b=0),
    legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
    scene=dict(
        xaxis_title="X",
        yaxis_title="Y",
        zaxis_title="Z",
        aspectmode="data",
    ),
)

st.plotly_chart(fig, use_container_width=True)

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
