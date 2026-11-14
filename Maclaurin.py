import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor,
)

st.set_page_config(page_title="Maclaurin Series Visualizer", layout="wide")

# Allow student-friendly math input like "2x", "sin(x)", "e^x", "x^3"
TRANSFORMATIONS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
)
X_SYM = sp.Symbol("x", real=True)
LOCAL_DICT = {
    "x": X_SYM,
    "e": sp.E,
    "pi": sp.pi,
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "exp": sp.exp,
    "ln": sp.log,
    "log": sp.log,
    "sqrt": sp.sqrt,
    "asin": sp.asin,
    "acos": sp.acos,
    "atan": sp.atan,
    "sinh": sp.sinh,
    "cosh": sp.cosh,
    "tanh": sp.tanh,
}


def compute_maclaurin(expr_str: str, order_n: int):
    """Parses f(x) and computes its symbolic Maclaurin polynomial of degree n."""
    expr = parse_expr(expr_str, local_dict=LOCAL_DICT, transformations=TRANSFORMATIONS)

    # Validate single-variable dependence
    extra_symbols = expr.free_symbols - {X_SYM}
    if extra_symbols:
        names = ", ".join(str(s) for s in extra_symbols)
        raise ValueError(f"Only variable 'x' is supported (found: {names}).")

    # Compute Maclaurin polynomial: sum_{k=0}^{n} (f^(k)(0) / k!) * x^k
    poly = sp.Integer(0)
    deriv = expr
    for k in range(order_n + 1):
        val_at_0 = sp.simplify(deriv.subs(X_SYM, 0))
        if not val_at_0.is_finite:
            raise ValueError(f"f(x) or its {k}-th derivative is undefined at x = 0.")
        poly += (val_at_0 / sp.factorial(k)) * (X_SYM**k)
        if k < order_n:
            deriv = sp.diff(deriv, X_SYM)

    poly = sp.expand(poly)

    # Convert symbolic expressions to fast NumPy callables
    f_num = sp.lambdify(X_SYM, expr, modules=["numpy"])
    p_num = sp.lambdify(X_SYM, poly, modules=["numpy"])

    return expr, poly, f_num, p_num


# --- Sidebar Controls ---
st.sidebar.header("Parameters")

presets = {
    "Custom": None,
    "sin(x)": "sin(x)",
    "cos(x)": "cos(x)",
    "e^x": "exp(x)",
    "ln(1 + x)": "ln(1 + x)",
    "1 / (1 - x)": "1 / (1 - x)",
    "arctan(x)": "atan(x)",
    "e^(-x^2) (Gaussian)": "exp(-x^2)",
}

selected_preset = st.sidebar.selectbox("Example Function Presets", list(presets.keys()), index=1)

if "func_input" not in st.session_state:
    st.session_state.func_input = "sin(x)"
if "last_preset" not in st.session_state:
    st.session_state.last_preset = selected_preset

if selected_preset != st.session_state.last_preset:
    st.session_state.last_preset = selected_preset
    if presets[selected_preset] is not None:
        st.session_state.func_input = presets[selected_preset]

func_str = st.sidebar.text_input(
    "Type f(x)",
    key="func_input",
    help="Examples: sin(x), cos(2x), exp(x), ln(1+x), 1/(1-x), sqrt(1+x)",
)

order_n = st.sidebar.slider("Maclaurin Order (n)", min_value=0, max_value=20, value=3, step=1)
x_max = st.sidebar.slider("Domain Upper Bound (x_max)", min_value=0.5, max_value=15.0, value=6.28, step=0.1)

st.sidebar.divider()
clip_y = st.sidebar.checkbox("Clamp Y-axis to f(x) range", value=True, help="Prevents high-degree polynomials from squashing the view of f(x).")

# --- Main Display ---
st.title("Maclaurin Series Visualizer")

if not func_str.strip():
    st.info("Enter a function $f(x)$ in the sidebar to begin.")
    st.stop()

try:
    expr_sym, poly_sym, f_num, p_num = compute_maclaurin(func_str, order_n)
except Exception as e:
    st.error(f"Could not evaluate Maclaurin series: {e}")
    st.stop()

# Display Symbolic Formula & Polynomial
col_f, col_p = st.columns(2)
with col_f:
    st.markdown("**Exact Function:**")
    st.latex(rf"f(x) = {sp.latex(expr_sym)}")
with col_p:
    st.markdown(f"**Order $n = {order_n}$ Maclaurin Polynomial:**")
    st.latex(rf"P_{{{order_n}}}(x) = {sp.latex(poly_sym)}")

# Evaluate over [0, x_max]
x_vals = np.linspace(0.0, float(x_max), 400)

with np.errstate(all="ignore"):
    y_exact = np.array(f_num(x_vals), dtype=float)
    y_approx = np.array(p_num(x_vals), dtype=float)

# Handle constant functions where lambdify returns a scalar
if y_exact.ndim == 0:
    y_exact = np.full_like(x_vals, float(y_exact))
if y_approx.ndim == 0:
    y_approx = np.full_like(x_vals, float(y_approx))

# Replace infinities with NaN so lines break cleanly at asymptotes (e.g., 1/(1-x) at x=1)
y_exact[~np.isfinite(y_exact)] = np.nan
y_approx[~np.isfinite(y_approx)] = np.nan

# Endpoint Error Metric
exact_end = y_exact[-1]
approx_end = y_approx[-1]
abs_error_end = abs(exact_end - approx_end) if np.isfinite(exact_end) and np.isfinite(approx_end) else np.nan

m1, m2, m3 = st.columns(3)
m1.metric(f"Exact f({x_max:.2f})", f"{exact_end:.5f}" if np.isfinite(exact_end) else "Undefined")
m2.metric(f"Approx P_{order_n}({x_max:.2f})", f"{approx_end:.5f}" if np.isfinite(approx_end) else "Undefined")
m3.metric(f"Absolute Error at x_max", f"{abs_error_end:.5e}" if np.isfinite(abs_error_end) else "N/A")

# Build Plot DataFrame
exact_label = f"Exact: f(x) = {func_str}"
approx_label = f"Maclaurin P_{order_n}(x)"

df_exact = pd.DataFrame({"x": x_vals, "y": y_exact, "Curve": exact_label})
df_approx = pd.DataFrame({"x": x_vals, "y": y_approx, "Curve": approx_label})
plot_df = pd.concat([df_exact, df_approx], ignore_index=True)

# Determine Y-axis scale
valid_exact = y_exact[np.isfinite(y_exact)]
if clip_y and len(valid_exact) > 0:
    y_min, y_max = float(np.min(valid_exact)), float(np.max(valid_exact))
    span = max(1.0, y_max - y_min)
    y_scale = alt.Scale(domain=[y_min - 0.25 * span, y_max + 0.25 * span], clamp=True)
else:
    y_scale = alt.Scale(zero=False)

chart = (
    alt.Chart(plot_df)
    .mark_line(strokeWidth=3, clip=True)
    .encode(
        x=alt.X("x:Q", title="x", scale=alt.Scale(domain=[0, float(x_max)])),
        y=alt.Y("y:Q", title="y", scale=y_scale),
        color=alt.Color(
            "Curve:N",
            scale=alt.Scale(
                domain=[exact_label, approx_label],
                range=["#38bdf8", "#f43f5e"],  # Cyan for f(x), Rose-Red for P_n(x)
            ),
            legend=alt.Legend(orient="top", title=None),
        ),
        strokeDash=alt.condition(
            alt.datum.Curve == approx_label,
            alt.value([6, 4]),
            alt.value([1, 0]),
        ),
        tooltip=[
            alt.Tooltip("Curve:N", title="Function"),
            alt.Tooltip("x:Q", title="x", format=".4f"),
            alt.Tooltip("y:Q", title="y", format=".5f"),
        ],
    )
    .properties(height=520)
    .interactive()
)

st.altair_chart(chart, use_container_width=True)
