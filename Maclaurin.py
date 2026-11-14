import ast
import math
import re
import streamlit as st
import numpy as np
import pandas as pd
import altair as alt

st.set_page_config(page_title="Maclaurin Series Visualizer", layout="wide")


class TaylorJet:
    """Truncated Taylor series arithmetic up to degree N for exact automatic derivatives at x = 0."""

    def __init__(self, coeffs: np.ndarray):
        self.c = np.asarray(coeffs, dtype=float)
        self.n = len(self.c) - 1

    @classmethod
    def constant(cls, val: float, n: int):
        c = np.zeros(n + 1, dtype=float)
        c[0] = float(val)
        return cls(c)

    @classmethod
    def variable(cls, n: int):
        c = np.zeros(n + 1, dtype=float)
        if n >= 1:
            c[1] = 1.0
        return cls(c)

    def _promote(self, other):
        if isinstance(other, TaylorJet):
            return other
        return TaylorJet.constant(float(other), self.n)

    def __add__(self, other):
        o = self._promote(other)
        return TaylorJet(self.c + o.c)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        o = self._promote(other)
        return TaylorJet(self.c - o.c)

    def __rsub__(self, other):
        o = self._promote(other)
        return TaylorJet(o.c - self.c)

    def __neg__(self):
        return TaylorJet(-self.c)

    def __pos__(self):
        return self

    def __mul__(self, other):
        o = self._promote(other)
        res = np.zeros(self.n + 1, dtype=float)
        for k in range(self.n + 1):
            res[k] = np.dot(self.c[: k + 1], o.c[k::-1])
        return TaylorJet(res)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        o = self._promote(other)
        if o.c[0] == 0.0:
            raise ValueError("Division by zero at x = 0.")
        res = np.zeros(self.n + 1, dtype=float)
        inv_b0 = 1.0 / o.c[0]
        for k in range(self.n + 1):
            s = self.c[k] - np.dot(res[:k], o.c[k:0:-1])
            res[k] = s * inv_b0
        return TaylorJet(res)

    def __rtruediv__(self, other):
        return self._promote(other).__truediv__(self)

    def __pow__(self, power):
        if isinstance(power, (int, float)) and float(power).is_integer():
            p = int(power)
            if p == 0:
                return TaylorJet.constant(1.0, self.n)
            if p < 0:
                return TaylorJet.constant(1.0, self.n) / (self ** (-p))
            result = TaylorJet.constant(1.0, self.n)
            base = self
            while p > 0:
                if p & 1:
                    result = result * base
                base = base * base
                p >>= 1
            return result
        o = self._promote(power)
        return (o * self.log()).exp()

    def __rpow__(self, other):
        return self._promote(other).__pow__(self)

    def exp(self):
        res = np.zeros(self.n + 1, dtype=float)
        res[0] = math.exp(self.c[0])
        for k in range(1, self.n + 1):
            j = np.arange(1, k + 1, dtype=float)
            res[k] = np.dot(j * self.c[1 : k + 1], res[k - 1 :: -1]) / k
        return TaylorJet(res)

    def log(self):
        if self.c[0] <= 0.0:
            raise ValueError("ln(u) requires u(0) > 0.")
        res = np.zeros(self.n + 1, dtype=float)
        res[0] = math.log(self.c[0])
        inv_a0 = 1.0 / self.c[0]
        for k in range(1, self.n + 1):
            if k == 1:
                res[1] = self.c[1] * inv_a0
            else:
                j = np.arange(1, k, dtype=float)
                s = np.dot(j * res[1:k], self.c[k - 1 : 0 : -1])
                res[k] = (self.c[k] - s / k) * inv_a0
        return TaylorJet(res)

    def sqrt(self):
        if self.c[0] < 0.0:
            raise ValueError("sqrt(u) requires u(0) >= 0.")
        if self.c[0] == 0.0:
            raise ValueError("sqrt(u) is not differentiable at u(0) = 0.")
        res = np.zeros(self.n + 1, dtype=float)
        res[0] = math.sqrt(self.c[0])
        denom = 2.0 * res[0]
        for k in range(1, self.n + 1):
            s = np.dot(res[1:k], res[k - 1 : 0 : -1])
            res[k] = (self.c[k] - s) / denom
        return TaylorJet(res)

    def sin_cos(self):
        s = np.zeros(self.n + 1, dtype=float)
        c = np.zeros(self.n + 1, dtype=float)
        s[0] = math.sin(self.c[0])
        c[0] = math.cos(self.c[0])
        for k in range(1, self.n + 1):
            j = np.arange(1, k + 1, dtype=float)
            jc = j * self.c[1 : k + 1]
            s[k] = np.dot(jc, c[k - 1 :: -1]) / k
            c[k] = -np.dot(jc, s[k - 1 :: -1]) / k
        return TaylorJet(s), TaylorJet(c)

    def sin(self):
        return self.sin_cos()[0]

    def cos(self):
        return self.sin_cos()[1]

    def tan(self):
        s, c = self.sin_cos()
        return s / c

    def sinh_cosh(self):
        sh = np.zeros(self.n + 1, dtype=float)
        ch = np.zeros(self.n + 1, dtype=float)
        sh[0] = math.sinh(self.c[0])
        ch[0] = math.cosh(self.c[0])
        for k in range(1, self.n + 1):
            j = np.arange(1, k + 1, dtype=float)
            jc = j * self.c[1 : k + 1]
            sh[k] = np.dot(jc, ch[k - 1 :: -1]) / k
            ch[k] = np.dot(jc, sh[k - 1 :: -1]) / k
        return TaylorJet(sh), TaylorJet(ch)

    def sinh(self):
        return self.sinh_cosh()[0]

    def cosh(self):
        return self.sinh_cosh()[1]

    def tanh(self):
        sh, ch = self.sinh_cosh()
        return sh / ch

    def atan(self):
        # d/dx atan(u) = u' / (1 + u^2)
        one = TaylorJet.constant(1.0, self.n)
        denom = one + (self * self)
        du = TaylorJet(np.append(np.arange(1, self.n + 1) * self.c[1:], 0.0))
        quot = du / denom
        res = np.zeros(self.n + 1, dtype=float)
        res[0] = math.atan(self.c[0])
        for k in range(1, self.n + 1):
            res[k] = quot.c[k - 1] / k
        return TaylorJet(res)

    def asin(self):
        if abs(self.c[0]) >= 1.0:
            raise ValueError("asin(u) is not differentiable at |u(0)| >= 1.")
        one = TaylorJet.constant(1.0, self.n)
        denom = (one - (self * self)).sqrt()
        du = TaylorJet(np.append(np.arange(1, self.n + 1) * self.c[1:], 0.0))
        quot = du / denom
        res = np.zeros(self.n + 1, dtype=float)
        res[0] = math.asin(self.c[0])
        for k in range(1, self.n + 1):
            res[k] = quot.c[k - 1] / k
        return TaylorJet(res)

    def acos(self):
        return TaylorJet.constant(math.pi / 2.0, self.n) - self.asin()


def normalize_expression(expr_str: str) -> str:
    """Converts student-friendly math notation (e.g. 2x, e^x) into valid Python syntax."""
    s = expr_str.strip().replace("^", "**")
    # Insert '*' between number and letter/open-paren: 2x -> 2*x, 3(x+1) -> 3*(x+1)
    s = re.sub(r"(\d)([a-zA-Z(])", r"\1*\2", s)
    # Insert '*' between ')' and '(' or letter/digit: )( -> )*(, )x -> )*x
    s = re.sub(r"(\))([a-zA-Z0-9(])", r"\1*\2", s)
    return s


SAFE_NODES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Call,
    ast.Name,
    ast.Constant,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Pow,
    ast.USub,
    ast.UAdd,
    ast.Load,
)


def evaluate_ast(node, env: dict):
    if not isinstance(node, SAFE_NODES):
        raise ValueError(f"Unsupported syntax: {type(node).__name__}")
    if isinstance(node, ast.Expression):
        return evaluate_ast(node.body, env)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise ValueError("Only numeric constants are allowed.")
    if isinstance(node, ast.Name):
        if node.id in env:
            return env[node.id]
        raise ValueError(f"Unknown variable or symbol '{node.id}'. Use 'x'.")
    if isinstance(node, ast.UnaryOp):
        val = evaluate_ast(node.operand, env)
        if isinstance(node.op, ast.USub):
            return -val
        if isinstance(node.op, ast.UAdd):
            return +val
    if isinstance(node, ast.BinOp):
        left = evaluate_ast(node.left, env)
        right = evaluate_ast(node.right, env)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.Pow):
            return left**right
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in env:
            raise ValueError("Unsupported function call.")
        fn = env[node.func.id]
        if not callable(fn):
            raise ValueError(f"'{node.func.id}' is not a function.")
        args = [evaluate_ast(arg, env) for arg in node.args]
        return fn(*args)
    raise ValueError("Invalid mathematical expression.")


def format_polynomial_latex(coeffs: np.ndarray) -> str:
    """Formats Maclaurin coefficients into a clean LaTeX polynomial string."""
    terms = []
    for k, c in enumerate(coeffs):
        if abs(c) < 1e-12:
            continue
        sign = "-" if c < 0 else "+"
        mag = abs(c)
        if k == 0:
            term_body = f"{mag:.4g}"
        elif k == 1:
            coeff_str = "" if math.isclose(mag, 1.0, rel_tol=1e-9) else f"{mag:.4g}"
            term_body = f"{coeff_str}x"
        else:
            coeff_str = "" if math.isclose(mag, 1.0, rel_tol=1e-9) else f"{mag:.4g}"
            term_body = f"{coeff_str}x^{{{k}}}"

        if not terms:
            terms.append(f"-{term_body}" if sign == "-" else term_body)
        else:
            terms.append(f" {sign} {term_body}")

    return "".join(terms) if terms else "0"


def compute_maclaurin_and_eval(expr_str: str, order_n: int, x_vals: np.ndarray):
    py_expr = normalize_expression(expr_str)
    tree = ast.parse(py_expr, mode="eval")

    # 1. Exact Maclaurin coefficients via TaylorJet automatic differentiation at x = 0
    jet_x = TaylorJet.variable(order_n)
    jet_env = {
        "x": jet_x,
        "e": math.e,
        "pi": math.pi,
        "sin": lambda u: u.sin() if isinstance(u, TaylorJet) else math.sin(u),
        "cos": lambda u: u.cos() if isinstance(u, TaylorJet) else math.cos(u),
        "tan": lambda u: u.tan() if isinstance(u, TaylorJet) else math.tan(u),
        "exp": lambda u: u.exp() if isinstance(u, TaylorJet) else math.exp(u),
        "ln": lambda u: u.log() if isinstance(u, TaylorJet) else math.log(u),
        "log": lambda u: u.log() if isinstance(u, TaylorJet) else math.log(u),
        "sqrt": lambda u: u.sqrt() if isinstance(u, TaylorJet) else math.sqrt(u),
        "asin": lambda u: u.asin() if isinstance(u, TaylorJet) else math.asin(u),
        "acos": lambda u: u.acos() if isinstance(u, TaylorJet) else math.acos(u),
        "atan": lambda u: u.atan() if isinstance(u, TaylorJet) else math.atan(u),
        "sinh": lambda u: u.sinh() if isinstance(u, TaylorJet) else math.sinh(u),
        "cosh": lambda u: u.cosh() if isinstance(u, TaylorJet) else math.cosh(u),
        "tanh": lambda u: u.tanh() if isinstance(u, TaylorJet) else math.tanh(u),
    }

    jet_out = evaluate_ast(tree, jet_env)
    if not isinstance(jet_out, TaylorJet):
        jet_out = TaylorJet.constant(float(jet_out), order_n)

    coeffs = jet_out.c
    if not np.all(np.isfinite(coeffs)):
        raise ValueError("f(x) or one of its derivatives is undefined at x = 0.")

    # 2. Evaluate exact f(x) over x_vals using NumPy
    np_env = {
        "x": x_vals,
        "e": np.e,
        "pi": np.pi,
        "sin": np.sin,
        "cos": np.cos,
        "tan": np.tan,
        "exp": np.exp,
        "ln": np.log,
        "log": np.log,
        "sqrt": np.sqrt,
        "asin": np.arcsin,
        "acos": np.arccos,
        "atan": np.arctan,
        "sinh": np.sinh,
        "cosh": np.cosh,
        "tanh": np.tanh,
    }
    with np.errstate(all="ignore"):
        y_exact = evaluate_ast(tree, np_env)
        if np.ndim(y_exact) == 0:
            y_exact = np.full_like(x_vals, float(y_exact))
        else:
            y_exact = np.asarray(y_exact, dtype=float)

    # 3. Evaluate Maclaurin polynomial P_n(x) via Horner's method (coeffs[::-1])
    y_approx = np.polyval(coeffs[::-1], x_vals)

    return coeffs, y_exact, y_approx


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
    help="Supported: sin, cos, tan, exp, ln, log, sqrt, asin, acos, atan, sinh, cosh, tanh, e, pi, +, -, *, /, ^",
)

order_n = st.sidebar.slider("Maclaurin Order (n)", min_value=0, max_value=20, value=3, step=1)
x_max = st.sidebar.slider("Domain Upper Bound (x_max)", min_value=0.5, max_value=15.0, value=6.28, step=0.1)

st.sidebar.divider()
clip_y = st.sidebar.checkbox(
    "Clamp Y-axis to f(x) range",
    value=True,
    help="Prevents high-degree polynomial tails from squashing the view of f(x).",
)

# --- Main Display ---
st.title("Maclaurin Series Visualizer")

if not func_str.strip():
    st.info("Enter a function $f(x)$ in the sidebar to begin.")
    st.stop()

x_vals = np.linspace(0.0, float(x_max), 400)

try:
    coeffs, y_exact, y_approx = compute_maclaurin_and_eval(func_str, order_n, x_vals)
except Exception as e:
    st.error(f"Could not evaluate function at x = 0: {e}")
    st.stop()

# Display Symbolic Maclaurin Polynomial
poly_latex = format_polynomial_latex(coeffs)
st.markdown(f"**Order $n = {order_n}$ Maclaurin Polynomial:**")
st.latex(rf"P_{{{order_n}}}(x) = {poly_latex}")

# Clean infinities at asymptotes (e.g. 1/(1-x) at x = 1)
y_exact[~np.isfinite(y_exact)] = np.nan
y_approx[~np.isfinite(y_approx)] = np.nan

# Endpoint Error Metrics
exact_end = y_exact[-1]
approx_end = y_approx[-1]
abs_error_end = abs(exact_end - approx_end) if np.isfinite(exact_end) and np.isfinite(approx_end) else np.nan

m1, m2, m3 = st.columns(3)
m1.metric(f"Exact f({x_max:.2f})", f"{exact_end:.5f}" if np.isfinite(exact_end) else "Undefined")
m2.metric(f"Approx P_{order_n}({x_max:.2f})", f"{approx_end:.5f}" if np.isfinite(approx_end) else "Undefined")
m3.metric("Absolute Error at x_max", f"{abs_error_end:.5e}" if np.isfinite(abs_error_end) else "N/A")

# Build Plot DataFrame
exact_label = f"Exact: f(x) = {func_str}"
approx_label = f"Maclaurin P_{order_n}(x)"

plot_df = pd.concat(
    [
        pd.DataFrame({"x": x_vals, "y": y_exact, "Curve": exact_label}),
        pd.DataFrame({"x": x_vals, "y": y_approx, "Curve": approx_label}),
    ],
    ignore_index=True,
)

# Y-axis scale
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
                range=["#38bdf8", "#f43f5e"],
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
