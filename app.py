from datetime import date, timedelta
from typing import Optional
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# ----------------------------
# Page setup
# ----------------------------
st.set_page_config(
    page_title="Easy Loan Calculator",
    page_icon="💸",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.title("Easy Loan Calculator")
st.caption("Calculate EMI, repayment schedule, and milestones for your loan")

# ----------------------------
# Helpers
# ----------------------------
PERYEAR = {"Monthly": 12, "Quarterly": 4, "Yearly": 1}

def currency(x: float) -> str:
    return f"₹{x:,.2f}"

def period_label(freq: str) -> str:
    return "Monthly Payment (EMI)" if freq == "Monthly" else "Payment per period"

def pmt(r: float, n: int, p: float, when: str = "end") -> float:
    if n <= 0:
        return 0.0
    pay = p / n if r == 0 else p * r / (1 - (1 + r) ** (-n))
    return pay * (1 - r) if when == "begin" else pay

def amortize(principal: float, rate_annual: float, years: int, freq: str, start: date, when: str) -> pd.DataFrame:
    m = PERYEAR[freq]
    r = (rate_annual / 100.0) / m
    n = int(round(years * m))
    base = pmt(r, n, principal, when=when)

    rows, bal, d, step = [], principal, start, int(round(365 / m))
    for k in range(1, n + 1):
        interest = 0.0 if r == 0 else bal * r
        principal_pay = base if when == "begin" else base - interest
        principal_pay = max(0.0, min(principal_pay, bal))
        total = principal_pay + interest
        bal -= principal_pay
        rows.append(
            {
                "Period": k,
                "Date": d,
                "Payment": round(total, 2),
                "Principal": round(principal_pay, 2),
                "Interest": round(interest, 2),
                "Balance": round(bal, 2),
            }
        )
        d += timedelta(days=step)
        if bal <= 1e-8:
            break
    return pd.DataFrame(rows)

def compute_scenario(cfg: dict):
    principal = max(0.0, (cfg["price"] or 0.0) - (cfg["deposit"] or 0.0))
    m = PERYEAR[cfg["freq"]]
    r = (cfg["rate"] / 100.0) / m
    n = int(round(cfg["years"] * m))
    when = "begin" if cfg["when_begin"] else "end"
    emi = 0.0 if principal == 0 else pmt(r, n, principal, when=when)
    sched = amortize(principal, cfg["rate"], cfg["years"], cfg["freq"], cfg["start_date"], when=when)
    totals = {
        "principal": principal,
        "emi": emi,
        "interest": float(sched["Interest"].sum()) if not sched.empty else 0.0,
        "payment": float(sched["Payment"].sum()) if not sched.empty else 0.0,
        "n": n,
        "when": when,
        "freq": cfg["freq"],
    }
    return totals, sched

# ---------- Milestones (FIXED LABELS) ----------
def _fmt_period_as_ym(period: int, freq: str) -> str:
    """
    Human label for a payment period index (1-based).
    """
    m = PERYEAR[freq]
    if m == 12:
        years = (period - 1) // 12 + 1
        month = (period - 1) % 12 + 1
        return f"Year {years}, Month {month}"
    elif m == 4:
        years = (period - 1) // 4 + 1
        q = (period - 1) % 4 + 1
        return f"Year {years}, Quarter {q}"
    else:  # yearly
        return f"Year {period}"

def compute_milestones(schedule: pd.DataFrame, original_principal: float, freq: str):
    """
    original_principal is the borrowed amount after deposit.
    Returns the FIRST period where each milestone is reached, with date and balance.
    """
    if schedule.empty or original_principal <= 0:
        return {}

    sched = schedule.copy()
    sched["CumPrincipal"] = sched["Principal"].cumsum()

    def _mk(row):
        return dict(
            period=int(row["Period"]),
            date=row["Date"],
            balance=float(row["Balance"]),
            label=_fmt_period_as_ym(int(row["Period"]), freq),
        )

    turning = None
    mask_turn = sched["Principal"] >= sched["Interest"]
    if mask_turn.any():
        turning = _mk(sched.loc[mask_turn].iloc[0])

    half = None
    mask_half = sched["CumPrincipal"] >= 0.5 * original_principal
    if mask_half.any():
        half = _mk(sched.loc[mask_half].iloc[0])

    quarter = None
    mask_q = sched["Balance"] <= 0.25 * original_principal
    if mask_q.any():
        quarter = _mk(sched.loc[mask_q].iloc[0])

    return {"turning": turning, "half": half, "quarter": quarter}
def format_duration(n_periods: int, freq: str) -> str:
    m = PERYEAR[freq]
    years_i, rem = divmod(n_periods, m)
    if freq == "Monthly":
        # e.g., "6 years (72 months)" or "6 years 6 months (78 months)"
        if rem == 0:
            return f"{years_i} years ({n_periods} months)"
        return f"{years_i} years {rem} months ({n_periods} months)"
    elif freq == "Quarterly":
        # e.g., "6 years (24 quarters)" or "6 years 1 quarter (25 quarters)"
        if rem == 0:
            return f"{years_i} years ({n_periods} quarters)"
        return f"{years_i} years {rem} quarters ({n_periods} quarters)"
    else:  # Yearly
        # n_periods equals years
        return f"{n_periods} years"

# ----------------------------
# State (defaults)
# ----------------------------
SINGLE_DEFAULTS = dict(
    name="",
    age=25,
    start_date=date.today(),
    price=0.0,
    deposit=0.0,
    rate=8.5,
    years=20,
    freq="Monthly",
    when_begin=False,
)
A_DEFAULTS = dict(price=0.0, deposit=0.0, rate=8.5, years=20, freq="Monthly", when_begin=False, start_date=date.today())
B_DEFAULTS = dict(price=0.0, deposit=0.0, rate=8.5, years=20, freq="Monthly", when_begin=False, start_date=date.today())

for k, v in SINGLE_DEFAULTS.items():
    st.session_state.setdefault(k, v)
st.session_state.setdefault("submitted", False)
st.session_state.setdefault("compare_mode", False)
st.session_state.setdefault("when_begin", False)
st.session_state.setdefault("A", A_DEFAULTS.copy())
st.session_state.setdefault("B", B_DEFAULTS.copy())

# ----------------------------
# Sync helpers (FIX for edit form)
# ----------------------------
def sync_main_to_form():
    ss = st.session_state
    ss["fs_name"] = ss.get("name", "")
    ss["fs_age"] = int(ss.get("age", 25))
    ss["fs_start_date"] = ss.get("start_date", date.today())
    ss["fs_price"] = float(ss.get("price", 0.0))
    ss["fs_deposit"] = float(ss.get("deposit", 0.0))
    ss["fs_rate"] = float(ss.get("rate", 8.5))
    ss["fs_years"] = int(ss.get("years", 20))
    ss["fs_freq"] = ss.get("freq", "Monthly")
    ss["fs_when_begin"] = bool(ss.get("when_begin", False))

def open_fullscreen_form():
    st.session_state["submitted"] = False
    sync_main_to_form()

# ----------------------------
# Global toggle (ALWAYS visible once)
# ----------------------------
def mode_toggle():
    current = st.session_state.get("compare_mode", False)
    new_val = st.toggle("Compare two loan scenarios?", value=current)
    if new_val != current:
        st.session_state.update({"compare_mode": new_val, "submitted": False})
        st.rerun()

# ----------------------------
# UI pieces
# ----------------------------
def sidebar_single_controls():
    with st.sidebar:
        st.header("Inputs")
        is_submitted = st.session_state.get("submitted", False)
        ss = st.session_state  # shorthand

        # Personal
        if is_submitted:
            st.text_input("Your Name", key="name", value=ss.get("name", ""))
            st.number_input("Age", min_value=18, max_value=100, key="age", value=int(ss.get("age", 25)))
            st.date_input("Start Date", key="start_date", value=ss.get("start_date", date.today()))
        else:
            st.text_input("Your Name", key="name")
            st.number_input("Age", min_value=18, max_value=100, key="age")
            st.date_input("Start Date", key="start_date")

        # Amounts
        c1, c2 = st.columns(2)
        with c1:
            if is_submitted:
                st.number_input(
                    "Asset Price / Loan Target",
                    min_value=0.0,
                    step=50_000.0,
                    format="%.2f",
                    key="price",
                    value=float(ss.get("price", 0.0)),
                )
            else:
                st.number_input("Asset Price / Loan Target", min_value=0.0, step=50_000.0, format="%.2f", key="price")
        with c2:
            if is_submitted:
                st.number_input(
                    "Deposit / Down Payment",
                    min_value=0.0,
                    step=10_000.0,
                    format="%.2f",
                    key="deposit",
                    value=float(ss.get("deposit", 0.0)),
                )
            else:
                st.number_input("Deposit / Down Payment", min_value=0.0, step=10_000.0, format="%.2f", key="deposit")

        # Rate & Tenure
        c3, c4 = st.columns(2)
        with c3:
            if is_submitted:
                st.slider("Interest Rate (APR, %)", 0.0, 30.0, key="rate", value=float(ss.get("rate", 8.5)), step=0.1)
            else:
                st.slider("Interest Rate (APR, %)", 0.0, 30.0, key="rate", step=0.1)
        with c4:
            if is_submitted:
                st.slider("Duration (Years)", 1, 40, key="years", value=int(ss.get("years", 20)), step=1)
            else:
                st.slider("Duration (Years)", 1, 40, key="years", step=1)

        # Frequency & Timing
        freq_options = list(PERYEAR.keys())
        if is_submitted:
            st.selectbox("Repayment Frequency", freq_options, key="freq", index=freq_options.index(ss.get("freq", "Monthly")))
        else:
            st.selectbox("Repayment Frequency", freq_options, key="freq")

        st.toggle("Pay at beginning (advance)?", key="when_begin")

        st.button("Edit in full-screen form", on_click=open_fullscreen_form)

def scenario_form(prefix: str, defaults: dict):
    c = {}
    st.markdown(f"### Scenario {prefix}")
    c["price"] = st.number_input(
        f"{prefix} • Asset Price / Loan Target", 0.0, value=float(defaults["price"]), step=50_000.0, format="%.2f"
    )
    c["deposit"] = st.number_input(
        f"{prefix} • Deposit / Down Payment", 0.0, value=float(defaults["deposit"]), step=10_000.0, format="%.2f"
    )
    r1, r2 = st.columns(2)
    with r1:
        c["rate"] = st.slider(f"{prefix} • Interest Rate (APR, %)", 0.0, 30.0, value=float(defaults["rate"]), step=0.1)
    with r2:
        c["years"] = st.slider(f"{prefix} • Duration (Years)", 1, 40, value=int(defaults["years"]), step=1)
    r3, r4 = st.columns(2)
    with r3:
        c["freq"] = st.selectbox(
            f"{prefix} • Repayment Frequency", list(PERYEAR.keys()), index=list(PERYEAR.keys()).index(defaults["freq"])
        )
    with r4:
        t = st.selectbox(
            f"{prefix} • Payment timing",
            ["End of period (pay after month ends)", "Beginning of period (pay upfront each period)"],
            index=1 if defaults["when_begin"] else 0,
        )
        c["when_begin"] = t.startswith("Beginning")
    c["start_date"] = st.date_input(f"{prefix} • Start Date", value=defaults["start_date"])
    return c

def render_summary_and_table(
    title: str,
    totals: dict,
    sched: pd.DataFrame,
    freq: str,
    milestones: Optional[dict] = None,
    name: Optional[str] = None,
    age: Optional[int] = None,
    start_date: Optional[date] = None,
):
    left, right = st.columns([3, 1])

    with left:
        st.subheader(f"{title} Summary")
        a, b, c = st.columns(3)
        a.metric("Principal", currency(totals["principal"]))
        b.metric(period_label(freq), currency(totals["emi"]) if totals["principal"] > 0 else "—")
        c.metric("Total Interest", currency(totals["interest"]))

        # --- Milestones under the summary ---
        if milestones is not None:
            st.markdown("##### Milestones")
            m1, m2, m3 = st.columns(3)

            def unpack(item):
                if not item:
                    return ("—", None, None)
                lab = item.get("label", "—")
                dt = item.get("date", None)
                bal = item.get("balance", None)
                date_txt = dt.strftime("%b %d, %Y") if hasattr(dt, "strftime") else None
                bal_txt = currency(bal) if isinstance(bal, (int, float)) else None
                return (lab, date_txt, bal_txt)

            lab_turn, date_turn, bal_turn = unpack(milestones.get("turning"))
            lab_half, date_half, bal_half = unpack(milestones.get("half"))
            lab_quar, date_quar, bal_quar = unpack(milestones.get("quarter"))

            with m1:
                st.write("📉 Principal overtakes Interest")
                st.write(f"**{lab_turn}**")
                if date_turn or bal_turn:
                    st.caption(f"{date_turn or ''}{' • ' if date_turn and bal_turn else ''}{('Balance ' + bal_turn) if bal_turn else ''}")

            with m2:
                st.write("🎯 50% of Principal Repaid")
                st.write(f"**{lab_half}**")
                if date_half or bal_half:
                    st.caption(f"{date_half or ''}{' • ' if date_half and bal_half else ''}{('Balance ' + bal_half) if bal_half else ''}")

            with m3:
                st.write("✅ Balance ≤ 25% of Original")
                st.write(f"**{lab_quar}**")
                if date_quar or bal_quar:
                    st.caption(f"{date_quar or ''}{' • ' if date_quar and bal_quar else ''}{('Balance ' + bal_quar) if bal_quar else ''}")

        st.write("---")

    with right:
        st.markdown("#### Details")
        if name:
            st.write(f"**Name:** {name}")
        if age is not None:
            st.write(f"**Age:** {age}")
        if start_date:
            st.write(f"**Start Date:** {start_date.strftime('%b %d, %Y')}")
        st.write(f"**Frequency:** {freq}")
        st.write(f"**Duration:** {format_duration(totals['n'], freq)}") 
        st.write(f"**Payment timing:** {'Beginning of Month' if totals['when']=='begin' else 'End of Month'}")

    st.subheader(f"{title} Amortization Schedule")
    if sched.empty:
        st.info("No schedule (check inputs).")
        return

    # Arrow-friendly display table (Period -> str, Date -> NaT for totals)
    sched_display = sched.copy()
    sched_display["Period"] = sched_display["Period"].astype(str)

    totals_row = pd.DataFrame(
        [
            {
                "Period": "Total",
                "Date": pd.NaT,
                "Payment": round(sched["Payment"].sum(), 2),
                "Principal": round(sched["Principal"].sum(), 2),
                "Interest": round(sched["Interest"].sum(), 2),
                "Balance": 0.00,
            }
        ]
    )
    disp = pd.concat([sched_display, totals_row], ignore_index=True)

    st.dataframe(disp, use_container_width=True, hide_index=True)
    st.download_button(
        f"⬇️ Download {title} CSV",
        data=disp.to_csv(index=False).encode("utf-8"),
        file_name=f"amortization_{title.replace(' ','_').lower()}.csv",
        mime="text/csv",
    )

def render_pie(title: str, totals: dict, size: float = 5.0, radius: float = 0.7):
    if totals["payment"] <= 0:
        return
    fig, ax = plt.subplots(figsize=(size, size))
    parts = [totals["interest"], max(totals["payment"] - totals["interest"], 0.0)]
    wedges, texts, autotexts = ax.pie(parts, labels=["Interest", "Principal"], autopct="%1.1f%%", startangle=90, radius=radius)
    ax.axis("equal")
    ax.set_title(f"{title} • Total Paid Split")
    st.pyplot(fig)

# ----------------------------
# Main flow (inputs)
# ----------------------------
mode_toggle()  # keep at top, once

if not st.session_state["submitted"]:
    if not st.session_state["compare_mode"]:
        st.subheader("Enter loan details")

        # Ensure fs_* reflect the current main values each time we show the form
        # Seed fs_* only if they don't exist (prevents wiping user input on reruns)
        _seed_keys = ["fs_name","fs_age","fs_start_date","fs_price","fs_deposit","fs_rate","fs_years","fs_freq","fs_when_begin"]
        if not all(k in st.session_state for k in _seed_keys):
            sync_main_to_form()


        with st.form("single_form", clear_on_submit=False):
            ss = st.session_state  # shorthand

            # Row 1: Personal
            r1c1, r1c2, r1c3 = st.columns([2, 1, 1])
            with r1c1:
                st.text_input("Your Name", key="fs_name", value=ss.get("fs_name", ""))
            with r1c2:
                st.number_input("Age", min_value=18, max_value=100, key="fs_age", value=int(ss.get("fs_age", 25)))
            with r1c3:
                st.date_input("Start Date", key="fs_start_date", value=ss.get("fs_start_date", date.today()))

            # Row 2: Price & Deposit
            r2c1, r2c2 = st.columns(2)
            with r2c1:
                st.number_input(
                    "Asset Price / Loan Target",
                    min_value=0.0,
                    step=50_000.0,
                    format="%.2f",
                    key="fs_price",
                    value=float(ss.get("fs_price", 0.0)),
                )
            with r2c2:
                st.number_input(
                    "Deposit / Down Payment",
                    min_value=0.0,
                    step=10_000.0,
                    format="%.2f",
                    key="fs_deposit",
                    value=float(ss.get("fs_deposit", 0.0)),
                )

            # Row 3: Rate & Tenure
            r3c1, r3c2 = st.columns(2)
            with r3c1:
                st.slider("Interest Rate (APR, %)", 0.0, 30.0, step=0.1, key="fs_rate", value=float(ss.get("fs_rate", 8.5)))
            with r3c2:
                st.slider("Duration (Years)", 1, 40, step=1, key="fs_years", value=int(ss.get("fs_years", 20)))

            # Row 4: Frequency & Timing
            r4c1, r4c2 = st.columns(2)
            with r4c1:
                opts = list(PERYEAR.keys())
                st.selectbox("Repayment Frequency", opts, key="fs_freq", index=opts.index(ss.get("fs_freq", "Monthly")))
            with r4c2:
                st.toggle("Pay at beginning (advance)?", key="fs_when_begin", value=bool(ss.get("fs_when_begin", False)))

            if st.form_submit_button("Calculate"):
                # copy form values back to main keys in one shot
                ss.update(
                    {
                        "name": ss.get("fs_name", ""),
                        "age": ss.get("fs_age", 25),
                        "start_date": ss.get("fs_start_date", date.today()),
                        "price": ss.get("fs_price", 0.0),
                        "deposit": ss.get("fs_deposit", 0.0),
                        "rate": ss.get("fs_rate", 0.0),
                        "years": ss.get("fs_years", 1),
                        "freq": ss.get("fs_freq", "Monthly"),
                        "when_begin": ss.get("fs_when_begin", False),
                        "submitted": True,
                    }
                )
                st.rerun()

    else:
        # --------- Compare mode form ----------
        st.subheader("Enter two scenarios to compare")
        with st.form("compare_form", clear_on_submit=False):
            colA, colB = st.columns(2, gap="large")
            with colA:
                A = scenario_form("A", st.session_state["A"])
            with colB:
                B = scenario_form("B", st.session_state["B"])

            if st.form_submit_button("Compare"):
                st.session_state["A"] = A
                st.session_state["B"] = B
                st.session_state["submitted"] = True
                st.rerun()

else:
    if not st.session_state["compare_mode"]:
        # Single results: compute/render FIRST, then show sidebar for editing
        cfg = dict(
            price=st.session_state["price"],
            deposit=st.session_state["deposit"],
            rate=st.session_state["rate"],
            years=st.session_state["years"],
            freq=st.session_state["freq"],
            when_begin=st.session_state["when_begin"],
            start_date=st.session_state["start_date"],
        )
        totals, sched = compute_scenario(cfg)

        # Warnings
        if st.session_state["deposit"] > st.session_state["price"]:
            st.warning("Deposit is greater than asset price. Principal set to 0.")
        if totals["principal"] == 0:
            st.info("Principal is zero. Adjust price/deposit to compute a meaningful schedule.")

        # Milestones first, then Summary (pass borrower info)
        ms = compute_milestones(sched, totals["principal"], totals["freq"])
        render_summary_and_table(
            "Loan",
            totals,
            sched,
            totals["freq"],
            milestones=ms,
            name=st.session_state["name"],
            age=st.session_state["age"],
            start_date=st.session_state["start_date"],
        )

        st.subheader("Chart")
        render_pie("Loan", totals, size=4.5, radius=0.6)

        # NOW show sidebar for after-submit editing
        sidebar_single_controls()

    else:
        st.subheader("Comparison Summary")
        A_tot, A_sched = compute_scenario(st.session_state["A"])
        B_tot, B_sched = compute_scenario(st.session_state["B"])

        ca1, ca2, ca3, ca4 = st.columns(4)
        cb1, cb2, cb3, cb4 = st.columns(4)
        ca1.metric("A • Principal", currency(A_tot["principal"]))
        ca2.metric(
            f"A • {period_label(A_tot['freq'])}", currency(A_tot["emi"]) if A_tot["principal"] > 0 else "—"
        )
        ca3.metric("A • Total Interest", currency(A_tot["interest"]))
        ca4.metric("A • Total Paid", currency(A_tot["payment"]))
        cb1.metric("B • Principal", currency(B_tot["principal"]))
        cb2.metric(
            f"B • {period_label(B_tot['freq'])}", currency(B_tot["emi"]) if B_tot["principal"] > 0 else "—"
        )
        cb3.metric("B • Total Interest", currency(B_tot["interest"]))
        cb4.metric("B • Total Paid", currency(B_tot["payment"]))

        if A_tot["payment"] > 0 and B_tot["payment"] > 0:
            diff = A_tot["payment"] - B_tot["payment"]
            if abs(diff) < 1e-6:
                st.info("Both scenarios result in the same total paid.")
            elif diff > 0:
                st.success(f"Scenario **B** saves **{currency(abs(diff))}** compared to Scenario **A**.")
            else:
                st.success(f"Scenario **A** saves **{currency(abs(diff))}** compared to Scenario **B**.")

        st.write("---")
        st.subheader("Charts")
        c1, c2 = st.columns(2)
        with c1:
            render_pie("Scenario A", A_tot, size=5, radius=0.7)
        with c2:
            render_pie("Scenario B", B_tot, size=5, radius=0.7)

        # Milestones under each scenario's Summary in tabs
        st.write("---")
        st.subheader("Schedules")
        tabA, tabB = st.tabs(["Scenario A", "Scenario B"])
        with tabA:
            msA = compute_milestones(A_sched, A_tot["principal"], A_tot["freq"])
            render_summary_and_table("Scenario A", A_tot, A_sched, A_tot["freq"], milestones=msA)
        with tabB:
            msB = compute_milestones(B_sched, B_tot["principal"], B_tot["freq"])
            render_summary_and_table("Scenario B", B_tot, B_sched, B_tot["freq"], milestones=msB)
