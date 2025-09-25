import datetime as dt
import time
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ====== Config ======
APP_ID   = "759513a3"
APP_KEY  = "584093599c649f8eac78b2c00c6f2c0a"
BASE_URL = "https://api.schiphol.nl/public-flights/flights"

# Kies ophaalmethode
USE_WINDOW = True          # True = tijdvenster (24u terug, 6u vooruit). False = per dag.
DIRECTIONS = ["D", "A"]    # "D" = departures, "A" = arrivals

AIRLINE_MAP = {
    "KL": "KLM", "HV": "Transavia", "U2": "easyJet", "EJU": "easyJet Europe",
    "LH": "Lufthansa", "AF": "Air France", "BA": "British Airways", "LX": "SWISS",
    "OS": "Austrian", "SK": "SAS", "AY": "Finnair", "IB": "Iberia", "VY": "Vueling",
    "UX": "Air Europa", "TP": "TAP Air Portugal", "SN": "Brussels Airlines",
    "FR": "Ryanair", "W6": "Wizz Air", "PC": "Pegasus", "TK": "Turkish Airlines",
    "QR": "Qatar Airways", "EK": "Emirates", "ET": "Ethiopian Airlines",
    "SQ": "Singapore Airlines", "CX": "Cathay Pacific", "CI": "China Airlines",
    "KE": "Korean Air", "NH": "ANA", "JL": "Japan Airlines", "VS": "Virgin Atlantic",
    "DL": "Delta Air Lines", "UA": "United Airlines", "AA": "American Airlines",
    "AC": "Air Canada", "OR": "TUI fly Netherlands", "QY": "DHL (EAT Leipzig)",
    "MP": "Martinair Cargo", "CD": "Corendon Dutch Airlines",
    "MB": "MNG Airlines Cargo"
}

# ====== Tijdvenster (laatste 24 uur t/m 6 uur vooruit) ======
now = dt.datetime.now(dt.UTC)
time_from = (now - dt.timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%S")
time_to   = (now + dt.timedelta(hours=6)).strftime("%Y-%m-%dT%H:%M:%S")

headers = {
    "ResourceVersion": "v4",
    "Accept": "application/json",
    "app_id": APP_ID,
    "app_key": APP_KEY,
}

# Basis params voor window-mode
params_window = {
    "includedelays": "true",
    "page": 0,
    "sort": "+scheduleTime",
    "fromDateTime": time_from,
    "toDateTime": time_to,
    # "flightDirection": "A" / "D" wordt in de fetch-functie gezet
}

# === HTTP helper met backoff ===
def get_with_retries(url, headers, params, tries=8, timeout=30):
    """HTTP GET met exponential backoff; respecteert Retry-After bij 429."""
    wait = 0.5
    for attempt in range(1, tries + 1):
        r = requests.get(url, headers=headers, params=params, timeout=timeout)
        if r.status_code == 200:
            return r

        if r.status_code == 429:
            ra = r.headers.get("Retry-After")
            if ra:
                try:
                    sleep_s = float(ra)
                except ValueError:
                    sleep_s = wait
            else:
                sleep_s = wait
            print(f"[429] rate-limited op page={params.get('page')} → slapen {sleep_s:.1f}s (poging {attempt}/{tries})")
            time.sleep(sleep_s)
            wait = min(wait * 2, 16)
            continue

        if r.status_code in (500, 502, 503, 504):
            print(f"[{r.status_code}] server error op page={params.get('page')} → slapen {wait:.1f}s (poging {attempt}/{tries})")
            time.sleep(wait)
            wait = min(wait * 2, 16)
            continue

        raise RuntimeError(f"HTTP {r.status_code} voor params={params} :: {r.text[:300]}")
    raise RuntimeError(f"Failed after retries: {params}")

# === Ophalen binnen tijdvenster ===
def fetch_window(params_base: dict, directions=("D","A"), max_pages_per_dir=2000):
    """Haal alle pagina's op binnen een tijdvenster; stopt op links.next ontbreekt of lege page."""
    all_flights = []
    for direction in directions:
        p = params_base.copy()
        p["page"] = 0
        p["flightDirection"] = direction
        while p["page"] < max_pages_per_dir:
            try:
                resp = get_with_retries(BASE_URL, headers, p)
            except RuntimeError as e:
                print(f"[WARN] stop {direction} op page={p['page']} door error: {e}")
                break

            data = resp.json() or {}
            chunk = data.get("flights", [])
            if not chunk:
                print(f"[INFO] {direction}: lege page op {p['page']} → klaar.")
                break

            all_flights.extend(chunk)

            links = data.get("links", [])
            has_next = any(l.get("rel") == "next" for l in links)
            if not has_next:
                print(f"[INFO] {direction}: geen next-link na page {p['page']} → klaar.")
                break

            p["page"] += 1
            time.sleep(0.15)
    return all_flights

# === Ophalen per kalenderdag ===
def fetch_days(n_days: int, directions=("D","A")):
    """Haal per dag (scheduleDate) alle pagina's op voor n_days en richtingen."""
    all_flights = []
    dates = [dt.date.today() - dt.timedelta(days=i) for i in range(n_days)]
    for day in dates:
        for direction in directions:
            page = 0
            while True:
                p = {
                    "scheduleDate": str(day),
                    "flightDirection": direction,
                    "sort": "+scheduleTime",
                    "page": page
                }
                resp = get_with_retries(BASE_URL, headers, p)
                data = resp.json() or {}
                chunk = data.get("flights", [])
                if not chunk:
                    break
                all_flights.extend(chunk)
                page += 1
                time.sleep(0.12)
    return all_flights

# === Tijd casting helper ===
def ensure_series_datetime(df: pd.DataFrame, colname: str) -> pd.Series:
    """Altijd datetime64[ns, UTC] Series."""
    if colname not in df.columns:
        return pd.Series([pd.NaT] * len(df), index=df.index, dtype="datetime64[ns, UTC]")
    return pd.to_datetime(df[colname], errors="coerce", utc=True)

# === DataFrame bouwen + features ===
def build_dataframe(all_flights: list) -> pd.DataFrame:
    df = pd.json_normalize(all_flights)

    # Geplande tijd
    df["scheduled_dt"] = ensure_series_datetime(df, "scheduleDateTime")

    # ARRIVALS
    arr_actual   = ensure_series_datetime(df, "actualLandingTime")
    arr_est      = ensure_series_datetime(df, "estimatedLandingTime")
    arr_expected = ensure_series_datetime(df, "expectedTimeOnBelt")  # fallback
    arrival_best = arr_actual.fillna(arr_est).fillna(arr_expected)

    # DEPARTURES
    dep_actual   = ensure_series_datetime(df, "actualOffBlockTime")
    dep_est      = ensure_series_datetime(df, "estimatedOffBlockTime")
    dep_expected = ensure_series_datetime(df, "expectedTimeGate")    # fallback
    departure_best = dep_actual.fillna(dep_est).fillna(dep_expected)

    # Combineer arrivals/departures
    flight_dir = df.get("flightDirection")
    if flight_dir is None:
        flight_dir = pd.Series(["D"] * len(df), index=df.index)
    df["best_actual_dt"] = np.where(flight_dir.eq("A"), arrival_best, departure_best)

    # Vertraging in minuten
    diff = (df["best_actual_dt"] - df["scheduled_dt"])
    df["delay_minutes"] = (diff.dt.total_seconds() / 60).round(0)

    # Airline naam kolom
    if "prefixIATA" in df.columns:
        df["airline_name"] = df["prefixIATA"].map(AIRLINE_MAP).fillna(df["prefixIATA"])
    else:
        df["airline_name"] = "Onbekend"

    # Cancel-flag (CNX)
    def has_cnx(states):
        if isinstance(states, list):
            return any(s == "CNX" for s in states)
        if pd.isna(states):
            return False
        return "CNX" in str(states)

    if "publicFlightState.flightStates" in df.columns:
        df["is_cancelled"] = df["publicFlightState.flightStates"].apply(has_cnx)
    else:
        df["is_cancelled"] = False

    # Handige datum (Europe/Amsterdam)
    df["scheduled_local"] = df["scheduled_dt"].dt.tz_convert("Europe/Amsterdam")
    df["day"] = df["scheduled_local"].dt.date

    # --- Aircraft type (robuust opbouwen) ---
    def coalesce_cols(df, cols):
        s = pd.Series(pd.NA, index=df.index, dtype="object")
        for c in cols:
            if c in df.columns:
                s = s.fillna(df[c])
        return s

    df["aircraft_type"] = (
        coalesce_cols(df, [
            "aircraftType.iataMain", "aircraftType.iataSub",
            "aircraftType.icaoMain", "aircraftType.icaoSub",
            "aircraftType"  # fallback
        ])
        .astype(str).str.strip().str.upper()
        .replace({"NAN": np.nan, "NONE": np.nan, "": np.nan})
    )

    return df

# === Main ===
def main():
    print("Ophalen van flights...")
    if USE_WINDOW:
        flights = fetch_window(params_window, DIRECTIONS)
    else:
        flights = fetch_days(N_DAYS, DIRECTIONS)

    if not flights:
        print("Geen data ontvangen in dit venster.")
        return

    df = build_dataframe(flights)
    print(f"Totaal vluchten: {len(df)}  |  Unieke airlines: {df['airline_name'].nunique()}")

    # ====== Analyses / Visualisaties (één figuur met knoppen) ======
    viz = df.dropna(subset=["delay_minutes"]).copy()

    # --- 1) Histogram per airline (top 10) ---
    top_airlines = viz["airline_name"].value_counts().head(10).index
    hist_airline_traces = []
    for a in top_airlines:
        hist_airline_traces.append(go.Histogram(
            x=viz.loc[viz["airline_name"] == a, "delay_minutes"],
            name=str(a), opacity=0.6, nbinsx=40
        ))

    # --- 2) Bar: gemiddelde vertraging per airline ---
    df_airline = (
        viz.groupby("airline_name", dropna=False)["delay_minutes"]
        .mean().reset_index()
    )
    bar_airline_trace = go.Bar(
        x=df_airline["airline_name"],
        y=df_airline["delay_minutes"],
        name="Gemiddelde vertraging (min)"
    )

    # --- 3) Line: cancel-rate per dag ---
    daily_cancel = (
        df.groupby("day")
        .agg(flights=("is_cancelled", "size"), cancelled=("is_cancelled", "sum"))
        .reset_index()
    )
    daily_cancel["cancel_rate_%"] = (daily_cancel["cancelled"] / daily_cancel["flights"] * 100).round(1)
    line_cancel_trace = go.Scatter(
        x=daily_cancel["day"], y=daily_cancel["cancel_rate_%"],
        mode="lines+markers", name="Cancel-rate (%)"
    )

    # --- 4) Histogram per vliegtuigtype (top 8) ---
    vt = df.dropna(subset=["delay_minutes", "aircraft_type"]).copy()
    top_types = vt["aircraft_type"].value_counts().head(8).index
    hist_type_traces = []
    for t in top_types:
        hist_type_traces.append(go.Histogram(
            x=vt.loc[vt["aircraft_type"] == t, "delay_minutes"],
            name=str(t), opacity=0.55, nbinsx=50
        ))

    # --- 5) Pie: aandeel vertraagde vluchten per vliegtuigtype ---
    delayed = vt[vt["delay_minutes"] > 0].copy()
    pie_trace = None
    if not delayed.empty:
        counts = delayed["aircraft_type"].value_counts().reset_index()
        counts.columns = ["aircraft_type", "delayed_flights"]
        keep = counts.head(10)
        other_sum = counts["delayed_flights"][10:].sum()
        if other_sum > 0:
            keep = pd.concat(
                [keep, pd.DataFrame([{"aircraft_type": "Overig", "delayed_flights": other_sum}])],
                ignore_index=True
            )
        pie_trace = go.Pie(labels=keep["aircraft_type"], values=keep["delayed_flights"], name="Pie vertraagd")

    # === Figuur opbouwen ===
    fig = go.Figure()

    # volgorde: [hist_airline_traces] + [bar_airline_trace] + [line_cancel_trace] + [hist_type_traces] + [pie_trace]
    trace_groups = []

    # group 0: histogram per airline
    for tr in hist_airline_traces:
        fig.add_trace(tr)
    trace_groups.append(list(range(len(fig.data))))  # indices tot nu toe

    # group 1: bar airline mean
    idx_bar = len(fig.data)
    fig.add_trace(bar_airline_trace)
    trace_groups.append([idx_bar])

    # group 2: line cancel rate
    idx_line = len(fig.data)
    fig.add_trace(line_cancel_trace)
    trace_groups.append([idx_line])

    # group 3: histogram per aircraft type
    start_idx_types = len(fig.data)
    for tr in hist_type_traces:
        fig.add_trace(tr)
    trace_groups.append(list(range(start_idx_types, len(fig.data))))

    # group 4: pie per aircraft type
    idx_pie = None
    if pie_trace is not None:
        idx_pie = len(fig.data)
        fig.add_trace(pie_trace)
        trace_groups.append([idx_pie])
    else:
        trace_groups.append([])  # leeg als geen pie

    total_traces = len(fig.data)

    def visible_mask(group_ix):
        mask = [False] * total_traces
        for i in trace_groups[group_ix]:
            mask[i] = True
        return mask

    # ====== View-metadata: titel + assen per view (incl. as-typen) ======
    VIEW_META = {
        0: {"title": "Verdeling van vertragingen per airline (top 10)",
            "x": "Vertraging (minuten)", "y": "Aantal vluchten",
            "x_type": "linear", "y_type": "linear", "barmode": "overlay"},
        1: {"title": "Gemiddelde vertraging per airline",
            "x": "Airline", "y": "Gemiddelde vertraging (minuten)",
            "x_type": "category", "y_type": "linear", "barmode": "group"},
        2: {"title": "Cancel-rate per dag (%)",
            "x": "Dag", "y": "Cancel-rate (%)",
            "x_type": "date", "y_type": "linear"},
        3: {"title": "Verdeling van vertragingen per vliegtuigtype (top 8)",
            "x": "Vertraging (minuten)", "y": "Aantal vluchten",
            "x_type": "linear", "y_type": "linear", "barmode": "overlay"},
        4: {"title": "Vertraagde vluchten (>0 min) per vliegtuigtype",
            "x": "", "y": "", "x_type": "linear", "y_type": "linear"}
    }

    # ---------- Slider builders per view ----------
    # View 0: histogram per airline → max vertraging (SLIDER BLIJFT)
    airlines_order = list(top_airlines)  # zelfde volgorde als je traces
    delay_by_airline = {a: viz.loc[viz["airline_name"] == a, "delay_minutes"] for a in airlines_order}
    max_delay_all = int(np.nanmax(viz["delay_minutes"])) if len(viz) else 0

    def sliders_for_view0():
        steps = []
        for m in range(0, max_delay_all + 10, 10):
            xs = [delay_by_airline[a][delay_by_airline[a] <= m] for a in airlines_order]
            steps.append(dict(
                method="update",
                args=[{"x": [x for x in xs]}],  # 1 x-array per histogram-trace
                label=f"≤ {m} min"
            ))
        return [dict(active=0, currentvalue={"prefix": "Max vertraging: "}, steps=steps)]

    # View 1: bar → GEEN SLIDER
    def sliders_for_view1():
        return []

    # View 2: line → GEEN SLIDER (ook geen Plotly rangeslider)
    def sliders_for_view2():
        return []

    # View 3: histogram per aircraft type → max vertraging (SLIDER BLIJFT)
    types_order = list(top_types)
    delay_by_type = {t: vt.loc[vt["aircraft_type"] == t, "delay_minutes"] for t in types_order}
    max_delay_types = int(np.nanmax(vt["delay_minutes"])) if len(vt) else 0

    def sliders_for_view3():
        steps = []
        for m in range(0, max_delay_types + 10, 10):
            xs = [delay_by_type[t][delay_by_type[t] <= m] for t in types_order]
            steps.append(dict(
                method="update",
                args=[{"x": [x for x in xs]}],
                label=f"≤ {m} min"
            ))
        return [dict(active=0, currentvalue={"prefix": "Max vertraging: "}, steps=steps)]

    # View 4: pie → GEEN SLIDER
    def sliders_for_view4():
        return []

    # central dispatcher
    def sliders_for(view_ix):
        return {
            0: sliders_for_view0,
            1: sliders_for_view1,
            2: sliders_for_view2,
            3: sliders_for_view3,
            4: sliders_for_view4,
        }[view_ix]()

    # ---------- Layout helper (zet ook as-typen en (un)visible rangeslider) ----------
    def layout_args_for(view_ix, showlegend=True):
        m = VIEW_META[view_ix]
        args = {
            "title": m["title"],
            "showlegend": showlegend,
            "xaxis": {
                "title": {"text": m["x"]},
                "type": m["x_type"],
                "autorange": True,
                # GEEN ingebouwde rangeslider meer
                "rangeslider": {"visible": False},
            },
            "yaxis": {"title": {"text": m["y"]}, "type": m["y_type"], "autorange": True},
            "sliders": sliders_for(view_ix)  # leeg (= geen slider) voor views 1,2,4
        }
        if "barmode" in m:
            args["barmode"] = m["barmode"]
        return args

    # ---------- Knoppen ----------
    buttons = [
        dict(label="Histogram: vertraging per airline",
             method="update",
             args=[{"visible": visible_mask(0)}, layout_args_for(0, showlegend=True)]),
        dict(label="Bar: gemiddelde vertraging per airline",
             method="update",
             args=[{"visible": visible_mask(1)}, layout_args_for(1, showlegend=False)]),
        dict(label="Lijn: cancel-rate per dag",
             method="update",
             args=[{"visible": visible_mask(2)}, layout_args_for(2, showlegend=False)]),
        dict(label="Histogram: vertraging per vliegtuigtype",
             method="update",
             args=[{"visible": visible_mask(3)}, layout_args_for(3, showlegend=True)]),
        dict(label="Pie: vertraagde vluchten per type",
             method="update",
             args=[{"visible": visible_mask(4)}, layout_args_for(4, showlegend=True)]),
    ]

    # Standaard layout (view 0) – zet ook init-sliders
    fig.update_layout(
        title=VIEW_META[0]["title"],
        xaxis_title=VIEW_META[0]["x"],
        yaxis_title=VIEW_META[0]["y"],
        xaxis=dict(type=VIEW_META[0]["x_type"], rangeslider={"visible": False}),
        yaxis=dict(type=VIEW_META[0]["y_type"]),
        sliders=sliders_for(0),  # alleen voor view 0
        updatemenus=[dict(type="dropdown", direction="down", showactive=True, x=0, y=1.15, buttons=buttons)],
        bargap=0.05
    )
    fig.update_xaxes(autorange=True)
    fig.update_yaxes(autorange=True)

    # Start: alleen view 0 zichtbaar
    fig.update_traces(visible=False)
    for i in trace_groups[0]:
        fig.data[i].visible = True

    fig.show()

    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
