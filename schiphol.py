import requests
import pandas as pd
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt

BASE_URL = "https://api.schiphol.nl/public-flights"

headers = {
    "Accept": "application/json",
    "app_id": "709add58",       
    "app_key": "2c6ef2180d93276ba15cec90dab6143d",
    "ResourceVersion": "v4"
}

# Tip: filter desgewenst op richting of tijdvenster (bv. alleen departures 'D' of arrivals 'A')
params = {
    "includedelays": "false",
    "page": 1,
    "sort": "+scheduleTime",
    "flightDirection": "D",             # optioneel: 'D' (vertrek) of 'A' (aankomst)
    "fromDateTime": "2025-09-22T00:00:00",
    "toDateTime":   "2025-09-22T23:59:59",
}

flights = []
page = 0

while len(flights) < 20:
    params["page"] = page
    resp = requests.get(f"{BASE_URL}/flights", headers=headers, params=params)
    resp.raise_for_status()
    data = resp.json()

    chunk = data.get("flights", [])
    if not chunk:
        break  # geen extra resultaten meer

    flights.extend(chunk)
    page += 1

# Helper om veilig waarden uit het Schiphol-object te halen
def safe_first(lst, default=""):
    return (lst[0] if isinstance(lst, list) and lst else default)

def bestemming_of_herkomst(f):
    code = (f.get("route", {}) or {}).get("destinations") or []
    code = code[0] if code else ""
    if f.get("flightDirection") == "D":
        label = "bestemming"
    else:
        label = "herkomst"
    return label, code

rows = []
for f in flights:
    label, code = bestemming_of_herkomst(f)
    rows.append({
        "vlucht": f.get("flightName") or f.get("prefixIATA","")+str(f.get("flightNumber","")),
        "richting": f.get("flightDirection"),
        "datum": f.get("scheduleDate"),
        "tijd (gepland)": f.get("scheduleTime"),
        label: code,           # geeft kolom 'bestemming' of 'herkomst'
        "gate": f.get("gate",""),
        "terminal": f.get("terminal",""),
        "status": ", ".join(f.get("publicFlightState", {}).get("flightStates", []))
    })

df = pd.DataFrame(rows)

# Netjes sorteren op geplande tijd
df = df.sort_values(["datum", "tijd (gepland)"], kind="stable").reset_index(drop=True)

#plt.hist(df["tijd (gepland)"], bins=100)
#plt.show()

fig1 = px.histogram(df, x="tijd (gepland)", labels={'tijd (gepland)':'Tijd (gepland)'})
fig1.update_yaxes(title_text="Aantal vluchten")
fig1.show()

fig2 = px.histogram(df, x="gate", labels={'gate':'Gate'})
fig2.update_yaxes(title_text="Aantal vluchten")
fig2.show()
# Tabel tonen
print(df.to_string(index=False))
print(len(df))

st.plotly_chart(fig1, use_container_width=True)



