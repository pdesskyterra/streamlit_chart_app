import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
from matplotlib.patches import Patch
from notion_client import Client
import ast

# --- CONFIG ---
NOTION_TOKEN = st.secrets["NOTION_TOKEN"]
DATABASE_ID  = st.secrets["DATABASE_ID"]

# --- PAGE SETUP ---
st.set_page_config(layout="wide")
st.title("📊 Profit & Expense Tracker")

# --- FETCH & EXPAND NOTION DATA ---
@st.cache_data(ttl=600)
def fetch_notion_data():
    notion = Client(auth=NOTION_TOKEN)
    rows = []
    resp = notion.databases.query(database_id=DATABASE_ID)

    for r in resp["results"]:
        p = r["properties"]

        # 1) Client list
        raw = p.get("Client",{}).get("formula",{}).get("string","")
        clients = [c.strip() for c in raw.split(",") if c.strip()]

        # 2) Expense Category rollup → ['Paid','Potential',…]
        exp_entries = p.get("Expense Category",{}).get("rollup",{}).get("array",[])
        cats = []
        for e in exp_entries:
            if e.get("type") == "select":
                cats.append(e["select"]["name"])
            elif t := e.get("text"):
                cats.append(t.get("content",""))
        is_potential = any(c.lower()=="potential" for c in cats)

        # 3) Calculated Revenue (formula)
        calc_rev = p.get("Calculated Revenue",{}).get("formula",{}).get("number",0)

        # 4) Costs
        emp_tot = p.get("Monthly Employee Cost",{}).get("formula",{}).get("number",0)
        ovh_tot = p.get("Overhead Costs",{}).get("number",0)

        # 5) Month
        month = p.get("Month",{}).get("select",{}).get("name")
        if not month or not clients:
            continue

        # 6) Distribute
        n = len(clients)
        rev_pc = calc_rev/n
        emp_pc = emp_tot/n
        ovh_pc = ovh_tot/n

        for c in clients:
            rows.append({
                "Month": month,
                # assign either to Paid or to Potential
                "Paid Revenue": 0 if is_potential else rev_pc,
                "Potential Revenue": rev_pc if is_potential else 0,
                "Monthly Employee Cost": emp_pc,
                "Overhead Costs": ovh_pc,
                "Client": c
            })

    return pd.DataFrame(rows)

df = fetch_notion_data()
if df.empty:
    st.warning("No data found or invalid Notion credentials.")
    st.stop()

# --- COMMON PROCESSING ---
month_order = [
    'February 2025','March 2025','April 2025',
    'May 2025','June 2025','July 2025','August 2025'
]
df['Month'] = pd.Categorical(df['Month'], categories=month_order, ordered=True)
df = df.sort_values(['Month','Client'])

# aggregates
monthly = df.groupby("Month")[
    ["Paid Revenue","Potential Revenue","Monthly Employee Cost","Overhead Costs"]
].sum().reindex(month_order)
monthly["Total Expenses"] = monthly["Monthly Employee Cost"] + monthly["Overhead Costs"]
monthly["Profit"]        = monthly["Paid Revenue"] - monthly["Total Expenses"]
monthly["Profit (Pot)"]  = monthly["Potential Revenue"] - monthly["Total Expenses"]

# plotting setup
clients = sorted(df['Client'].unique())
colors  = dict(zip(clients, plt.cm.tab20(np.linspace(0,1,len(clients)))))
x = np.arange(len(month_order))
width = 0.35

# --- STREAMLIT TABS ---
tab1, tab2 = st.tabs(["📊 Bar Chart","📈 Line Chart"])

with tab1:
    fig, ax = plt.subplots(figsize=(22,12))

    # 1) draw ALL potential first (hatched)
    pot_stack = np.zeros(len(month_order))
    grouped = df.groupby(['Month','Client']).sum().reset_index()
    hatch_styles = ['///','\\\\\\','xxx','...','+++','ooo']
    for i, c in enumerate(clients):
        cd = (grouped[grouped["Client"]==c]
              .set_index("Month")
              .reindex(month_order)
              .fillna(0))
        vals = cd["Potential Revenue"].values
        hatch = hatch_styles[i%len(hatch_styles)]
        ax.bar(x-width/2, vals, width,
               bottom=pot_stack,
               fill=False,
               edgecolor=colors[c],
               hatch=hatch,
               linewidth=1)
        pot_stack += vals

    # 2) draw ALL paid on top (solid)
    paid_stack = np.zeros(len(month_order))
    for c in clients:
        cd = (grouped[grouped["Client"]==c]
              .set_index("Month")
              .reindex(month_order)
              .fillna(0))
        vals = cd["Paid Revenue"].values
        ax.bar(x-width/2, vals, width,
               bottom=paid_stack,
               color=colors[c])
        paid_stack += vals

    # 3) costs
    emp = monthly["Monthly Employee Cost"].values
    ovh = monthly["Overhead Costs"].values
    ax.bar(x+width/2, emp, width, color="#d62728")
    ax.bar(x+width/2, ovh, width, bottom=emp, color="#9467bd")

    # 4) highlight negative‐profit months
    for i in range(len(month_order)):
        if paid_stack[i] < monthly["Total Expenses"].iloc[i]:
            ax.bar(x[i]-width/2,
                   pot_stack[i],
                   width,
                   fill=False,
                   edgecolor='red',
                   linewidth=2)

    # 5) formatting
    ax.set_xticks(x)
    ax.set_xticklabels([m[:3]+" "+m.split()[1] for m in month_order], rotation=45)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y,_: f"${y:,.0f}"))
    ax.set_title("Revenue (by Client) & Expenses", fontsize=18)
    ax.set_xlabel("Month"); ax.set_ylabel("Amount ($)")
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    fig.subplots_adjust(right=0.75)

    # 6) legends
    comp = [
        Patch(facecolor='none', edgecolor='gray', hatch='///', label="Client Rev (Potential)"),
        Patch(facecolor='gray', label="Client Rev (Paid)"),
        Patch(facecolor='#d62728', label="Employee Costs"),
        Patch(facecolor='#9467bd', label="Overhead Costs"),
    ]
    l1 = ax.legend(handles=comp,
                   loc="upper left", bbox_to_anchor=(1.01,1),
                   title="Components")
    ax.add_artist(l1)

    client_h = [Patch(facecolor=colors[c], label=c) for c in clients]
    l2 = ax.legend(handles=client_h,
                   loc="upper left", bbox_to_anchor=(1.01,0.6),
                   title="Clients")
    ax.add_artist(l2)

    st.pyplot(fig)

with tab2:
    fig2, ax2 = plt.subplots(figsize=(16,8))

    # paid & pot & profit lines
    ax2.plot(x, monthly["Paid Revenue"],      'r-',  lw=3, marker='o', label='Paid Revenue')
    ax2.plot(x, monthly["Potential Revenue"], 'b-',  lw=3, marker='s', label='Potential Revenue')
    ax2.plot(x, monthly["Profit"],            'g--', lw=2.5, marker='^', label='Profit (Paid)')
    ax2.plot(x, monthly["Profit (Pot)"],      'c--', lw=2.5, marker='v', label='Profit (Potential)')

    ax2.set_xticks(x)
    ax2.set_xticklabels([m[:3]+" "+m.split()[1] for m in month_order], rotation=45)
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda y,_: f"${y:,.0f}"))
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.set_title("Paid, Potential & Profit Over Time", fontsize=18)
    ax2.set_xlabel("Month"); ax2.set_ylabel("Amount ($)")
    ax2.legend(loc="upper left")

    st.pyplot(fig2)
