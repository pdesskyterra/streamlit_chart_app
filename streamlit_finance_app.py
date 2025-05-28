import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.patches import Patch
from notion_client import Client

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
    rows   = []
    resp   = notion.databases.query(database_id=DATABASE_ID)

    for page in resp["results"]:
        p = page["properties"]

        # 1) Month
        month = p.get("Month", {})\
                 .get("select", {})\
                 .get("name")
        if not month:
            continue

        # 2) Clients list
        client_raw = p.get("Client", {})\
                      .get("formula", {})\
                      .get("string", "")
        clients = [c.strip() for c in client_raw.split(",") if c.strip()]
        if not clients:
            continue
        n = len(clients)

        # 3) Expense Category rollup → list of selects
        exp_rollup = p.get("Expense Category", {})\
                      .get("rollup", {})\
                      .get("array", [])
        cats = []
        for e in exp_rollup:
            if e.get("type") == "select":
                name = e.get("select", {})\
                        .get("name", "")
                cats.append(name)
        # fallback if lengths mismatch
        if len(cats) != n:
            cats = ["Paid"] * n

        # 4) Calculated Revenue (your formula field)
        calc_rev = p.get("Calculated Revenue", {})\
                    .get("formula", {})\
                    .get("number", 0) or 0

        # 5) Paid Revenue rollup
        paid_total = p.get("Paid Revenue", {})\
                      .get("rollup", {})\
                      .get("number", 0) or 0

        # 6) Costs
        emp_tot = p.get("Monthly Employee Cost", {})\
                   .get("formula", {})\
                   .get("number", 0) or 0
        ovh_tot = p.get("Overhead Costs", {})\
                   .get("number", 0) or 0

        # split shares
        paid_share = paid_total / n
        pot_share  = max(0, calc_rev - paid_total) / n
        emp_share  = emp_tot  / n
        ovh_share  = ovh_tot  / n

        # build one row per client
        for idx, client in enumerate(clients):
            cat = cats[idx].lower()
            rows.append({
                "Month": month,
                "Client": client,
                "Paid Revenue":      paid_share  if cat == "paid"      else 0.0,
                "Potential Revenue": pot_share   if cat == "potential" else 0.0,
                "Monthly Employee Cost": emp_share,
                "Overhead Costs":        ovh_share
            })

    return pd.DataFrame(rows)


df = fetch_notion_data()
if df.empty:
    st.warning("No data found or invalid Notion credentials.")
    st.stop()

# --- COMMON PROCESSING ---
month_order = [
    'November 2024','December 2024','February 2025','March 2025',
    'April 2025','May 2025','June 2025','July 2025','August 2025'
]
df['Month'] = pd.Categorical(df['Month'], categories=month_order, ordered=True)
df = df.sort_values(['Month','Client'])

# monthly aggregates
monthly = df.groupby("Month")[
    ["Paid Revenue","Potential Revenue","Monthly Employee Cost","Overhead Costs"]
].sum().reindex(month_order, fill_value=0)
monthly["Total Expenses"]      = (monthly["Monthly Employee Cost"] +
                                  monthly["Overhead Costs"])
monthly["Profit (Paid)"]       = (monthly["Paid Revenue"] -
                                  monthly["Total Expenses"])
monthly["Profit (Potential)"]  = (monthly["Potential Revenue"] -
                                  monthly["Total Expenses"])

clients = sorted(df['Client'].unique())
colors  = dict(zip(clients, plt.cm.tab20(np.linspace(0,1,len(clients)))))
x = np.arange(len(month_order))
width = 0.35

# --- STREAMLIT TABS ---
tab1, tab2 = st.tabs(["📊 Bar Chart","📈 Line Chart"])

with tab1:
    fig, ax = plt.subplots(figsize=(22,12))

    # draw potential (hatched) then paid (solid)
    grouped = df.groupby(['Month','Client']).sum().reset_index()
    stack = np.zeros(len(month_order))
    for client in clients:
        cd   = (grouped[grouped["Client"]==client]
                .set_index("Month")
                .reindex(month_order, fill_value=0))
        pot  = cd["Potential Revenue"].values
        paid = cd["Paid Revenue"].values

        ax.bar(x-width/2, pot, width,
               bottom=stack, fill=False,
               edgecolor=colors[client], hatch='///', linewidth=1)
        ax.bar(x-width/2, paid, width,
               bottom=stack, color=colors[client])
        stack += pot + paid

    # costs
    emp = monthly["Monthly Employee Cost"].values
    ovh = monthly["Overhead Costs"].values
    ax.bar(x+width/2, emp, width, color="#d62728")
    ax.bar(x+width/2, ovh, width, bottom=emp, color="#9467bd")

    # outline months where paid < expenses
    for i in range(len(month_order)):
        if monthly["Paid Revenue"].iloc[i] < monthly["Total Expenses"].iloc[i]:
            total = (monthly["Paid Revenue"].iloc[i] +
                     monthly["Potential Revenue"].iloc[i])
            ax.bar(x[i]-width/2, total,
                   width, fill=False,
                   edgecolor='red', linewidth=2)

    # formatting
    ax.set_xticks(x)
    ax.set_xticklabels([m[:3]+" "+m.split()[1] for m in month_order],
                       rotation=45)
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda y, _: f"${y:,.0f}")
    )
    ax.set_title("Revenue (by Client) & Expenses", fontsize=18)
    ax.set_xlabel("Month"); ax.set_ylabel("Amount ($)")
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # legends
    fig.subplots_adjust(right=0.75)
    comp_handles = [
        Patch(facecolor="#1f77b4", label="Client Rev (Paid)"),
        Patch(facecolor="#1f77b4", alpha=0.5,
              hatch="///", label="Client Rev (Potential)"),
        Patch(facecolor="#d62728", label="Employee Costs"),
        Patch(facecolor="#9467bd", label="Overhead Costs"),
    ]
    comp_leg = ax.legend(handles=comp_handles,
                         loc="upper left",
                         bbox_to_anchor=(1.01,1),
                         title="Components")
    ax.add_artist(comp_leg)

    client_handles = [Patch(facecolor=colors[c], label=c)
                      for c in clients]
    client_leg = ax.legend(handles=client_handles,
                           loc="upper left",
                           bbox_to_anchor=(1.01,0.6),
                           title="Clients")
    ax.add_artist(client_leg)

    st.pyplot(fig)


with tab2:
    fig2, ax2 = plt.subplots(figsize=(16,8))

    # revenue lines
    ax2.plot(x, monthly["Paid Revenue"],      'r-',  lw=3,
             marker='o', label='Paid Revenue')
    ax2.plot(x, monthly["Potential Revenue"], 'b-',  lw=3,
             marker='s', label='Potential Revenue')

    # profit lines
    ax2.plot(x, monthly["Profit (Paid)"],     'g--', lw=2.5,
             marker='^', label='Profit (Paid)')
    ax2.plot(x, monthly["Profit (Potential)"],'c--', lw=2.5,
             marker='v', label='Profit (Potential)')

    # formatting
    ax2.set_xticks(x)
    ax2.set_xticklabels([m[:3]+" "+m.split()[1]
                         for m in month_order],
                        rotation=45)
    ax2.yaxis.set_major_formatter(
        FuncFormatter(lambda y, _: f"${y:,.0f}")
    )
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.set_title("Paid, Potential & Profit Over Time", fontsize=18)
    ax2.set_xlabel("Month"); ax2.set_ylabel("Amount ($)")
    ax2.legend(loc="upper left", bbox_to_anchor=(1.01,1))

    st.pyplot(fig2)
