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

        # Clients
        raw = p.get("Client", {})\
               .get("formula", {})\
               .get("string", "")
        clients = [c.strip() for c in raw.split(",") if c.strip()]
        if not clients:
            continue

        # Expense Category (select)
        cat = p.get("Expense Category", {})\
               .get("select", {})\
               .get("name", "") or ""
        is_potential = (cat.lower() == "potential")

        # Calculated Revenue
        calc_rev = p.get("Calculated Revenue", {})\
                    .get("formula", {})\
                    .get("number", 0) or 0

        # Costs
        emp_tot = p.get("Monthly Employee Cost", {})\
                   .get("formula", {})\
                   .get("number", 0) or 0
        ovh_tot = p.get("Overhead Costs", {})\
                   .get("number", 0) or 0

        # Month (now safe)
        month = p.get("Month", {})\
                 .get("select", {})\
                 .get("name")
        if not month:
            continue

        # Evenly split across clients
        n = len(clients)
        rev_pc = calc_rev / n
        emp_pc = emp_tot  / n
        ovh_pc = ovh_tot  / n

        for c in clients:
            rows.append({
                "Month": month,
                "Client": c,
                "Paid Revenue":      0.0      if is_potential else rev_pc,
                "Potential Revenue": rev_pc   if is_potential else 0.0,
                "Monthly Employee Cost": emp_pc,
                "Overhead Costs":        ovh_pc
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

monthly = df.groupby("Month")[
    ["Paid Revenue","Potential Revenue","Monthly Employee Cost","Overhead Costs"]
].sum().reindex(month_order)
monthly["Total Expenses"]      = monthly["Monthly Employee Cost"] + monthly["Overhead Costs"]
monthly["Profit (Paid)"]       = monthly["Paid Revenue"] - monthly["Total Expenses"]
monthly["Profit (Potential)"]  = monthly["Potential Revenue"] - monthly["Total Expenses"]

clients = sorted(df['Client'].unique())
colors  = dict(zip(clients, plt.cm.tab20(np.linspace(0,1,len(clients)))))
x = np.arange(len(month_order))
width = 0.35

# --- STREAMLIT TABS ---
tab1, tab2 = st.tabs(["📊 Bar Chart","📈 Line Chart"])

with tab1:
    fig, ax = plt.subplots(figsize=(22,12))

    # 1) Draw ALL Potential first (hatched)
    grouped = df.groupby(['Month','Client']).sum().reset_index()
    pot_stack = np.zeros(len(month_order))
    for client in clients:
        cd = (grouped[grouped["Client"]==client]
              .set_index("Month")
              .reindex(month_order, fill_value=0))
        vals = cd["Potential Revenue"].values
        ax.bar(
            x - width/2, vals, width,
            bottom=pot_stack,
            fill=False,
            edgecolor=colors[client],
            hatch='///',
            linewidth=1
        )
        pot_stack += vals

    # 2) Draw ALL Paid on top (solid)
    paid_stack = np.zeros(len(month_order))
    for client in clients:
        cd = (grouped[grouped["Client"]==client]
              .set_index("Month")
              .reindex(month_order, fill_value=0))
        vals = cd["Paid Revenue"].values
        ax.bar(
            x - width/2, vals, width,
            bottom=paid_stack,
            color=colors[client]
        )
        paid_stack += vals

    # 3) Employee + Overhead costs
    emp = monthly["Monthly Employee Cost"].values
    ovh = monthly["Overhead Costs"].values
    ax.bar(x + width/2, emp, width, color="#d62728")
    ax.bar(x + width/2, ovh, width, bottom=emp, color="#9467bd")

    # 4) Highlight negative-profit months
    for i in range(len(month_order)):
        if paid_stack[i] < monthly["Total Expenses"].iloc[i]:
            ax.bar(
                x[i] - width/2,
                pot_stack[i],
                width,
                fill=False,
                edgecolor='red',
                linewidth=2
            )

    # 5) Formatting
    ax.set_xticks(x)
    ax.set_xticklabels(
        [m.split()[0][:3] + ' ' + m.split()[1] for m in month_order],
        rotation=45
    )
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y,_: f"${y:,.0f}"))
    ax.set_title("Revenue (by Client) & Expenses", fontsize=18)
    ax.set_xlabel("Month"); ax.set_ylabel("Amount ($)")
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # 6) Legends at axis level
    fig.subplots_adjust(right=0.75)
    comp_handles = [
        Patch(facecolor='none', edgecolor='gray', hatch='///', label="Client Rev (Potential)"),
        Patch(facecolor='gray', label="Client Rev (Paid)"),
        Patch(facecolor="#d62728", label="Employee Costs"),
        Patch(facecolor="#9467bd", label="Overhead Costs"),
    ]
    comp_leg = ax.legend(
        handles=comp_handles,
        loc="upper left",
        bbox_to_anchor=(1.01,1),
        title="Components"
    )
    ax.add_artist(comp_leg)

    client_handles = [Patch(facecolor=colors[c], label=c) for c in clients]
    client_leg = ax.legend(
        handles=client_handles,
        loc="upper left",
        bbox_to_anchor=(1.01,0.6),
        title="Clients"
    )
    ax.add_artist(client_leg)

    st.pyplot(fig)

with tab2:
    fig2, ax2 = plt.subplots(figsize=(16,8))

    # Paid & Potential Revenue lines
    ax2.plot(x, monthly["Paid Revenue"],      'r-',  lw=3, marker='o', label='Paid Revenue')
    ax2.plot(x, monthly["Potential Revenue"], 'b-',  lw=3, marker='s', label='Potential Revenue')

    # Profit lines
    ax2.plot(x, monthly["Profit (Paid)"],      'g--', lw=2.5, marker='^', label='Profit (Paid)')
    ax2.plot(x, monthly["Profit (Potential)"], 'c--', lw=2.5, marker='v', label='Profit (Potential)')

    ax2.set_xticks(x)
    ax2.set_xticklabels(
        [m.split()[0][:3] + ' ' + m.split()[1] for m in month_order],
        rotation=45
    )
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda y,_: f"${y:,.0f}"))
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.set_title("Paid, Potential & Profit Over Time", fontsize=18)
    ax2.set_xlabel("Month"); ax2.set_ylabel("Amount ($)")
    ax2.legend(loc="upper left", bbox_to_anchor=(1.01,1))

    st.pyplot(fig2)
