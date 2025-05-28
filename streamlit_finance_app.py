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
    for result in resp["results"]:
        props = result["properties"]
        # split clients
        raw = props["Client"]["formula"]["string"]
        clients = [c.strip() for c in raw.split(",") if c.strip()]
        # split potential rollup
        pot_raw = []
        for e in props["Potential Revenue (rollup)"]["rollup"]["array"]:
            if e["type"] == "formula":
                s = e["formula"]["string"].replace("$","").replace(",","")
                pot_raw += [p.strip() for p in s.split(",") if p.strip()]
        pot_vals = [float(v) if v.replace('.','',1).isdigit() else 0.0 for v in pot_raw]
        n = len(clients)
        if n == 0:
            continue
        paid = props["Paid Revenue"]["rollup"]["number"]
        emp  = props["Monthly Employee Cost"]["formula"]["number"]
        ovh  = props["Overhead Costs"]["number"]
        month = props["Month"]["select"]["name"]
        # pair or average
        if len(pot_vals) == n:
            pairs = zip(clients, pot_vals)
        else:
            avg = sum(pot_vals)/n if pot_vals else 0
            pairs = [(c, avg) for c in clients]
        for client, pot in pairs:
            rows.append({
                "Month": month,
                "Client": client,
                "Paid Revenue": paid / n,
                "Potential Revenue": pot,
                "Monthly Employee Cost": emp / n,
                "Overhead Costs": ovh / n
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
df = df[df['Month'].notna()].sort_values(['Month','Client'])

monthly = df.groupby("Month")[
    ["Paid Revenue","Potential Revenue","Monthly Employee Cost","Overhead Costs"]
].sum().reindex(month_order)
monthly["Total Expenses"] = monthly["Monthly Employee Cost"] + monthly["Overhead Costs"]

clients = sorted(df['Client'].unique())
colors  = dict(zip(clients, plt.cm.tab20(np.linspace(0,1,len(clients)))))
x = np.arange(len(month_order))
width = 0.35

tab1, tab2 = st.tabs(["📊 Bar Chart","📈 Line Chart"])

with tab1:
    fig, ax = plt.subplots(figsize=(22,12))

    # --- NEW: Draw ALL potential values first (hatched) ---
    pot_stack = np.zeros(len(month_order))
    grouped = df.groupby(['Month','Client']).sum().reset_index()
    for client in clients:
        cd = (grouped[grouped["Client"]==client]
              .set_index("Month")
              .reindex(month_order)
              .fillna(0))
        pot_vals = cd["Potential Revenue"].values
        ax.bar(x - width/2,
               pot_vals,
               width,
               bottom=pot_stack,
               fill=False,
               edgecolor=colors[client],
               hatch='///',
               linewidth=0)  # you can tweak linewidth
        pot_stack += pot_vals

    # --- Then draw ALL paid values on top (solid) ---
    paid_stack = np.zeros(len(month_order))
    for client in clients:
        cd = (grouped[grouped["Client"]==client]
              .set_index("Month")
              .reindex(month_order)
              .fillna(0))
        paid_vals = cd["Paid Revenue"].values
        ax.bar(x - width/2,
               paid_vals,
               width,
               bottom=paid_stack,
               color=colors[client])
        paid_stack += paid_vals

    # --- Expenses side bar ---
    emp_costs = monthly["Monthly Employee Cost"].values
    ovh_costs = monthly["Overhead Costs"].values
    ax.bar(x + width/2, emp_costs, width, color="#d62728")
    ax.bar(x + width/2, ovh_costs, width, bottom=emp_costs, color="#9467bd")

    # Highlight months where Paid < Expenses
    for i in range(len(month_order)):
        if paid_stack[i] < monthly["Total Expenses"].iloc[i]:
            ax.bar(x[i] - width/2,
                   pot_stack[i],
                   width,
                   fill=False,
                   edgecolor='red',
                   linewidth=2)

    # Formatting
    ax.set_xticks(x)
    ax.set_xticklabels([m[:3] + ' ' + m.split()[1] for m in month_order], rotation=45)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y,_: f"${y:,.0f}"))
    ax.set_title("Revenue (by Client) & Expenses", fontsize=18)
    ax.set_xlabel("Month")
    ax.set_ylabel("Amount ($)")
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Reserve right margin
    fig.tight_layout(rect=[0,0,0.85,1])

    # Components legend
    comp_handles = [
        Patch(facecolor='none', edgecolor=colors[clients[0]], hatch='///',
              label="Client Rev (Potential)"),
        Patch(facecolor=colors[clients[0]], label="Client Rev (Paid)"),
        Patch(facecolor="#d62728", label="Employee Costs"),
        Patch(facecolor="#9467bd", label="Overhead Costs")
    ]
    fig.legend(comp_handles,
               [h.get_label() for h in comp_handles],
               loc="upper right", bbox_to_anchor=(0.98,0.98),
               title="Components")

    # Clients legend
    client_handles = [Patch(facecolor=colors[c], label=c) for c in clients]
    fig.legend(client_handles,
               [h.get_label() for h in client_handles],
               loc="upper right", bbox_to_anchor=(0.98,0.75),
               title="Clients")

    st.pyplot(fig)

with tab2:
    fig2, ax2 = plt.subplots(figsize=(16,8))

    # Paid & Potential Revenue
    l1, = ax2.plot(x, monthly["Paid Revenue"],      'r-', lw=3, marker='o', label='Paid Revenue')
    l2, = ax2.plot(x, monthly["Potential Revenue"], 'b-', lw=3, marker='s', label='Potential Revenue')

    # Potential Profit
    l3, = ax2.plot(x, monthly["Profit (Potential)"], 'c--', lw=2.5, marker='v', label='Profit (Potential)')

    # Profit Margin (%) on secondary axis
    ax3 = ax2.twinx()
    l4, = ax3.plot(x, monthly["Potential Margin (%)"], 'm-.', lw=2, marker='d', label='Potential Margin (%)')
    ax3.set_ylabel("Profit Margin (%)")
    ax3.yaxis.set_major_formatter(FuncFormatter(lambda p,_: f"{p:.0f}%"))

    # annotate margin
    for i, pm in enumerate(monthly["Potential Margin (%)"]):
        if not np.isnan(pm):
            ax3.annotate(f"{pm:.0f}%",
                         (x[i], pm),
                         textcoords="offset points", xytext=(10,0),
                         ha='left', color='m', fontsize=9)

    # formatting
    ax2.set_xticks(x)
    ax2.set_xticklabels([m.split()[0][:3]+' '+m.split()[1] for m in month_order],
                        rotation=45)
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda y,_: f"${y:,.0f}"))
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.set_title("Paid, Potential & Potential Profit Over Time", fontsize=18)
    ax2.set_xlabel("Month"); ax2.set_ylabel("Amount ($)")

    # legends at figure‐level
    fig2.tight_layout(rect=[0,0,0.75,1])
    fig2.legend(handles=[l1,l2,l3,l4],
                labels=[l.get_label() for l in [l1,l2,l3,l4]],
                loc="upper right", bbox_to_anchor=(0.98,0.98))

    st.pyplot(fig2)
