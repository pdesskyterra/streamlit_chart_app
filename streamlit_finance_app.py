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
        # split potential
        pot_raw = []
        for e in props["Potential Revenue (rollup)"]["rollup"]["array"]:
            if e["type"] == "formula":
                s = e["formula"]["string"].replace("$","").replace(",","")
                pot_raw += [p.strip() for p in s.split(",") if p.strip()]
        pot_vals = [float(v) if v.replace('.','',1).isdigit() else 0.0 for v in pot_raw]
        # metrics
        n = len(clients)
        if n == 0: continue
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
                "Paid Revenue": paid/n,
                "Potential Revenue": pot,
                "Monthly Employee Cost": emp/n,
                "Overhead Costs": ovh/n
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
monthly["Total Expenses"]       = monthly["Monthly Employee Cost"] + monthly["Overhead Costs"]
monthly["Profit (Paid)"]        = monthly["Paid Revenue"] - monthly["Total Expenses"]
monthly["Profit (Potential)"]   = monthly["Potential Revenue"] - monthly["Total Expenses"]
monthly["Paid Margin (%)"]      = np.where(
    monthly["Paid Revenue"]>0,
    monthly["Profit (Paid)"]/monthly["Paid Revenue"]*100,
    np.nan
)
monthly["Potential Margin (%)"] = np.where(
    monthly["Potential Revenue"]>0,
    monthly["Profit (Potential)"]/monthly["Potential Revenue"]*100,
    np.nan
)

clients = sorted(df['Client'].unique())
colors  = dict(zip(clients, plt.cm.tab20(np.linspace(0,1,len(clients)))))
x = np.arange(len(month_order))
width = 0.35

tab1, tab2 = st.tabs(["📊 Bar Chart","📈 Line Chart"])

with tab1:
    # -- BAR CHART --
    fig, ax = plt.subplots(figsize=(22,12))

    # 1) stacked revenue
    stack = np.zeros(len(month_order))
    grouped = df.groupby(['Month','Client']).sum().reset_index()
    for client in clients:
        cd = (grouped[grouped["Client"]==client]
              .set_index("Month").reindex(month_order).fillna(0))
        paid_vals = cd["Paid Revenue"].values
        pot_vals  = cd["Potential Revenue"].values
        delta     = np.maximum(0, pot_vals - paid_vals)

        ax.bar(x-width/2, paid_vals, width, bottom=stack, color=colors[client])
        stack += paid_vals
        ax.bar(x-width/2, delta, width, bottom=stack,
               color=colors[client], alpha=0.5, hatch='///')
        stack += delta

    # 2) costs
    emp_costs = monthly["Monthly Employee Cost"].values
    ovh_costs = monthly["Overhead Costs"].values
    ax.bar(x+width/2, emp_costs, width, color="#d62728")
    ax.bar(x+width/2, ovh_costs, width, bottom=emp_costs, color="#9467bd")

    # 3) highlight negative
    for i, prof in enumerate(monthly["Profit (Paid)"]):
        if prof < 0:
            ax.bar(x[i]-width/2,
                   monthly["Potential Revenue"].iloc[i],
                   width=width, fill=False,
                   edgecolor='red', linewidth=2)

    # 4) format
    ax.set_xticks(x)
    ax.set_xticklabels([m.split()[0][:3]+' '+m.split()[1] for m in month_order],
                       rotation=45)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y,_: f"${y:,.0f}"))
    ax.set_title("Revenue (by Client) & Expenses (Employee + Overhead)", fontsize=18)
    ax.set_xlabel("Month"); ax.set_ylabel("Amount ($)")
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # 5) legends at figure-level
    fig.tight_layout(rect=[0,0,0.5,1])

    # components legend
    comp_handles = [
        Patch(facecolor="#1f77b4", label="Client Rev (Paid)"),
        Patch(facecolor="#1f77b4", alpha=0.5, hatch="///", label="Client Rev (Potential)"),
        Patch(facecolor="#d62728", label="Employee Costs"),
        Patch(facecolor="#9467bd", label="Overhead Costs"),
    ]
    comp_labels = [h.get_label() for h in comp_handles]
    fig.legend(comp_handles, comp_labels,
               loc="upper right", bbox_to_anchor=(0.98,0.98), title="Components")

    # clients legend
    client_handles = [Patch(facecolor=colors[c], label=c) for c in clients]
    client_labels = [h.get_label() for h in client_handles]
    fig.legend(client_handles, client_labels,
               loc="upper right", bbox_to_anchor=(0.98,0.75), title="Clients")

    st.pyplot(fig)

with tab2:
    # -- LINE CHART --
    fig2, ax2 = plt.subplots(figsize=(16,8))

    # primary lines
    l1, = ax2.plot(x, monthly["Paid Revenue"],      'r-',  lw=3, marker='o', label='Paid Revenue')
    l2, = ax2.plot(x, monthly["Potential Revenue"], 'b-',  lw=3, marker='s', label='Potential Revenue')
    l3, = ax2.plot(x, monthly["Profit (Paid)"],      'g--', lw=2.5, marker='^', label='Profit (Paid)')
    l4, = ax2.plot(x, monthly["Profit (Potential)"], 'c--', lw=2.5, marker='v', label='Profit (Potential)')

    # secondary margins
    ax3 = ax2.twinx()
    l5, = ax3.plot(x, monthly["Paid Margin (%)"],      'm-.', lw=2, marker='d', label='Paid Margin (%)')
    l6, = ax3.plot(x, monthly["Potential Margin (%)"], 'y-.', lw=2, marker='x', label='Potential Margin (%)')
    ax3.set_ylabel("Profit Margin (%)")
    ax3.yaxis.set_major_formatter(FuncFormatter(lambda p,_: f"{p:.0f}%"))

    # annotate
    for i in range(len(x)):
        ax2.annotate(f"${monthly['Profit (Paid)'].iloc[i]:,.0f}",
                     (x[i], monthly['Profit (Paid)'].iloc[i]),
                     textcoords="offset points", xytext=(0,10),
                     ha='center', color='green', fontsize=9)
        ax2.annotate(f"${monthly['Profit (Potential)'].iloc[i]:,.0f}",
                     (x[i], monthly['Profit (Potential)'].iloc[i]),
                     textcoords="offset points", xytext=(0,20),
                     ha='center', color='teal', fontsize=9)
        pm = monthly["Paid Margin (%)"].iloc[i]
        qm = monthly["Potential Margin (%)"].iloc[i]
        if not np.isnan(pm):
            ax3.annotate(f"{pm:.0f}%",
                         (x[i], pm),
                         textcoords="offset points", xytext=(10,0),
                         ha='left', color='m', fontsize=8)
        if not np.isnan(qm):
            ax3.annotate(f"{qm:.0f}%",
                         (x[i], qm),
                         textcoords="offset points", xytext=(10,-10),
                         ha='left', color='y', fontsize=8)

    # format axes
    ax2.set_xticks(x)
    ax2.set_xticklabels([m.split()[0][:3]+' '+m.split()[1] for m in month_order],
                        rotation=45)
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda y,_: f"${y:,.0f}"))
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.set_title("Paid, Potential, Profit & Margin Over Time", fontsize=18)
    ax2.set_xlabel("Month"); ax2.set_ylabel("Amount ($)")

    # figure‐level legend
    fig2.tight_layout(rect=[0,0,0.75,1])
    all_handles = [l1,l2,l3,l4,l5,l6]
    all_labels  = [h.get_label() for h in all_handles]
    fig2.legend(all_handles, all_labels,
                loc="upper right", bbox_to_anchor=(0.98,0.98))

    st.pyplot(fig2)
