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

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
from matplotlib.patches import Patch
from notion_client import Client
import ast

# … (your fetch_notion_data and processing code above remains unchanged) …

tab1, tab2 = st.tabs(["📊 Bar Chart","📈 Line Chart"])

with tab1:
    fig, ax = plt.subplots(figsize=(22, 12))

    # A small set of distinct hatch patterns
    hatch_patterns = ['///', '\\\\\\', 'xxx', '...', '+++', 'ooo']

    # Precompute grouped sums
    grouped = df.groupby(['Month','Client']).sum().reset_index()
    pot_stack  = np.zeros(len(month_order))
    paid_stack = np.zeros(len(month_order))

    # 1) Draw all "Potential" first (hatched, edgecolor = client color)
    for i, client in enumerate(clients):
        cd = (grouped[grouped["Client"] == client]
              .set_index("Month")
              .reindex(month_order)
              .fillna(0))
        pot_vals = cd["Potential Revenue"].values
        hatch    = hatch_patterns[i % len(hatch_patterns)]

        ax.bar(
            x - width/2, pot_vals, width,
            bottom=pot_stack,
            fill=False,
            edgecolor=colors[client],
            hatch=hatch,
            linewidth=1
        )
        pot_stack += pot_vals

    # 2) Draw all "Paid" on top (solid)
    for i, client in enumerate(clients):
        cd = (grouped[grouped["Client"] == client]
              .set_index("Month")
              .reindex(month_order)
              .fillna(0))
        paid_vals = cd["Paid Revenue"].values

        ax.bar(
            x - width/2, paid_vals, width,
            bottom=paid_stack,
            color=colors[client]
        )
        paid_stack += paid_vals

    # 3) Employee + Overhead costs
    emp_costs = monthly["Monthly Employee Cost"].values
    ovh_costs = monthly["Overhead Costs"].values
    ax.bar(x + width/2, emp_costs, width, color="#d62728")
    ax.bar(x + width/2, ovh_costs, width, bottom=emp_costs, color="#9467bd")

    # 4) Highlight months where Paid < Expenses
    for i in range(len(month_order)):
        if paid_stack[i] < monthly["Total Expenses"].iloc[i]:
            ax.bar(
                x[i] - width/2,
                pot_stack[i],
                width, fill=False,
                edgecolor='red',
                linewidth=2
            )

    # 5) Formatting
    ax.set_xticks(x)
    ax.set_xticklabels(
        [m.split()[0][:3] + ' ' + m.split()[1] for m in month_order],
        rotation=45
    )
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"${y:,.0f}"))
    ax.set_title("Revenue (by Client) & Expenses", fontsize=18)
    ax.set_xlabel("Month")
    ax.set_ylabel("Amount ($)")
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # 6) Legends – use axis‐level legends and reserve right‐margin
    fig.subplots_adjust(right=0.75)

    # Components legend
    comp_handles = [
        Patch(facecolor='none', edgecolor='gray', hatch=hatch_patterns[0], label="Client Revenue (Potential)"),
        Patch(facecolor='gray', label="Client Revenue (Paid)"),
        Patch(facecolor="#d62728", label="Employee Costs"),
        Patch(facecolor="#9467bd", label="Overhead Costs"),
    ]
    main_legend = ax.legend(
        handles=comp_handles,
        loc="upper left",
        bbox_to_anchor=(1.01, 1),
        title="Components"
    )
    ax.add_artist(main_legend)

    # Clients legend
    client_handles = [Patch(facecolor=colors[c], label=c) for c in clients]
    client_legend = ax.legend(
        handles=client_handles,
        loc="upper left",
        bbox_to_anchor=(1.01, 0.6),
        title="Clients"
    )
    ax.add_artist(client_legend)

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
