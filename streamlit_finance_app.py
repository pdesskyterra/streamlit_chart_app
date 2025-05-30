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
st.title("📊 Profit & Expense Tracker (Expense-Category Basis)")

# --- FETCH & EXPAND NOTION DATA ---
@st.cache_data(ttl=600)
def fetch_notion_data():
    notion = Client(auth=NOTION_TOKEN)
    rows   = []
    resp   = notion.databases.query(database_id=DATABASE_ID)
    for page in resp["results"]:
        p     = page["properties"]
        month = p["Month"]["select"]["name"] if p["Month"]["select"] else None
        if not month:
            continue

        # 1) Clients
        raw_clients = p["Client"]["formula"]["string"]
        clients     = [c.strip() for c in raw_clients.split(",") if c.strip()]
        n = len(clients)
        if n == 0:
            continue

        # 2) Expense Categories
        tags = [
            e["select"]["name"]
            for e in p["Expense Category"]["rollup"]["array"]
            if e["type"]=="select"
        ]
        if len(tags) != n:
            tags = ["Paid"] * n

        # 3) Paid Revenue per client
        paid_vals = []
        for e in p["Paid Revenue"]["rollup"]["array"]:
            if e["type"]=="number":
                paid_vals.append(e["number"] or 0)
        if len(paid_vals) != n:
            total_paid = p["Paid Revenue"]["rollup"]["number"] or 0
            paid_vals = [total_paid/n] * n

        # 4) Potential Revenue per client
        pot_vals = []
        for e in p["Potential Revenue (rollup)"]["rollup"]["array"]:
            if e["type"]=="formula":
                s = e["formula"]["string"].replace("$","").replace(",","")
                if s and s.replace(".", "",1).isdigit():
                    pot_vals += [float(v) for v in s.split(",") if v.strip()]
        if len(pot_vals) != n:
            total_pot = sum(pot_vals)
            avg = total_pot/n if n else 0
            pot_vals = [avg]*n

        # 5) Costs per client
        emp_tot = p["Monthly Employee Cost"]["formula"]["number"] or 0
        ovh_tot = p["Overhead Costs"]["number"] or 0
        emp_share = emp_tot / n
        ovh_share = ovh_tot / n

        # 6) Build rows
        for i, client in enumerate(clients):
            rows.append({
                "Month":          month,
                "Client":         client,
                "Tag":            tags[i],
                "Paid":           paid_vals[i],
                "Potential":      pot_vals[i],
                "Employee Cost":  emp_share,
                "Overhead Cost":  ovh_share
            })

    return pd.DataFrame(rows)

df = fetch_notion_data()
if df.empty:
    st.warning("No data found or invalid credentials.")
    st.stop()

# --- FILTER MONTHS & AGGREGATE ---
months = [
    'February 2025','March 2025','April 2025',
    'May 2025','June 2025','July 2025','August 2025'
]
df['Month'] = pd.Categorical(df['Month'], categories=months, ordered=True)
df = df[df['Month'].notna()]

# one row per Month×Client
df_mc = (
    df.groupby(['Month','Client'], sort=False)
      .sum()[["Paid","Potential","Employee Cost","Overhead Cost"]]
      .reset_index()
)

# Tag lookup
tag_map = (
    df[['Month','Client','Tag']]
      .drop_duplicates()
      .set_index(['Month','Client'])['Tag']
)

# month‐level totals for line chart
monthly = df_mc.groupby('Month').sum().reindex(months, fill_value=0)
revenue   = monthly["Paid"] + monthly["Potential"]
profit    = revenue - (monthly["Employee Cost"]+monthly["Overhead Cost"])
margin    = np.where(revenue>0, profit/revenue*100, np.nan)

# plot settings
clients   = df_mc['Client'].unique().tolist()
colors    = dict(zip(clients, plt.cm.tab20(np.linspace(0,1,len(clients)))))
hatch_map = {
    "Paid":      "",
    "Invoiced":  "//",
    "Committed": "xx",
    "Proposal":  ".."
}
x      = np.arange(len(months))
width  = 0.35

# --- DRAW CHARTS ---
tab1, tab2 = st.tabs(["📊 Stacked Bar","📈 Line Chart"])

with tab1:
    fig, ax = plt.subplots(figsize=(16,8))
    base = np.zeros(len(months))

    # 1) Revenue bars by client & hatch by Tag
    for client in clients:
        sub = (df_mc[df_mc['Client']==client]
               .set_index('Month')
               .reindex(months, fill_value=0))
        paid = sub["Paid"].values
        pot  = sub["Potential"].values

        # solid Paid
        ax.bar(x-width/2, paid, width, bottom=base,
               color=colors[client], label=client)
        base += paid

        # hatch Pattern = tag, using Potential slice
        for i, m in enumerate(months):
            tag = tag_map.get((m,client), "Paid")
            h   = hatch_map.get(tag, "")
            if pot[i]>0:
                ax.bar(x[i]-width/2, pot[i], width, bottom=base[i],
                       color=colors[client],
                       hatch=h, edgecolor='black')
        base += pot

    # 2) Costs
    cost_base = np.zeros(len(months))
    ax.bar(x+width/2, monthly["Employee Cost"], width,
           bottom=cost_base, color="#d62728", label="Employee Cost")
    cost_base += monthly["Employee Cost"]
    ax.bar(x+width/2, monthly["Overhead Cost"], width,
           bottom=cost_base, color="#9467bd", label="Overhead Cost")

    # 3) highlight negative‐profit
    for i in range(len(months)):
        if profit.iloc[i] < 0:
            ax.bar(x[i], revenue.iloc[i], width*2,
                   fill=False, edgecolor='red', linewidth=2)

    # 4) formatting
    ax.set_xticks(x)
    ax.set_xticklabels([m[:3]+" "+m.split()[1] for m in months],
                       rotation=45)
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda y,_: f"${y:,.0f}")
    )
    ax.set_title("Revenue (by Client & Expense Category) and Costs", fontsize=16)
    ax.set_xlabel("Month"); ax.set_ylabel("Amount ($)")

    # 5) legends
    client_patches = [Patch(facecolor=colors[c], label=c) for c in clients]
    tag_patches    = [Patch(facecolor='white',
                            edgecolor='black', hatch=h, label=tag)
                      for tag,h in hatch_map.items()]
    cost_patches   = [
        Patch(facecolor="#d62728", label="Employee Cost"),
        Patch(facecolor="#9467bd", label="Overhead Cost")
    ]

    leg1 = ax.legend(handles=client_patches, title="Clients",
                     loc="upper right", bbox_to_anchor=(0.95,0.6))
    ax.add_artist(leg1)
    ax.legend(handles=tag_patches+cost_patches,
              title="Expense Categories",
              loc="upper right", bbox_to_anchor=(0.95,0.3))

    fig.tight_layout(rect=[0,0,0.8,1])
    st.pyplot(fig)

with tab2:
    fig2, ax2 = plt.subplots(figsize=(16,8))
    # Paid, Potential, Profit
    l1, = ax2.plot(x, monthly["Paid"],      'o-', label='Paid Revenue')
    l2, = ax2.plot(x, monthly["Potential"], 's-', label='Potential Revenue')
    l3, = ax2.plot(x, profit,               '^-',label='Potential Profit')

    ax3 = ax2.twinx()
    l4, = ax3.plot(x, margin, 'd--', label='Profit Margin (%)')
    ax3.set_ylabel("Profit Margin (%)")
    ax3.yaxis.set_major_formatter(FuncFormatter(lambda p,_: f"{p:.0f}%"))

    # annotate non-overlapping
    for i in range(len(x)):
        ax2.annotate(f"${monthly['Paid'].iloc[i]:,.0f}",
                     (x[i], monthly['Paid'].iloc[i]),
                     xytext=(0,10), textcoords="offset points", ha='center')
        ax2.annotate(f"${monthly['Potential'].iloc[i]:,.0f}",
                     (x[i], monthly['Potential'].iloc[i]),
                     xytext=(0,-10), textcoords="offset points", ha='center')
        ax2.annotate(f"${profit.iloc[i]:,.0f}",
                     (x[i], profit.iloc[i]),
                     xytext=(0,20), textcoords="offset points", ha='center')
        if not np.isnan(margin[i]):
            ax3.annotate(f"{margin[i]:.0f}%",
                         (x[i], margin[i]),
                         xytext=(0,30), textcoords="offset points", ha='center')

    ax2.set_xticks(x)
    ax2.set_xticklabels([m[:3]+" "+m.split()[1] for m in months],
                        rotation=45)
    ax2.set_title("Paid, Potential & Profit Over Time", fontsize=16)
    ax2.set_xlabel("Month"); ax2.set_ylabel("Amount ($)")
    fig2.legend(handles=[l1,l2,l3,l4], loc='upper right')
    fig2.tight_layout(rect=[0,0,0.8,1])
    st.pyplot(fig2)
