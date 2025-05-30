import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
from matplotlib.patches import Patch
from notion_client import Client

# --- CONFIG ---
NOTION_TOKEN = st.secrets["NOTION_TOKEN"]
DATABASE_ID  = st.secrets["DATABASE_ID"]

# --- PAGE SETUP ---
st.set_page_config(layout="wide")
st.title("📊 Profit & Expense Tracker")

# --- FETCH & TAG DATA ---
@st.cache_data(ttl=600)
def fetch_notion_data():
    notion = Client(auth=NOTION_TOKEN)
    rows = []
    resp = notion.databases.query(database_id=DATABASE_ID)

    for page in resp["results"]:
        p = page["properties"]
        month = p.get("Month",{}).get("select",{}).get("name")
        if not month:
            continue

        # Clients
        raw = p.get("Client",{}).get("formula",{}).get("string","")
        clients = [c.strip() for c in raw.split(",") if c.strip()]
        n = len(clients)
        if n==0:
            continue

        # Expense Category tags
        cats = []
        for e in p.get("Expense Category",{}).get("rollup",{}).get("array",[]):
            if e.get("type")=="select":
                cats.append(e["select"]["name"])
        # default to “Paid” if missing
        if len(cats)!=n:
            cats = ["Paid"]*n

        # Paid revenue total
        paid_total = p.get("Paid Revenue",{}).get("rollup",{}).get("number",0) or 0
        paid_share = paid_total / n

        # Potential revenue total
        pot_raw = []
        for e in p.get("Potential Revenue (rollup)",{}).get("rollup",{}).get("array",[]):
            if e.get("type")=="formula":
                s = e["formula"]["string"].replace("$","").replace(",","")
                pot_raw += [float(v) for v in s.split(",") if v.strip() and v.replace('.','',1).isdigit()]
        total_pot = sum(pot_raw)
        pot_share = max(0, total_pot - paid_total) / n if n else 0

        # Costs per client
        emp = p.get("Monthly Employee Cost",{}).get("formula",{}).get("number",0) or 0
        ovh = p.get("Overhead Costs",{}).get("number",0) or 0
        emp_share = emp / n
        ovh_share = ovh / n

        # Build one row per client
        for client, cat in zip(clients, cats):
            # allocate revenue to exactly one bucket
            rev_paid = paid_share if cat=="Paid" else 0.0
            rev_inv  = pot_share if cat=="Invoiced"  else 0.0
            rev_comm = pot_share if cat=="Committed" else 0.0
            rev_prop = pot_share if cat=="Proposal"  else 0.0

            rows.append({
                "Month": month, "Client": client,
                "Paid":   rev_paid,
                "Invoiced":  rev_inv,
                "Committed": rev_comm,
                "Proposal":  rev_prop,
                "Employee Cost": emp_share,
                "Overhead Cost": ovh_share
            })

    return pd.DataFrame(rows)

df = fetch_notion_data()
if df.empty:
    st.warning("No data found or invalid credentials.")
    st.stop()

# --- FILTER MONTHS ---
months = ['February 2025','March 2025','April 2025',
          'May 2025','June 2025','July 2025','August 2025']
df['Month'] = pd.Categorical(df['Month'], categories=months, ordered=True)
df = df[df['Month'].notna()]

# --- AGGREGATE ---
by_month = df.groupby('Month').sum().reindex(months, fill_value=0)
profit = (by_month[['Paid','Invoiced','Committed','Proposal']].sum(axis=1)
          - by_month[['Employee Cost','Overhead Cost']].sum(axis=1))
margin = np.where(
    by_month[['Paid','Invoiced','Committed','Proposal']].sum(axis=1)>0,
    profit / by_month[['Paid','Invoiced','Committed','Proposal']].sum(axis=1) * 100,
    np.nan
)

# --- PLOT ---
tab1, tab2 = st.tabs(["Revenue & Expenses","Trends Over Time"])

with tab1:
    fig, ax = plt.subplots(figsize=(12,6))
    x = np.arange(len(months))
    width = 0.35

    # stacked revenue buckets
    rev_cats = ["Paid","Invoiced","Committed","Proposal"]
    stack = np.zeros(len(months))
    colors = plt.cm.tab10(np.arange(len(rev_cats)))
    for col, color in zip(rev_cats, colors):
        vals = by_month[col].values
        ax.bar(x - width/2, vals, width, bottom=stack,
               color=color, label=col)
        stack += vals

    # stacked costs
    cost_cats = ["Employee Cost","Overhead Cost"]
    stack2 = np.zeros(len(months))
    cost_colors = ["#d62728","#9467bd"]
    for cat, c in zip(cost_cats, cost_colors):
        vals = by_month[cat].values
        ax.bar(x + width/2, vals, width, bottom=stack2,
               color=c, label=cat)
        stack2 += vals

    # highlight negatives
    for i in range(len(months)):
        if profit.iloc[i] < 0:
            ax.bar(x[i], stack[i], width*2,
                   fill=False, edgecolor="red", linewidth=2)

    ax.set_xticks(x)
    ax.set_xticklabels([m[:3]+" "+m.split()[1] for m in months], rotation=45)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y,_: f"${y:,.0f}"))
    ax.set_title("Revenue (by Category) & Expenses")
    ax.set_xlabel("Month"); ax.set_ylabel("Amount ($)")
    # two legends
    rev_handles = [Patch(color=colors[i], label=rev_cats[i]) for i in range(len(rev_cats))]
    cost_handles= [Patch(color=cost_colors[i], label=cost_cats[i]) for i in range(len(cost_cats))]
    leg1 = ax.legend(handles=rev_handles, title="Revenue Categories",
                     loc="upper right")
    ax.add_artist(leg1)
    ax.legend(handles=cost_handles, title="Cost Categories",
              loc="upper right", bbox_to_anchor=(0.98,0.6))
    fig.tight_layout(rect=[0,0,0.8,1])
    st.pyplot(fig)

with tab2:
    fig2, ax2 = plt.subplots(figsize=(12,6))
    l1, = ax2.plot(x, by_month['Paid']   , 'o-', label='Paid')
    l2, = ax2.plot(x, by_month['Invoiced'], 's-', label='Invoiced')
    l3, = ax2.plot(x, by_month['Committed'],'^-', label='Committed')
    l4, = ax2.plot(x, by_month['Proposal'], 'v-', label='Proposal')
    ax3 = ax2.twinx()
    l5, = ax3.plot(x, margin, 'd--', label='Margin (%)')
    ax3.set_ylabel("Margin (%)")
    ax3.yaxis.set_major_formatter(FuncFormatter(lambda p,_: f"{p:.0f}%"))

    # annotate
    for i in range(len(x)):
        for ser,off in zip([by_month['Paid'],by_month['Invoiced'],
                             by_month['Committed'],by_month['Proposal']],
                           [10, -10, 20, -20]):
            val = ser.iloc[i]
            ax2.annotate(f"${val:,.0f}", (x[i], val),
                         textcoords="offset points",
                         xytext=(0,off), ha="center")
        m = margin[i]
        if not np.isnan(m):
            ax3.annotate(f"{m:.0f}%", (x[i], m),
                         textcoords="offset points", xytext=(0,30),
                         ha="center")

    ax2.set_xticks(x)
    ax2.set_xticklabels([m[:3]+" "+m.split()[1] for m in months], rotation=45)
    ax2.set_title("Revenue Pipeline & Profit Margin Over Time")
    ax2.set_xlabel("Month"); ax2.set_ylabel("Amount ($)")
    fig2.legend(handles=[l1,l2,l3,l4,l5], loc="upper right")
    fig2.tight_layout(rect=[0,0,0.8,1])
    st.pyplot(fig2)
