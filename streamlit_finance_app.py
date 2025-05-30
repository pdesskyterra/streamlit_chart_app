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

        raw_clients = p.get("Client",{}).get("formula",{}).get("string","")
        clients = [c.strip() for c in raw_clients.split(",") if c.strip()]
        n = len(clients)
        if n == 0:
            continue

        # pull each client's tag
        raw_tags = p.get("Expense Category",{}).get("rollup",{}).get("array",[])
        tags = [e["select"]["name"] for e in raw_tags if e.get("type")=="select"]
        if len(tags)!=n:
            tags = ["Paid"]*n

        # paid vs. total‐potential shares
        paid_total = p.get("Paid Revenue",{}).get("rollup",{}).get("number",0) or 0
        calc_rev   = p.get("Calculated Revenue",{}).get("formula",{}).get("number",0) or 0
        paid_share = paid_total / n
        pot_share  = max(0, calc_rev-paid_total) / n

        # cost shares
        emp = p.get("Monthly Employee Cost",{}).get("formula",{}).get("number",0) or 0
        ovh = p.get("Overhead Costs",{}).get("number",0) or 0
        emp_share = emp/n
        ovh_share = ovh/n

        for client, tag in zip(clients, tags):
            rows.append({
                "Month": month,
                "Client": client,
                "Tag": tag,
                "Paid": paid_share if tag=="Paid" else 0.0,
                "Pot" : pot_share  if tag!="Paid" else 0.0,
                "Emp" : emp_share,
                "Ovh" : ovh_share
            })
    return pd.DataFrame(rows)

df = fetch_notion_data()
if df.empty:
    st.warning("No data found")
    st.stop()

# --- FILTER & AGGREGATE ---
months = ['February 2025','March 2025','April 2025',
          'May 2025','June 2025','July 2025','August 2025']
df['Month'] = pd.Categorical(df['Month'], categories=months, ordered=True)
df = df[df['Month'].notna()]

# aggregate per Month & Client (one row each)
df_month = (
    df
    .groupby(['Month','Client'], sort=False)
    .agg({'Paid':'sum','Pot':'sum','Emp':'sum','Ovh':'sum'})
    .reset_index()
)

# build a Tag lookup per Month×Client
tag_map = (
    df[['Month','Client','Tag']]
    .drop_duplicates()
    .set_index(['Month','Client'])
)

# month-level totals for line chart
monthly = df_month.groupby('Month').sum().reindex(months, fill_value=0)
revenue = monthly['Paid'] + monthly['Pot']
costs   = monthly['Emp']   + monthly['Ovh']
profit  = revenue - costs
margin  = np.where(revenue>0, profit/revenue*100, np.nan)

# color & hatch maps
clients = sorted(df_month['Client'].unique())
colors  = dict(zip(clients, plt.cm.tab20(np.linspace(0,1,len(clients)))))
hatches = {"Invoiced":"//","Committed":"xx","Proposal":".."}

# --- PLOTTING ---
tab1, tab2 = st.tabs(["Revenue & Expenses","Trends Over Time"])

with tab1:
    fig, ax = plt.subplots(figsize=(14,6))
    x = np.arange(len(months))
    width = 0.35

    # stacked revenue: solid Paid + hatch for Pot by Tag
    rev_base = np.zeros(len(months))
    for client in clients:
        sub = (
            df_month[df_month['Client']==client]
            .set_index('Month')
            .reindex(months, fill_value=0)
        )
        paid = sub['Paid'].values
        pot  = sub['Pot'].values
        # plot Paid
        ax.bar(x-width/2, paid, width, bottom=rev_base, color=colors[client])
        rev_base += paid
        # overlay Pot only if Tag != Paid
        for i,mon in enumerate(months):
            tag = tag_map.reindex([(mon,client)]).iloc[0]['Tag']
            if pot[i]>0 and tag in hatches:
                ax.bar(x[i]-width/2, pot[i], width,
                       bottom=rev_base[i],
                       color=colors[client],
                       hatch=hatches[tag],
                       edgecolor='black')
        rev_base += pot

    # stacked costs
    cost_base = np.zeros(len(months))
    ax.bar(x+width/2, monthly['Emp'].values, width, bottom=cost_base, color='#d62728', label='Employee Cost')
    cost_base += monthly['Emp'].values
    ax.bar(x+width/2, monthly['Ovh'].values, width, bottom=cost_base, color='#9467bd', label='Overhead Cost')

    # highlight negatives
    for i in range(len(months)):
        if profit.iloc[i]<0:
            ax.bar(x[i], revenue.iloc[i], width*2,
                   fill=False, edgecolor='red', linewidth=2)

    # formatting
    ax.set_xticks(x)
    ax.set_xticklabels([m[:3]+" "+m.split()[1] for m in months], rotation=45)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y,_: f"${y:,.0f}"))
    ax.set_title("Revenue & Expenses by Expense Category")
    ax.set_xlabel("Month"); ax.set_ylabel("Amount ($)")

    # legends
    client_h = [Patch(facecolor=colors[c], label=c) for c in clients]
    tag_h    = [Patch(facecolor="white", edgecolor="black", hatch=v, label=k)
                for k,v in hatches.items()]
    cost_h   = [Patch(facecolor=c, label=l)
                for l,c in zip(["Employee Cost","Overhead Cost"],["#d62728","#9467bd"])]

    leg1 = ax.legend(handles=client_h, title="Clients",
                     loc="upper right", bbox_to_anchor=(0.95,0.6))
    ax.add_artist(leg1)
    ax.legend(handles=tag_h+cost_h, title="Categories",
              loc="upper right", bbox_to_anchor=(0.95,0.3))

    fig.tight_layout(rect=[0,0,0.8,1])
    st.pyplot(fig)

with tab2:
    fig2, ax2 = plt.subplots(figsize=(14,6))
    l1, = ax2.plot(x, revenue, 'o-',  label='Revenue')
    l2, = ax2.plot(x, profit,  's--', label='Profit')
    ax3 = ax2.twinx()
    l3, = ax3.plot(x, margin, 'd-.', label='Margin (%)')
    ax3.set_ylabel("Margin (%)"); ax3.yaxis.set_major_formatter(FuncFormatter(lambda p,_: f"{p:.0f}%"))

    for i in range(len(x)):
        ax2.annotate(f"${revenue.iloc[i]:,.0f}", (x[i], revenue.iloc[i]),
                     xytext=(0,10), textcoords='offset points', ha='center')
        ax2.annotate(f"${profit.iloc[i]:,.0f}",  (x[i], profit.iloc[i]),
                     xytext=(0,-10), textcoords='offset points', ha='center')
        if not np.isnan(margin[i]):
            ax3.annotate(f"{margin[i]:.0f}%", (x[i], margin[i]),
                         xytext=(0,15), textcoords='offset points', ha='center')

    ax2.set_xticks(x)
    ax2.set_xticklabels([m[:3]+" "+m.split()[1] for m in months], rotation=45)
    ax2.set_title("Revenue & Profit Trends"); ax2.set_xlabel("Month"); ax2.set_ylabel("Amount ($)")
    fig2.legend(handles=[l1,l2,l3], loc='upper right')
    fig2.tight_layout(rect=[0,0,0.8,1])
    st.pyplot(fig2)
