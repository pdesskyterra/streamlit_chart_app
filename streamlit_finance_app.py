import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
from matplotlib.patches import Patch
from notion_client import Client

# --- CONFIG & PAGE SETUP ---
NOTION_TOKEN = st.secrets["NOTION_TOKEN"]
DATABASE_ID  = st.secrets["DATABASE_ID"]
st.set_page_config(layout="wide")
st.title("📊 Profit & Expense Tracker (By Expense Category)")

# --- FETCH & TAG DATA ---
@st.cache_data(ttl=600)
def fetch_notion_data():
    notion = Client(auth=NOTION_TOKEN)
    rows = []
    for page in notion.databases.query(database_id=DATABASE_ID)["results"]:
        p = page["properties"]
        month = p.get("Month",{}).get("select",{}).get("name")
        if not month:
            continue
        raw_clients = p.get("Client",{}).get("formula",{}).get("string","")
        clients = [c.strip() for c in raw_clients.split(",") if c.strip()]
        n = len(clients)
        if n==0:
            continue
        # Expense-tag per client
        raw_tags = p.get("Expense Category",{}).get("rollup",{}).get("array",[])
        tags = [e.get("select",{}).get("name","") for e in raw_tags if e.get("type")=="select"]
        if len(tags)!=n:
            tags = ["Paid"]*n
        # Paid vs. Potential shares
        paid_total = p.get("Paid Revenue",{}).get("rollup",{}).get("number",0) or 0
        calc_rev   = p.get("Calculated Revenue",{}).get("formula",{}).get("number",0) or 0
        paid_share = paid_total/n
        pot_share  = max(0, calc_rev-paid_total)/n
        # Costs
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
                "Pot":  pot_share  if tag!="Paid" else 0.0,
                "Emp":  emp_share,
                "Ovh":  ovh_share
            })
    return pd.DataFrame(rows)

# Load data
df = fetch_notion_data()
if df.empty:
    st.warning("No data found or invalid credentials.")
    st.stop()

# Months filter
months = ['February 2025','March 2025','April 2025','May 2025',
          'June 2025','July 2025','August 2025']
df['Month'] = pd.Categorical(df['Month'], categories=months, ordered=True)
df = df[df['Month'].notna()]

# Month×Client summary
df_mc = (
    df.groupby(['Month','Client'], sort=False)
      .sum()[['Paid','Pot','Emp','Ovh']]
      .reset_index()
)

# Tag map for hatches
tag_map = (
    df[['Month','Client','Tag']]
      .drop_duplicates()
      .set_index(['Month','Client'])['Tag']
)

# Pivot for easy plotting
paid_df = df_mc.pivot(index='Month', columns='Client', values='Paid').reindex(months, fill_value=0)
pot_df  = df_mc.pivot(index='Month', columns='Client', values='Pot').reindex(months, fill_value=0)

# Totals for line chart
monthly = df_mc.groupby('Month')[['Paid','Pot','Emp','Ovh']].sum().reindex(months, fill_value=0)
revenue = monthly['Paid'] + monthly['Pot']
costs   = monthly['Emp']   + monthly['Ovh']
profit  = revenue - costs
margin  = np.where(revenue>0, profit/revenue*100, np.nan)

# Plot prep
clients = list(paid_df.columns)
colors  = dict(zip(clients, plt.cm.tab20(np.linspace(0,1,len(clients)))))
hatches = {"Invoiced":"//", "Committed":"xx", "Proposal":".."}
x = np.arange(len(months))
w = 0.35

# --- CHARTS ---
tab1, tab2 = st.tabs(["Revenue & Expenses","Trends Over Time"])

with tab1:
    fig, ax = plt.subplots(figsize=(16,8))
    base = np.zeros(len(months))
    for c in clients:
        paid = paid_df[c].values
        pot  = pot_df[c].values
        # solid Paid
        ax.bar(x-w/2, paid, w, bottom=base, color=colors[c])
        # hatched Pot per Tag
        for i, mon in enumerate(months):
            tag = tag_map.get((mon,c), "Paid")
            if pot[i]>0 and tag in hatches:
                ax.bar(x[i]-w/2, pot[i], w,
                       bottom=base[i],
                       color=colors[c],
                       hatch=hatches[tag],
                       edgecolor='black')
        base += paid + pot
    # expenses
    cbase = np.zeros(len(months))
    ax.bar(x+w/2, monthly['Emp'], w, bottom=cbase, color='#d62728', label='Employee Cost')
    cbase += monthly['Emp']
    ax.bar(x+w/2, monthly['Ovh'], w, bottom=cbase, color='#9467bd', label='Overhead Cost')
    # highlight negative
    for i in range(len(months)):
        if profit.iloc[i]<0:
            ax.bar(x[i], revenue.iloc[i], w*2,
                   fill=False, edgecolor='red', linewidth=2)
    # format
    ax.set_xticks(x)
    ax.set_xticklabels([m[:3]+" "+m.split()[1] for m in months], rotation=45)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y,_: f"${y:,.0f}"))
    ax.set_title("Revenue & Expenses by Expense Category")
    ax.set_xlabel("Month"); ax.set_ylabel("Amount ($)")
    # legends
    client_h = [Patch(facecolor=colors[c], label=c) for c in clients]
    tag_h    = [Patch(facecolor='white', edgecolor='black', hatch=h, label=t)
                for t,h in hatches.items()]
    cost_h   = [Patch(facecolor='#d62728', label='Employee Cost'),
                Patch(facecolor='#9467bd', label='Overhead Cost')]
    leg1 = ax.legend(handles=client_h, title='Clients',
                     loc='upper right', bbox_to_anchor=(0.98,0.6))
    ax.add_artist(leg1)
    ax.legend(handles=tag_h+cost_h, title='Expense Categories',
              loc='upper right', bbox_to_anchor=(0.98,0.3))
    fig.tight_layout(rect=[0,0,0.8,1])
    st.pyplot(fig)

with tab2:
    fig2, ax2 = plt.subplots(figsize=(16,8))
    l1, = ax2.plot(x, revenue, 'o-',  label='Revenue')
    l2, = ax2.plot(x, profit,  's--', label='Profit')
    ax3 = ax2.twinx()
    l3, = ax3.plot(x, margin, 'd-.', label='Margin (%)')
    ax3.set_ylabel('Margin (%)')
    ax3.yaxis.set_major_formatter(FuncFormatter(lambda p,_: f"{p:.0f}%"))
    for i in range(len(x)):
        ax2.annotate(f"${revenue.iloc[i]:,.0f}", (x[i], revenue.iloc[i]),
                     textcoords='offset points', xytext=(0,10), ha='center')
        ax2.annotate(f"${profit.iloc[i]:,.0f}",   (x[i], profit.iloc[i]),
                     textcoords='offset points', xytext=(0,-10), ha='center')
        if not np.isnan(margin[i]):
            ax3.annotate(f"{margin[i]:.0f}%", (x[i], margin[i]),
                         textcoords='offset points', xytext=(0,15), ha='center')
    ax2.set_xticks(x)
    ax2.set_xticklabels([m[:3]+" "+m.split()[1] for m in months], rotation=45)
    ax2.set_title('Revenue & Profit Trends Over Time')
    ax2.set_xlabel('Month'); ax2.set_ylabel('Amount ($)')
    fig2.legend(handles=[l1,l2,l3], loc='upper right')
    fig2.tight_layout(rect=[0,0,0.8,1])
    st.pyplot(fig2)
