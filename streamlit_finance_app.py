import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
from matplotlib.patches import Patch
from notion_client import Client

# --- CONFIG ---
NOTION_TOKEN = "ntn_C39727399952EQvWgyq93N3o2dpj78XQZjhLfCpYqXs2vo"
DATABASE_ID = "1f50eccb339b802699b7d1d1e0d08134"

# --- STREAMLIT PAGE SETUP ---
st.set_page_config(layout="wide")
st.title("📊 Profit & Expense Tracker")

# --- LOAD DATA FROM NOTION ---
@st.cache_data(ttl=600)
def fetch_notion_data():
    notion = Client(auth=NOTION_TOKEN)
    results = []
    response = notion.databases.query(database_id=DATABASE_ID)
    for result in response['results']:
        props = result['properties']
        try:
            row = {
                "Month": props["Month"]['select']['name'],
                "Client": props["Client"]['rich_text'][0]['plain_text'],
                "Paid Revenue": props["Paid Revenue"]['number'],
                "Potential Revenue": float(props["Potential Revenue"]['rich_text'][0]['plain_text'].replace('$','').replace(',','')),
                "Monthly Employee Cost": props["Monthly Employee Cost"]['number'],
                "Overhead Costs": props["Overhead Costs"]['number']
            }
            results.append(row)
        except Exception as e:
            print("Skipping row due to error:", e)
            continue
    return pd.DataFrame(results)

df = fetch_notion_data()

if df.empty:
    st.warning("No data found or Notion credentials are incorrect.")
    st.stop()

# --- PROCESS DATA ---
month_order = [
    'November 2024', 'December 2024', 'January 2025', 'February 2025',
    'March 2025', 'April 2025', 'May 2025', 'June 2025',
    'July 2025', 'August 2025', 'September 2025', 'October 2025'
]

df['Month'] = pd.Categorical(df['Month'], categories=month_order, ordered=True)
df = df.sort_values('Month')

# Split and expand clients if needed
processed_data = []
for _, row in df.iterrows():
    clients = [c.strip() for c in row['Client'].split(',')]
    paid = row['Paid Revenue']
    potential = row['Potential Revenue']
    emp_cost = row['Monthly Employee Cost']
    overhead = row['Overhead Costs']
    
    paid_per_client = paid / len(clients)
    potential_per_client = potential / len(clients)
    emp_cost_per_client = emp_cost / len(clients)
    overhead_per_client = overhead / len(clients)
    
    for client in clients:
        processed_data.append({
            'Month': row['Month'],
            'Client': client,
            'Paid Revenue': paid_per_client,
            'Potential Revenue': potential_per_client,
            'Monthly Employee Cost': emp_cost_per_client,
            'Overhead Costs': overhead_per_client
        })

processed_df = pd.DataFrame(processed_data)

# --- AGGREGATE FOR PLOTTING ---
months = sorted(processed_df['Month'].unique(), key=lambda x: month_order.index(x))
clients = sorted(processed_df['Client'].unique())

client_colors = {
    'Atlas Power Technologies': '#1f77b4',
    'Leap Manufacturing': '#ff7f0e',
    'Roxbox': '#2ca02c',
    'LauryTMG': '#d62728',
    'Group1': '#9467bd',
    'Origis Energy': '#8c564b',
    'Synseer': '#e377c2'
}

bar_width = 0.8
month_indices = {month: i for i, month in enumerate(months)}
client_data_by_month = {client: np.zeros(len(months)) for client in clients}
potential_data_by_month = {client: np.zeros(len(months)) for client in clients}
employee_costs = np.zeros(len(months))
overhead_costs = np.zeros(len(months))

for _, row in processed_df.iterrows():
    m = month_indices[row['Month']]
    c = row['Client']
    client_data_by_month[c][m] += row['Paid Revenue']
    potential_data_by_month[c][m] += row['Potential Revenue']
    employee_costs[m] += row['Monthly Employee Cost']
    overhead_costs[m] += row['Overhead Costs']

# --- PLOT ---
fig, ax = plt.subplots(figsize=(14, 10))
paid_bottom = np.zeros(len(months))
for client in clients:
    ax.bar(range(len(months)), client_data_by_month[client], bottom=paid_bottom,
           width=bar_width, color=client_colors.get(client, 'gray'), label=client)
    paid_bottom += client_data_by_month[client]

# Potential revenue
potential_bottom = paid_bottom.copy()
for client in clients:
    potential_diff = np.maximum(potential_data_by_month[client] - client_data_by_month[client], 0)
    ax.bar(range(len(months)), potential_diff, bottom=potential_bottom,
           width=bar_width, color=client_colors.get(client, 'gray'), hatch='///', alpha=0.7)
    potential_bottom += potential_diff

# Costs
ax.bar(range(len(months)), -employee_costs, width=bar_width, color='#8c564b', hatch='xxx', label='Employee Costs')
ax.bar(range(len(months)), -overhead_costs, bottom=-employee_costs, width=bar_width,
       color='#e377c2', hatch='...', label='Overhead Costs')

# Profit lines
profit = paid_bottom - (employee_costs + overhead_costs)
potential_profit = potential_bottom - (employee_costs + overhead_costs)
ax.plot(range(len(months)), profit, 'k--', linewidth=2, label='Profit')
ax.plot(range(len(months)), potential_profit, 'k-', linewidth=2, label='Potential Profit')

# X-axis
ax.set_xticks(range(len(months)))
ax.set_xticklabels([m.split(' ')[0][:3] + ' ' + m.split(' ')[1] for m in months], rotation=45)

# Y-axis formatting
def currency(x, pos): return f"${x:,.0f}"
ax.yaxis.set_major_formatter(FuncFormatter(currency))

# Titles and labels
plt.title('Client Revenue and Expenses by Month', fontsize=16, pad=20)
plt.ylabel('Amount ($)')
plt.xlabel('Month')
ax.grid(axis='y', linestyle='--', alpha=0.6)

# Legends
client_handles = [Patch(color=client_colors.get(c, 'gray'), label=c) for c in clients]
client_legend = plt.legend(handles=client_handles, title='Clients', loc='upper left', bbox_to_anchor=(1.01, 1))
plt.gca().add_artist(client_legend)

category_handles = [
    Patch(facecolor='gray', label='Paid Revenue'),
    Patch(facecolor='gray', hatch='///', label='Potential Revenue', alpha=0.7),
    Patch(facecolor='#8c564b', hatch='xxx', label='Employee Costs'),
    Patch(facecolor='#e377c2', hatch='...', label='Overhead Costs'),
    plt.Line2D([0], [0], color='k', linestyle='--', label='Profit'),
    plt.Line2D([0], [0], color='k', linestyle='-', label='Potential Profit')
]
plt.legend(handles=category_handles, title='Categories', loc='upper left', bbox_to_anchor=(1.01, 0.5))

plt.tight_layout()
plt.subplots_adjust(right=0.85)

# --- DISPLAY ---
st.pyplot(fig)
