# Date-Based Cost Filtering Configuration

## Overview
The finance app now supports date-based filtering for both employee costs and overhead costs using separate Employee Table and Cost Tracker databases in Notion.

## Configuration

### Streamlit Secrets
Add the following to your `.streamlit/secrets.toml` file:

```toml
# Required - Main database
NOTION_TOKEN = "your_notion_token"
DATABASE_ID = "your_main_database_id"

# Optional - Employee Table database
EMPLOYEE_DB_ID = "your_employee_table_database_id"

# Optional - Cost Tracker database  
COST_TRACKER_DB_ID = "your_cost_tracker_database_id"
```

### Database Structure Requirements

#### Employee Table Database
Required columns:
- `Name` (Title field) - Employee name
- `Start Date` (Date field) - When employee started
- `End Date` (Date field) - When employee ended (optional)
- `Employee Cost` (Number field) - Monthly cost for this employee

#### Cost Tracker Database
Required columns:
- `Cost Item` (Title field) - Description of the cost
- `Start Date` (Date field) - When cost becomes active
- `End Date` (Date field) - When cost becomes inactive (optional)
- `Active Costs/Month` (Number field) - Monthly cost amount
- `Category` (Select field) - Cost category (optional)

## How It Works

1. **Employee Cost Filtering**: For each month, only includes costs for employees who were active (between their start and end dates)

2. **Overhead Cost Filtering**: For each month, only includes costs for items that were active during that month

3. **Fallback**: If Employee or Cost Tracker databases are not configured, falls back to the original logic using "Monthly Employee Cost" and "Overhead Costs" fields from the main database

4. **Date Format**: All dates should be in YYYY-MM-DD format in Notion

## Benefits

- **Accurate Historical Data**: Costs are only applied to months when employees/expenses were actually active
- **Flexible Workforce**: Handles employees starting/ending at different times
- **Dynamic Overhead**: Tracks when different overhead costs were added/removed
- **Backward Compatible**: Works with existing setups without requiring immediate migration

## Testing

The app will show warnings if it cannot fetch employee or cost tracker data, but will continue to function using the fallback logic.