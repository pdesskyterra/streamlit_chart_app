# Debugging Guide for Cost Display Issues

## What the Debug Output Should Show You

When you run the updated app, you should see detailed debugging information that will help us identify why costs aren't displaying. Here's what to look for:

### 1. Database Connection Status
- ✅ **Success messages**: "Found X employees with date filtering" and "Found X cost items with date filtering"
- ⚠️ **Warning messages**: "No employee data found" or "No cost tracker data found"

### 2. Employee/Cost Data Details
If databases are connected, you'll see lists like:
```
- John Doe: $5,000/month, Start: 2025-01-01, End: None
- Jane Smith: $4,500/month, Start: 2024-12-01, End: 2025-03-31
```

### 3. Monthly Cost Calculations
For each month, you'll see:
```
🔍 February 2025: Filtered costs - Employee: $9,500, Overhead: $2,000
👥 February 2025: Shares for 3 clients - Employee: $3,167, Overhead: $667
```

### 4. Final Cost Summary
At the end, you'll see:
```
### 💰 Cost Data Summary
February 2025: Employee: $9,500 | Overhead: $2,000 | Total: $11,500
March 2025: Employee: $5,000 | Overhead: $2,000 | Total: $7,000
```

## Troubleshooting Based on Debug Output

### If you see "No employee data found" or "No cost tracker data found":
1. Check that `EMPLOYEE_DB_ID` and `COST_TRACKER_DB_ID` are correctly set in secrets
2. Verify the database IDs are correct in Notion
3. Ensure the Employee and Cost Tracker databases have the required columns

### If you see costs calculated but charts show $0:
1. Check if the month format matches between databases (should be "February 2025" format)
2. Verify that start/end dates are in YYYY-MM-DD format
3. Look at the date range - costs might be filtered out due to date ranges

### If filtered costs show $0 but fallback costs show values:
1. Check the date ranges in your Employee and Cost Tracker databases
2. The current month might be outside the active date ranges for employees/costs

## Next Steps After Testing

Once you run the app and see the debug output, share what messages you're seeing so I can:
1. Remove the verbose debugging
2. Fix any specific issues identified
3. Ensure both revenue and cost columns display accurately in the charts