import pandas as pd
import numpy as np
import sys

# Load power consumption log
# Filter only your model's PID (replace with actual PID)
if len(sys.argv) != 4:
    print(f"Usage: {sys.argv[0]} power-log.csv gpu-util.log PID")
    sys.exit(1)
your_pid = int(sys.argv[3])
power_log = pd.read_csv(sys.argv[1], skiprows=1)  # Skip header row
power_log.columns = ["timestamp", "Power (W)"]  # Rename column
power_log["Power (W)"] = power_log["Power (W)"].str.replace(" W", "", regex=False).astype(float)  # Convert to float

# inspect the log for issues
print(power_log.head())  # Inspect first few rows
print(power_log.dtypes)  # Check column types
#print(power_log["timestamp"].unique())  # See all unique timestamp values

# Check the exact data types in each row
print("checking exact datatypes in each row:")
print(power_log["timestamp"].apply(type).value_counts())

# Check for hidden or non-ASCII characters
#power_log["timestamp"].apply(lambda x: print(repr(x)))

# Ensure timestamp is string and clean
power_log["timestamp"] = power_log["timestamp"].astype(str).str.strip()
power_log["timestamp"] = pd.to_datetime(power_log["timestamp"], format="%Y/%m/%d %H:%M:%S.%f", errors='coerce').dt.floor("S")  # Remove milliseconds to match util_log
print(power_log.head())  # Check if values are correctly formatted as numbers

# Load GPU utilization log
util_log = pd.read_csv(sys.argv[2], delim_whitespace=True, comment="#",
                       names=["#Date", "Time", "gpu", "pid", "type", "sm", "mem", "enc", "dec", "jpg", "ofa", "fb", "ccpm", "command"])

# Convert both columns to string and strip whitespace
util_log["#Date"] = util_log["#Date"].astype(str).str.strip()
util_log["Time"] = util_log["Time"].astype(str).str.strip()

util_log["timestamp"] = pd.to_datetime(util_log["#Date"] + " " + util_log["Time"], errors='coerce', infer_datetime_format=True)

print(util_log.head())

print(util_log)

util_log["pid"] = pd.to_numeric(util_log["pid"], errors="coerce")  # Convert column to int
# filter util_log with pid
util_log = util_log[util_log["pid"] == your_pid]

# Drop any rows with missing timestamps
power_log = power_log.dropna(subset=["timestamp"])
print("power_log: ")
print(power_log)
util_log = util_log.dropna(subset=["timestamp"])
print("util_log: ")
print(util_log)

# Merge power log and utilization log based on closest timestamps
merged_data = pd.merge(
    power_log.sort_values("timestamp"),
    util_log.sort_values("timestamp"),
    on="timestamp",
    how="inner"  # Only keeps rows where timestamps match exactly
)
print(merged_data.head())  # Check if timestamps now match correctly
print (merged_data["timestamp"])

merged_data["Power (W)"] = pd.to_numeric(merged_data["Power (W)"], errors="coerce")
merged_data["sm"] = pd.to_numeric(merged_data["sm"], errors="coerce")

# Convert sm from percentage to fraction
merged_data["sm"] = merged_data["sm"].astype(float) / 100

print("Avg Power (W):" + str(merged_data["Power (W)"].mean()))
print("Avg SM value: " + str(merged_data["sm"].mean()))

# Compute energy consumption
sampling_interval = 1 / 3600  # Convert 1 sec to hours
total_energy_kwh = (merged_data["Power (W)"] * merged_data["sm"] * sampling_interval).sum() / 1000

print("Total Energy Consumption for PID "+ str(your_pid) + ": " + str(total_energy_kwh) + " kWh")
