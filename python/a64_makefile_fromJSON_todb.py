import json
import sqlite3
from pathlib import Path

# Load the JSON data from a file
file_path = r"C:\chungnam_chatbot\python"
with open(Path(file_path) / "params.json", "r", encoding="UTF8") as f:
    data = json.loads(f.read())

# Connect to the database (creates a new database if it doesn't exist)
conn = sqlite3.connect("example.db")

# Create a cursor object to execute SQL commands
c = conn.cursor()

# Create a table
c.execute(
    """CREATE TABLE students
             (name text, korean integer, math integer, english integer, science integer)"""
)

# Insert data into the table
for student in data:
    c.execute(
        "INSERT INTO students VALUES (?, ?, ?, ?, ?)",
        (
            student["name"],
            student["korean"],
            student["math"],
            student["english"],
            student["science"],
        ),
    )

# Save (commit) the changes
conn.commit()

# Close the connection
conn.close()
