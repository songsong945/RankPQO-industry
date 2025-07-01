import re

# Given template and query strings
template = "and ib_lower_bound >= %s * 10000\nand ib_upper_bound <= %s * 10000 + 50000"
query = "and ib_lower_bound >= 3 * 10000\nand ib_upper_bound <= 3 * 10000 + 50000"

# Regular expression to match the template against the query
# We're looking for the pattern and capturing the digits where '%s' is located
pattern = re.sub(r'%s', r'(\d+)', re.escape(template))

# Perform the search and extract the values
matches = re.search(pattern, query)

if matches:
    extracted_values = matches.groups()
else:
    extracted_values = []

print(extracted_values)