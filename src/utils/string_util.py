import re

def extract_key_from_string(string, pattern):

    # Search for the pattern in the URL
    match = re.search(pattern, string)

    # If a match is found, extract the asset ID
    if match:
        asset_id = match.group(1)
        return asset_id
    else:
        return None