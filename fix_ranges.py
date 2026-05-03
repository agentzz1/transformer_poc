import os
import re

# Regex for 'integer range ...' until a semicolon, closing parenthesis, or newline
# This captures both 0 to 511 and 0 to max_size_x - 1
pattern = re.compile(r'integer\s+range\s+[^;)]+')

for root, dirs, files in os.walk('.'):
    for file in files:
        if file.endswith('.vhd'):
            path = os.path.join(root, file)
            try:
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                new_content = pattern.sub('integer', content)
                
                if content != new_content:
                    with open(path, 'w', encoding='utf-8') as f:
                        f.write(new_content)
                    print(f"Fixed {path}")
            except Exception as e:
                print(f"Error fixing {path}: {e}")
