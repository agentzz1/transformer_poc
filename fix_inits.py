import os
import re

# Matches 'signal name : integer;' and adds ' := 0' initialization
# Regex explanation:
# signal  -> literal 'signal'
# \s+     -> one or more spaces
# ([^:]+) -> capture signal name(s)
# :       -> literal colon
# \s+     -> one or more spaces
# integer -> literal 'integer'
# \s*     -> optional spaces
# ;       -> literal semicolon
pattern = re.compile(r'(signal\s+[^:]+:\s+integer)\s*;')

for root, dirs, files in os.walk('.'):
    for file in files:
        if file.endswith('.vhd'):
            path = os.path.join(root, file)
            try:
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                new_content = pattern.sub(r'\1 := 0;', content)
                
                if content != new_content:
                    with open(path, 'w', encoding='utf-8') as f:
                        f.write(new_content)
                    print(f"Initialized signals in {path}")
            except Exception as e:
                print(f"Error fixing {path}: {e}")
