import urllib.request
import re
import os
from html import unescape

url = 'https://rikiyat.github.io/blog/weight-space-programming/'
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
html = urllib.request.urlopen(req).read().decode('utf-8')

m = re.search(r'<d-article[^>]*>(.*?)</d-article>', html, re.DOTALL)
body = m.group(1) if m else html

# Convert headings
body = re.sub(r'<h1[^>]*>(.*?)</h1>', r'\n# \1\n', body, flags=re.DOTALL)
body = re.sub(r'<h2[^>]*>(.*?)</h2>', r'\n## \1\n', body, flags=re.DOTALL)
body = re.sub(r'<h3[^>]*>(.*?)</h3>', r'\n### \1\n', body, flags=re.DOTALL)
body = re.sub(r'<h4[^>]*>(.*?)</h4>', r'\n#### \1\n', body, flags=re.DOTALL)

# Convert code blocks
body = re.sub(r'<pre[^>]*><code[^>]*>(.*?)</code></pre>', lambda m: '\n```\n' + unescape(m.group(1)).strip() + '\n```\n', body, flags=re.DOTALL)
body = re.sub(r'<code[^>]*>(.*?)</code>', lambda m: '`' + unescape(m.group(1)) + '`', body, flags=re.DOTALL)

# Convert blockquotes
body = re.sub(r'<blockquote[^>]*>(.*?)</blockquote>', lambda m: '\n> ' + '\n> '.join(unescape(m.group(1)).strip().splitlines()) + '\n', body, flags=re.DOTALL)

# Convert tables
body = re.sub(r'<th[^>]*>(.*?)</th>', r' \1 |', body, flags=re.DOTALL)
body = re.sub(r'<td[^>]*>(.*?)</td>', r' \1 |', body, flags=re.DOTALL)
body = re.sub(r'<tr[^>]*>', '\n|', body)
body = re.sub(r'</tr>', '|\n', body)

# Convert paragraphs and breaks
body = re.sub(r'<p[^>]*>(.*?)</p>', r'\n\1\n', body, flags=re.DOTALL)
body = re.sub(r'<br\s*/?>', '\n', body)

# Convert links
body = re.sub(r'<a\s+[^>]*href=["\']([^"\']*)["\'][^>]*>(.*?)</a>', r'[\2](\1)', body, flags=re.DOTALL)

# Convert lists
body = re.sub(r'<li[^>]*>(.*?)</li>', r'\n- \1', body, flags=re.DOTALL)
body = re.sub(r'</?[uo]l[^>]*>', '\n', body)

# Strip remaining tags
body = re.sub(r'<[^>]+>', '', body)
body = unescape(body)

# Clean extra blank lines
lines = [l.rstrip() for l in body.splitlines()]
cleaned = []
blank = False
for l in lines:
    if not l:
        if not blank:
            cleaned.append('')
            blank = True
    else:
        cleaned.append(l)
        blank = False

content_md = '\n'.join(cleaned).strip()

frontmatter = """---
source: https://rikiyat.github.io/blog/weight-space-programming/
title: "Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights"
author: Rikiya Takehi, Jayden Teoh, Alex Zhang, Omar Khattab
date: 2026-11-02
fetched: 2026-11-03
---

# Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights

> Authors: Rikiya Takehi, Jayden Teoh, Alex Zhang, Omar Khattab
> Published: 2026-11-02
> Source: https://rikiyat.github.io/blog/weight-space-programming/

"""

out_path = os.path.join("raw", "weight-space-programming", "weight-space-programming.md")
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "w", encoding="utf-8") as f:
    f.write(frontmatter + content_md)

print("Saved raw source to", out_path, "total length:", len(frontmatter + content_md))
