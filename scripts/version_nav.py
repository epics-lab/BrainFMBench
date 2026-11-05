#!/usr/bin/env python3
"""Add a version picker to every page of the built site.

The site root is the current main branch; each release tag v* lives at /<tag>/.
Usage: python scripts/version_nav.py site
"""
import os
import re
import sys


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "site"
    tags = sorted((d for d in os.listdir(root) if re.fullmatch(r"v\d+\.\d+\.\d+", d)
                   and os.path.isfile(os.path.join(root, d, "index.html"))),
                  key=lambda t: [int(x) for x in t[1:].split(".")], reverse=True)
    pages = [("latest", os.path.join(root, "index.html"))] + \
            [(t, os.path.join(root, t, "index.html")) for t in tags]
    for here, path in pages:
        if not os.path.isfile(path):
            continue
        up = "" if here == "latest" else "../"
        links = [f'<a href="{up}"{" class=on" if here == "latest" else ""}>latest</a>'] + \
                [f'<a href="{up}{t}/"{" class=on" if here == t else ""}>{t}</a>' for t in tags]
        note = "" if here == "latest" else (f' <span class="sub">You are viewing the archived {here} '
                                            f'release. <a href="{up}">Go to latest</a>.</span>')
        nav = ('<style>.vnav{font-size:13px;margin:4px 0 12px}.vnav a{margin-right:10px}'
               '.vnav a.on{font-weight:600;text-decoration:none;color:#222}</style>'
               f'<div class="vnav">Version: {" ".join(links)}{note}</div>')
        html = open(path).read()
        if 'class="vnav"' in html:
            continue
        html = html.replace("<body>\n", "<body>\n" + nav + "\n", 1)
        open(path, "w").write(html)
    print(f"version picker: latest + {len(tags)} release(s)")


if __name__ == "__main__":
    sys.exit(main())
