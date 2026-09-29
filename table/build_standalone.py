"""
=================================================================
  VBLS — Viability Boundaries of Living Systems
  Author:     Vasiliy Avdeev (independent researcher)
  Monograph:  VBLS_THEORY
=================================================================

  Copyright (c) 2026 Vasiliy Avdeev. Licensed under CC BY-NC-ND 4.0 -- see
  LICENSE in the repository root, or https://creativecommons.org/licenses/by-nc-nd/4.0/
  SPDX-License-Identifier: CC-BY-NC-ND-4.0
=================================================================
Build the single-file version of the VBLS table.

Takes vbls_table.html + vbls_table.css and writes vbls_table_standalone.html,
with the stylesheet inlined into a <style> block.

The two-file version is the source. Edit vbls_table.html / vbls_table.css and
re-run this script; never edit the standalone file directly.

Usage:
    python build_standalone.py
"""
import io
import os
import sys

HTML = u'vbls_table.html'
CSS = u'vbls_table.css'
OUT = u'vbls_table_standalone.html'
LINK = u'<link rel="stylesheet" href="vbls_table.css">'


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    html_p, css_p, out_p = (os.path.join(here, f) for f in (HTML, CSS, OUT))

    for p in (html_p, css_p):
        if not os.path.exists(p):
            sys.exit(u'not found: %s' % p)

    raw = open(html_p, 'rb').read()
    bom = raw[:3] == b'\xef\xbb\xbf'
    html = raw.decode('utf-8-sig')
    css = io.open(css_p, encoding='utf-8').read()

    n = html.count(LINK)
    if n != 1:
        sys.exit(u'expected exactly one stylesheet link, found %d' % n)

    # the stylesheet must not carry external dependencies of its own
    for bad in (u'url(', u'@import'):
        if bad in css:
            sys.exit(u'%s contains %s - inlining would break it' % (CSS, bad))

    single = html.replace(LINK, u'<style>\n/* %s - inlined */\n%s\n</style>' % (CSS, css))

    out = single.encode('utf-8')
    if bom:
        out = b'\xef\xbb\xbf' + out
    open(out_p, 'wb').write(out)
    print(u'%s written: %d KB (html %d KB + css %d KB)'
          % (OUT, len(out) // 1024, len(raw) // 1024, len(css.encode('utf-8')) // 1024))


if __name__ == '__main__':
    main()
