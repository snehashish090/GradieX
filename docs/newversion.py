#!/usr/bin/env python3
"""Cut a new version of the GradieX documentation.

    python docs/newversion.py 0.0.2

Copies the newest docs/vX.Y.Z directory to the new version, restamps the version
string throughout, demotes every older version (marking it as superseded and
adding a notice that links to the current release), and regenerates the version
list and release table in docs/index.html.

It does NOT touch pyproject.toml or python/gradiex_module.cpp -- update the
version there yourself and rebuild, so that gradiex.__version__ matches the
directory these pages live in.
"""

import os
import re
import shutil
import sys
from datetime import date

DOCS = os.path.dirname(os.path.abspath(__file__))
PAGES = ("index.html", "cpp.html", "python.html")

NOTICE_START = "<!-- OLD-VERSION-NOTICE-START -->"
NOTICE_END = "<!-- OLD-VERSION-NOTICE-END -->"


def parse(v):
    """'0.10.2' -> (0, 10, 2), so versions sort numerically rather than as text."""
    return tuple(int(p) for p in v.split("."))


def discover():
    """Every version directory present, newest first."""
    found = []
    for name in os.listdir(DOCS):
        if name.startswith("v") and os.path.isdir(os.path.join(DOCS, name)):
            body = name[1:]
            if re.fullmatch(r"\d+(\.\d+)*", body):
                found.append(body)
    return sorted(found, key=parse, reverse=True)


def strip_notice(s):
    return re.sub(re.escape(NOTICE_START) + r".*?" + re.escape(NOTICE_END),
                  "", s, flags=re.S)


def set_current(path, is_current, newest):
    """Mark a page as the current release or as superseded."""
    s = strip_notice(open(path).read())

    s = s.replace('<FONT SIZE="-2">(older release)</FONT>',
                  '<FONT SIZE="-2">(current)</FONT>')
    if not is_current:
        s = s.replace('<FONT SIZE="-2">(current)</FONT>',
                      '<FONT SIZE="-2">(older release)</FONT>')
        page = os.path.basename(path)
        notice = (
            NOTICE_START + "\n"
            '<TABLE BORDER="0" CELLPADDING="6" CELLSPACING="0" WIDTH="100%">\n'
            '<TR><TD BGCOLOR="#FFEEEE">&nbsp;&nbsp;\n'
            '  <FONT FACE="Verdana,Arial,Helvetica,sans-serif" SIZE="-1" COLOR="#AA0000">\n'
            '  <B>This documentation is for an older release.</B>\n'
            '  The current version is\n'
            '  <A HREF="../v' + newest + '/' + page + '">v' + newest + '</A>.\n'
            '  </FONT>&nbsp;\n'
            '</TD></TR>\n'
            '</TABLE>\n'
            + NOTICE_END + "\n")
        # sits directly below the banner, above the three-column table
        marker = "<!-- ==================== three columns ==================== -->"
        s = s.replace(marker, notice + marker, 1)

    open(path, "w").write(s)


def rewrite_index(versions, dates):
    """Regenerate the version list and release table in docs/index.html."""
    path = os.path.join(DOCS, "index.html")
    s = open(path).read()
    newest = versions[0]

    rows = []
    for i, v in enumerate(versions):
        mark = "<B>&#187; v%s</B>&nbsp;<FONT SIZE=\"-2\">(current)</FONT>" % v \
            if i == 0 else "&nbsp;&nbsp;<A HREF=\"v%s/index.html\">v%s</A>" % (v, v)
        rows.append(mark + "<BR>")
    listing = ('<FONT FACE="Verdana,Arial,Helvetica,sans-serif" SIZE="-1">\n'
               + "\n".join(rows) + "</FONT>")
    s = re.sub(re.escape("<!-- VERSION-LIST-START -->") + r".*?"
               + re.escape("<!-- VERSION-LIST-END -->"),
               "<!-- VERSION-LIST-START -->\n" + listing + "\n<!-- VERSION-LIST-END -->",
               s, flags=re.S)

    trows = []
    for i, v in enumerate(versions):
        trows.append(
            "<TR>\n"
            "  <TD><B>v%s</B></TD>\n" % v +
            "  <TD>%s</TD>\n" % ("Current" if i == 0 else "Superseded") +
            "  <TD>%s</TD>\n" % dates.get(v, "&mdash;") +
            '  <TD><A HREF="v%s/index.html">home</A> &#183;\n' % v +
            '      <A HREF="v%s/cpp.html">c++</A> &#183;\n' % v +
            '      <A HREF="v%s/python.html">python</A></TD>\n' % v +
            "</TR>")
    s = re.sub(re.escape("<!-- RELEASE-ROWS-START -->") + r".*?"
               + re.escape("<!-- RELEASE-ROWS-END -->"),
               "<!-- RELEASE-ROWS-START -->\n" + "\n".join(trows)
               + "\n<!-- RELEASE-ROWS-END -->",
               s, flags=re.S)

    # banner links, the doctest example and the footer all point at the newest
    s = re.sub(r'HREF="v\d+(?:\.\d+)*/(index|cpp|python)\.html"><FONT COLOR="#FFFFFF">',
               lambda m: 'HREF="v%s/%s.html"><FONT COLOR="#FFFFFF">' % (newest, m.group(1)), s)
    s = re.sub(r"'\d+(?:\.\d+)*'", "'%s'" % newest, s)
    s = re.sub(r'<A HREF="v\d+(?:\.\d+)*/index\.html">current documentation</A>',
               '<A HREF="v%s/index.html">current documentation</A>' % newest, s)
    s = re.sub(r'python docs/newversion\.py \d+(?:\.\d+)*',
               'python docs/newversion.py %s' % bump(newest), s)

    open(path, "w").write(s)


def bump(v):
    parts = parse(v)
    return ".".join(str(p) for p in parts[:-1] + (parts[-1] + 1,))


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: python docs/newversion.py <version>   e.g. 0.0.2")
    new = sys.argv[1].lstrip("v")
    if not re.fullmatch(r"\d+(\.\d+)*", new):
        sys.exit("version must look like 0.0.2, got %r" % new)

    existing = discover()
    if not existing:
        sys.exit("no existing docs/vX.Y.Z directory to copy from")
    if new in existing:
        sys.exit("docs/v%s already exists" % new)
    source = existing[0]
    if parse(new) <= parse(source):
        sys.exit("%s is not newer than the current %s" % (new, source))

    src = os.path.join(DOCS, "v" + source)
    dst = os.path.join(DOCS, "v" + new)
    shutil.copytree(src, dst)

    for page in PAGES:
        p = os.path.join(dst, page)
        if not os.path.exists(p):
            continue
        s = strip_notice(open(p).read()).replace(source, new)
        open(p, "w").write(s)

    versions = discover()
    for i, v in enumerate(versions):
        for page in PAGES:
            p = os.path.join(DOCS, "v" + v, page)
            if os.path.exists(p):
                set_current(p, i == 0, versions[0])

    today = date.today().strftime("%d %B %Y").lstrip("0")
    dates = {new: today}
    # keep whatever dates the index already records for older versions
    idx = open(os.path.join(DOCS, "index.html")).read()
    for v in versions:
        m = re.search(r"<TD><B>v" + re.escape(v) + r"</B></TD>\s*<TD>[^<]*</TD>\s*<TD>([^<]*)</TD>", idx)
        if m and v != new:
            dates[v] = m.group(1).strip()
    rewrite_index(versions, dates)

    print("created docs/v%s (copied from v%s)" % (new, source))
    print("marked v%s as current; %d older version(s) demoted"
          % (new, len(versions) - 1))
    print("\nremember to update the version in:")
    print("  pyproject.toml")
    print("  python/gradiex_module.cpp   (m.attr(\"__version__\"))")
    print("then rebuild:  pip install -e .")


if __name__ == "__main__":
    main()
