#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PDFLATEX=${PDFLATEX:-pdflatex}
BIBTEX=${BIBTEX:-bibtex}
XELATEX=${XELATEX:-xelatex}
if ! command -v "$BIBTEX" >/dev/null 2>&1 && command -v bibtex.original >/dev/null 2>&1; then BIBTEX=bibtex.original; fi
for f in fig_architecture fig_timeline; do
 (cd figures; "$PDFLATEX" -interaction=nonstopmode -halt-on-error "$f.tex" > "$f.build.log")
done
"$PDFLATEX" -interaction=nonstopmode -halt-on-error main.tex > build-pass1.log
"$BIBTEX" main > build-bibtex.log
"$PDFLATEX" -interaction=nonstopmode -halt-on-error main.tex > build-pass2.log
"$PDFLATEX" -interaction=nonstopmode -halt-on-error main.tex > build-pass3.log

# Chinese edition: XeLaTeX is required by ctex/xeCJK. Keep it as a separate
# job so the English and Chinese auxiliary files never collide.
"$XELATEX" -interaction=nonstopmode -halt-on-error main_zh.tex > build-zh-pass1.log
"$BIBTEX" main_zh > build-zh-bibtex.log
"$XELATEX" -interaction=nonstopmode -halt-on-error main_zh.tex > build-zh-pass2.log
"$XELATEX" -interaction=nonstopmode -halt-on-error main_zh.tex > build-zh-pass3.log
