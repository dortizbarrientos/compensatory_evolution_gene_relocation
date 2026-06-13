#!/usr/bin/env bash
# pre_push_cleanup.sh
# Run from the repository root, BEFORE your first commit/push.
# It (1) removes duplicates and junk, (2) reports anything heavy, so you
# can decide what to commit. It is conservative: it prints what it does.
set -uo pipefail

echo "==> Removing duplicates and junk (safe to re-run)"
for f in "repo_README.md" "run_all copy.sh" "Unconfirmed 417087.crdownload"; do
  if [ -e "$f" ]; then rm -f "$f" && echo "   removed: $f"; fi
done
# caches, OS cruft, incomplete downloads, LaTeX aux
find . -name "__pycache__" -type d -prune -exec rm -rf {} + 2>/dev/null && echo "   removed: __pycache__/"
find . -name ".DS_Store"   -delete 2>/dev/null && echo "   removed: .DS_Store files"
find . -name "*.crdownload" -delete 2>/dev/null
find . \( -name "*.aux" -o -name "*.log" -o -name "*.out" -o -name "*.toc" \
       -o -name "*.synctex.gz" -o -name "*.fls" -o -name "*.fdb_latexmk" \) -delete 2>/dev/null
echo "   removed: LaTeX build artifacts"

echo
echo "==> Total repo size (excluding .git):"
du -sh --exclude=.git . 2>/dev/null || du -sh .

echo
echo "==> Files larger than 50 MB (GitHub warns >50, BLOCKS >100):"
find . -path ./.git -prune -o -type f -size +50M -print 2>/dev/null | while read -r f; do
  printf "   %6s  %s\n" "$(du -h "$f" | cut -f1)" "$f"
done
echo "   (none listed above = you're clear of the hard limit)"

echo
echo "==> Count of regenerable SLiM tree-sequences (.trees) — these are .gitignore'd:"
echo "    $(find . -name '*.trees' 2>/dev/null | wc -l | tr -d ' ') files"

echo
echo "==> Largest 15 tracked-candidate items (sanity check):"
du -ah --exclude=.git . 2>/dev/null | sort -rh | grep -vE '/\.git' | head -15

echo
echo "Done. Now:  git add -A  &&  git status   (confirm no .trees / caches staged)"
echo "If you already 'git add'-ed the big stuff:  git rm -r --cached slim/**/archives* '*.trees'"
