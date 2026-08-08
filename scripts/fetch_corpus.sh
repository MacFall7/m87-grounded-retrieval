#!/usr/bin/env bash
# Rebuild the demo corpus from public sources. The corpus is not vendored: it is
# 85 markdown documents from public M87 repositories, and duplicating them here
# would fork documentation that lives somewhere else.
set -euo pipefail
REPOS="M87-Spine-lite m87-governed-loop m87-governance-sandbox m87-audit-agent m87-governed-swarm governed-langgraph spine-lite-python spine-lite-jvm m87-governed-code-change-evidence"
TMP="$(mktemp -d)"; mkdir -p corpus
for r in $REPOS; do
  git clone -q --depth 1 "https://github.com/MacFall7/$r.git" "$TMP/$r" || continue
  (cd "$TMP/$r" && find . -name '*.md' -not -path './.git/*' -not -path './node_modules/*' \
    -exec bash -c 'mkdir -p "'"$OLDPWD"'/corpus/'"$r"'/$(dirname {})" && cp {} "'"$OLDPWD"'/corpus/'"$r"'/{}"' \; ) || true
done
rm -rf "$TMP"
echo "corpus ready: $(find corpus -name '*.md' | wc -l) documents"
