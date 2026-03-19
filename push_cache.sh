#!/bin/bash
# Kjør etter: brew install git-lfs

set -e
cd "$(dirname "$0")"

echo "1. Aktiverer Git LFS..."
git lfs install

echo "2. Sporer store cache-filer med LFS..."
git lfs track "cache/embeddings.npy" "cache/faiss.index"

echo "3. Legger til filer..."
git add .gitattributes cache/

echo "4. Committer..."
git commit -m "Add cache with LFS"

echo "5. Pusher..."
git push origin Nils_submissions

echo "Ferdig!"
