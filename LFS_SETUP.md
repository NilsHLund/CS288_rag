# Git LFS setup for cache

Cache-filene `embeddings.npy` og `faiss.index` er >100MB. Installer Git LFS:

```bash
brew install git-lfs
git lfs install
```

Deretter:

```bash
git lfs track "cache/embeddings.npy" "cache/faiss.index"
git add .gitattributes cache/ .gitignore
git commit -m "Add cache with LFS"
git push origin Nils_submissions
```
