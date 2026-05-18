# Capital Team

## Manage dependencies for new projects with uv

```
capital-team/
├── .github/
│   └── workflows/
│       └── notion-sync.yml
├── notion/
│   ├── pyproject.toml    ← notion's own dependencies
│   └── sync.py
├── pyproject.toml        ← workspace root, lists all members
└── uv.lock               ← single lock file for the entire repo
```

The single uv.lock at the root covers all workspace members. When you add other-project/ later, just add it to members in the root pyproject.toml and run uv sync