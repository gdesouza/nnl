---
name: using-serena-mcp
description: "Navigate and understand source code using Serena MCP's symbolic tools. Use when exploring code structure, finding symbols, tracing references, reading file outlines, or performing symbol-level code navigation."
---

# Using Serena MCP for Code Navigation

Serena provides IDE-grade symbolic code navigation via MCP. Prefer Serena's semantic tools over raw grep/read when exploring code structure, finding definitions, tracing references, or understanding symbol relationships.

## Activation

At the start of each session, activate the project before using any Serena tools:

1. Call `mcp__serena__activate_project` with the workspace directory path
2. Call `mcp__serena__initial_instructions` to load Serena's usage instructions
3. Call `mcp__serena__check_onboarding_performed` — if onboarding hasn't been done, run `mcp__serena__onboarding`

## Core Navigation Tools

### Symbol Search
- **`mcp__serena__find_symbol`** — Global or local symbol search via the language server. Use this instead of grep when looking for functions, types, structs, traits, or modules by name.

### File Outline
- **`mcp__serena__get_symbols_overview`** — Get top-level symbols defined in a file. Use this instead of reading an entire file when you just need to understand its structure.

### Reference Tracing
- **`mcp__serena__find_referencing_symbols`** — Find all symbols that reference a given symbol. Use this to trace callers, implementors, or usages across the codebase.

### File Operations
- **`mcp__serena__read_file`** — Read a file or a specific range within it.
- **`mcp__serena__find_file`** — Find files by name/path pattern.
- **`mcp__serena__list_dir`** — List directory contents.
- **`mcp__serena__search_for_pattern`** — Regex search across the project.

## When to Use Serena vs Built-in Tools

| Task | Use Serena | Use Built-in |
|------|-----------|-------------|
| Find a function/type by name | `find_symbol` | — |
| Understand file structure | `get_symbols_overview` | — |
| Trace all callers of a function | `find_referencing_symbols` | — |
| Read file contents | Either | `Read` |
| Exact text search | Either | `Grep` |
| Glob file patterns | Either | `glob` |
| Conceptual/semantic code search | — | `finder` |

**Rule of thumb**: Use Serena for *symbol-level* operations (find definitions, references, outlines). Use built-in tools for *text-level* operations (exact string matching, file reading, conceptual search).

## Memory System

Serena stores project knowledge in `.serena/memories/`. Use these tools for context that persists across sessions:

- **`mcp__serena__list_memories`** — See available memories
- **`mcp__serena__read_memory`** — Read a memory file (only if relevant to current task)
- **`mcp__serena__write_memory`** — Save useful project knowledge for future sessions

## Tips

- Always activate the project first — tools won't work without it
- Use `get_symbols_overview` before reading entire files to understand structure
- Chain `find_symbol` → `find_referencing_symbols` to trace code flows
- Serena understands symbol hierarchy — you can navigate from module → type → method
- For Rust projects, the language server (rust-analyzer) must be available in PATH
