# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a **skills library** — a collection of modular instruction packages (skills) that extend Claude's capabilities in specialized domains. Each skill lives in its own directory and contains a `SKILL.md` file that Claude loads when the skill is triggered.

Skills are installed and managed via the Skills CLI:
```bash
npx skills find [query]          # Search for skills
npx skills add <owner/repo@skill> # Install a skill
npx skills add <owner/repo@skill> -g -y  # Install globally, skip confirmation
npx skills check                 # Check for updates
npx skills update                # Update all installed skills
```

Browse available skills at: https://skills.sh/

## Skill Architecture

### Structure of a Skill

```
skill-name/
├── SKILL.md          # Required: YAML frontmatter + markdown instructions
├── agents/           # Subagent prompt files (grader, comparator, analyzer)
├── scripts/          # Executable helper scripts bundled with the skill
├── references/       # Reference docs loaded into context as needed
└── assets/           # Templates, icons, fonts used in output
```

### SKILL.md Frontmatter

Every skill requires a YAML frontmatter block:

```yaml
---
name: skill-name
description: |
  What this skill does and WHEN to trigger it. This is the primary triggering
  mechanism — Claude decides whether to consult a skill based solely on this field.
license: MIT  # optional
metadata:     # optional
  author: ...
  version: "1.0.0"
---
```

### Three-Level Loading System

1. **Metadata** (name + description) — Always in context (~100 words)
2. **SKILL.md body** — Loaded when skill triggers (keep under 500 lines)
3. **Bundled resources** — Read on demand via references from SKILL.md

### Triggering

Claude consults a skill based on its `description` field. Skills tend to undertrigger, so descriptions should be explicit about trigger conditions. Only complex, multi-step queries reliably trigger skills — simple one-step queries may not.

## Included Skills

The repository contains 44 skills organized by domain:

**Document Creation/Editing**
- `docx/` — Create/edit Word documents (.docx) using docx-js and XML manipulation
- `pdf/` — PDF handling
- `pptx/` — PowerPoint creation/editing via pptxgenjs
- `xlsx/` — Excel spreadsheet manipulation

**Development Workflow**
- `brainstorming/` — Collaborative design before implementation; HARD GATE: no code until design approved
- `writing-plans/` — TDD implementation plans saved to `docs/plans/YYYY-MM-DD-<feature>.md`
- `executing-plans/` — Execute plans task-by-task (for separate/parallel sessions)
- `subagent-driven-development/` — Dispatch fresh subagent per task with code review
- `test-driven-development/` — TDD patterns and anti-patterns
- `dispatching-parallel-agents/` — Parallel subagent dispatch for independent problems
- `using-git-worktrees/` — Git worktree workflow
- `finishing-a-development-branch/` — Branch completion checklist

**Code Quality**
- `code-reviewer/` — Structured code review (Security > Performance > Correctness > Maintainability); uses `rules/` directory and `AGENTS.md` compilation
- `debugger/` — Systematic debugging
- `systematic-debugging/` — Debugging methodology
- `receiving-code-review/` — How to handle review feedback
- `requesting-code-review/` — How to request reviews

**Research & Analysis**
- `deep-research/` — Multi-source research with citations
- `academic-researcher/` — Academic research
- `data-analyst/` — Data analysis
- `exploratory-data-analysis/` — EDA workflows
- `fact-checker/` — Fact verification
- `forecasting-time-series-data/` — Time series forecasting

**Content & Communication**
- `brainstorming/` — Idea exploration
- `content-creator/` — Content creation
- `technical-writer/` — Technical documentation
- `editor/` — Writing editing
- `email-drafter/` — Email composition
- `meeting-notes/` — Meeting note taking

**UI/Design**
- `frontend-design/` — Frontend design patterns
- `fullstack-developer/` — Full-stack development
- `ux-designer/` — UX design
- `visualization-expert/` — Data visualization

**Planning & Strategy**
- `decision-helper/` — Decision making frameworks
- `project-planner/` — Project planning
- `sprint-planner/` — Sprint planning
- `strategy-advisor/` — Strategic advice

**Meta/Skills**
- `find-skills/` — Discover and install skills via `npx skills`
- `skill-creator/` — Create, evaluate, and iteratively improve skills (full eval loop with viewer)
- `writing-skills/` — Writing skill instructions
- `using-superpowers/` — Overview of available skills

**Integrations**
- `workflow/` — Vercel Workflow DevKit for durable/resumable workflows
- `vercel-react-best-practices/` — React/Next.js patterns from Vercel

## Skill Creator Eval Loop

The `skill-creator/` skill has a full evaluation pipeline:

```bash
# Aggregate benchmark results
python -m scripts.aggregate_benchmark <workspace>/iteration-N --skill-name <name>

# Generate the review viewer (always do this before self-evaluating)
nohup python skill-creator/eval-viewer/generate_review.py \
  <workspace>/iteration-N \
  --skill-name "my-skill" \
  --benchmark <workspace>/iteration-N/benchmark.json \
  > /dev/null 2>&1 &

# Static output for headless/cowork environments
python skill-creator/eval-viewer/generate_review.py <workspace>/iteration-N \
  --skill-name "my-skill" --static <output_path>

# Optimize skill description triggering
python -m scripts.run_loop \
  --eval-set <path-to-trigger-eval.json> \
  --skill-path <path-to-skill> \
  --model <model-id> \
  --max-iterations 5 --verbose

# Package a skill into a .skill file
python -m scripts.package_skill <path/to/skill-folder>
```

Eval workspaces go in `<skill-name>-workspace/` as a sibling to the skill directory, organized as `iteration-1/`, `iteration-2/`, etc., with `eval-<ID>/with_skill/` and `eval-<ID>/without_skill/` (or `old_skill/`) subdirectories.

Key files in `skill-creator/`:
- `agents/grader.md` — Grading assertions against outputs
- `agents/comparator.md` — Blind A/B comparison
- `agents/analyzer.md` — Benchmark analysis patterns
- `references/schemas.md` — JSON schemas for evals.json, grading.json, benchmark.json

## docx Skill Scripts

The `docx/scripts/` directory contains Python utilities for Word document manipulation:

```bash
python scripts/office/unpack.py document.docx unpacked/      # Unpack DOCX to XML
python scripts/office/pack.py unpacked/ output.docx --original document.docx  # Repack
python scripts/office/validate.py doc.docx                   # Validate XML schema
python scripts/office/soffice.py --headless --convert-to docx doc.doc  # Convert .doc
python scripts/accept_changes.py input.docx output.docx      # Accept tracked changes
python scripts/comment.py unpacked/ 0 "Comment text"         # Add comment
```

New documents are created with JavaScript (`npm install -g docx`), then validated with `validate.py`.

## Development Workflow Pattern

The intended workflow when building features in a project that uses these skills:

1. **brainstorming** → explore intent, propose approaches, get design approval, write design doc to `docs/plans/YYYY-MM-DD-<topic>-design.md`
2. **writing-plans** → create TDD implementation plan saved to `docs/plans/YYYY-MM-DD-<feature>.md`
3. **subagent-driven-development** or **executing-plans** → implement plan task by task
4. **code-reviewer** → review with priority: Security > Performance > Correctness > Maintainability

## Creating a New Skill

1. Create `<skill-name>/SKILL.md` with YAML frontmatter (name + description required)
2. Keep SKILL.md under 500 lines; use `references/` for overflow
3. Description field is the triggering mechanism — make it explicit about when to activate
4. Use `skill-creator` skill to run the eval/iterate loop
5. Package with `python -m scripts.package_skill <path/to/skill-folder>`
