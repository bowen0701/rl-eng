#!/bin/bash
# agent_init.sh
# Initializes project-level config structure for Gemini CLI and Claude Code.
# Run from the project root directory:
#   chmod +x agent_init.sh
#   ./agent_init.sh

set -e  # Exit immediately if any command fails

echo "==> Initializing agent config structure..."

# ── Gemini CLI ────────────────────────────────────────────────────────────────

# Create the project-level Gemini CLI skills directory
mkdir -p .gemini/skills

# Create a minimal project-level settings.json for Gemini CLI if it doesn't exist
if [ ! -f .gemini/settings.json ]; then
  cat > .gemini/settings.json << 'EOF'
{
  "sandbox": false
}
EOF
  echo "    Created .gemini/settings.json"
else
  echo "    Skipped .gemini/settings.json (already exists)"
fi

# ── Cross-platform (Gemini CLI + Claude Code) ─────────────────────────────────

# Create the universal .agents/skills/ directory.
# Both Gemini CLI and Claude Code discover skills here,
# avoiding duplication across .gemini/skills/ and .claude/skills/.
mkdir -p .agents/skills

# ── Claude Code ───────────────────────────────────────────────────────────────

# Create the project-level Claude Code skills directory
mkdir -p .claude/skills

# ── GEMINI.md ─────────────────────────────────────────────────────────────────

# Create GEMINI.md only if it doesn't exist.
# Prefer running `gemini /init` inside Gemini CLI to auto-generate content.
if [ ! -f GEMINI.md ]; then
  cat > GEMINI.md << 'EOF'
# Project Context

## Overview
<!-- Describe what this project is and its primary purpose -->

## Architecture Constraints
<!-- Define boundaries and rules AI must respect, not descriptions of what exists -->

## Behavior Rules
<!-- What the AI should and should not do in this codebase -->

## Known Pitfalls
<!-- Patterns or files where AI tends to make mistakes in this specific codebase -->
EOF
  echo "    Created GEMINI.md (placeholder — run /init in Gemini CLI to populate)"
else
  echo "    Skipped GEMINI.md (already exists)"
fi

# ── Summary ───────────────────────────────────────────────────────────────────

echo ""
echo "==> Done. Directory structure:"
echo ""

# Use tree if available, otherwise fall back to find
if command -v tree &> /dev/null; then
  tree -a -I '.git' --dirsfirst .gemini .claude .agents 2>/dev/null
else
  find .gemini .claude .agents -not -path '*/.git/*' | sort
fi

echo ""
echo "==> Next steps:"
echo "    1. Run Gemini CLI and execute /init to populate GEMINI.md"
echo "    2. Edit .gemini/settings.json for project-specific Gemini CLI config"
echo "    3. Add skills under .agents/skills/<skill-name>/SKILL.md"
echo "       (discoverable by both Gemini CLI and Claude Code)"
