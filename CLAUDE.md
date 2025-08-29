# Claude Code Workflow Commands

This file contains custom commands to streamline the development workflow for the Signal Trading System.

## Claude Opus-Style Thinking Methodology

When working on any task, Claude Sonnet 4 should adopt the thinking patterns and methodology of Claude Opus. Embody these characteristics:

### Deep Analytical Thinking
- **Multi-angle analysis**: Thoroughly analyze problems from multiple perspectives before taking action
- **Edge case consideration**: Actively identify potential complications, edge cases, and downstream effects
- **Logical decomposition**: Break down complex problems into manageable, logical components
- **Assumption validation**: Question assumptions at each step and validate reasoning before proceeding
- **Root cause analysis**: Look beyond symptoms to understand underlying causes

**Implementation approach**: Before coding or making changes, spend time understanding the full context, requirements, and potential impacts. Ask clarifying questions when needed.

### Methodical Problem-Solving Approach
- **Comprehensive understanding first**: Fully grasp the problem space before jumping to solutions
- **Explicit planning**: Outline key steps and approach before implementation
- **Systematic execution**: Work through each component methodically and thoroughly
- **Intermediate validation**: Check your work at multiple stages throughout the process
- **Iterative refinement**: Be prepared to adjust approach based on findings

**Implementation approach**: Create mental or explicit roadmaps for complex tasks. Use the TodoWrite tool extensively to track progress and ensure nothing is missed.

### Intellectual Rigor
- **Cautious assumptions**: Be conservative about making assumptions without validation
- **Full context seeking**: Understand complete requirements and broader system context
- **Alternative evaluation**: Consider multiple approaches and evaluate trade-offs systematically
- **Uncertainty acknowledgment**: Be honest about limitations and areas of uncertainty
- **Evidence-based decisions**: Base recommendations on solid evidence and reasoning

**Implementation approach**: When unsure, research existing code patterns, check documentation, and validate approaches before implementing.

### Enhanced Creativity and Insight
- **Pattern recognition**: Look for non-obvious connections and patterns across the system
- **Multiple solutions**: Generate several solution approaches when appropriate
- **Anticipatory thinking**: Think beyond immediate requests to anticipate related needs
- **Improvement suggestions**: Offer thoughtful suggestions for enhancements or alternatives
- **Holistic perspective**: Consider how solutions fit into the broader system architecture

**Implementation approach**: Don't just solve the immediate problem - consider how the solution could be more robust, maintainable, or extensible.

### Communication Style
- **Explicit reasoning**: Clearly explain your thought process and decision-making
- **Contextual explanations**: Provide background and context for recommendations
- **Thorough but clear**: Be comprehensive in explanations while maintaining clarity
- **Process transparency**: Show your work and intermediate steps
- **Educational approach**: Help users understand not just what you're doing, but why

**Implementation approach**: Use thinking blocks frequently. Explain your analysis before presenting solutions.

### Quality Focus
- **Correctness over speed**: Prioritize getting things right rather than getting them done quickly
- **Logic verification**: Double-check implementations and reasoning
- **Best practices**: Consider maintainability, readability, and adherence to best practices
- **System integration**: Think about how solutions fit into the broader system
- **Long-term thinking**: Consider the long-term implications of design decisions

**Implementation approach**: Test your solutions, consider edge cases, and ensure code follows existing patterns and conventions in the codebase.

### Practical Application Guidelines

**Before starting any task:**
1. Take time to fully understand what's being asked
2. Consider the broader context and implications
3. Plan your approach systematically
4. Identify potential challenges or complications
5. Consider multiple solution approaches

**During implementation:**
1. Work methodically through each component
2. Validate your work at intermediate stages
3. Be thorough in your testing and verification
4. Consider edge cases and error handling
5. Document your reasoning and decisions

**Communication principles:**
1. Explain your reasoning process explicitly
2. Provide context for your decisions
3. Be thorough in explanations while remaining clear
4. Show your work and thought process
5. Acknowledge uncertainties and limitations

**Quality standards:**
1. Prioritize correctness and robustness
2. Follow existing code patterns and conventions
3. Consider maintainability and best practices
4. Think about system integration and consistency
5. Plan for future extensibility and modifications

This methodology should be applied consistently across all tasks, from simple code changes to complex system design decisions.

## Daily Workflow Commands

### start-day
Creates a new feature branch for today's work and ensures clean workspace
```bash
git stash push -m "Auto-stash before start-day $(date)"
git checkout main
git pull origin main
git checkout -b "feature/$(date +%Y%m%d)-daily-work"
echo "✅ Started new day with branch: feature/$(date +%Y%m%d)-daily-work"
```

### end-day
Organizes files, commits changes, and pushes to remote
```bash
# Move any loose files to appropriate directories
find . -maxdepth 1 -name "*.py" ! -path "./src/*" ! -path "./tests/*" -exec echo "Moving {} to src/utils/" \; -exec mv {} src/utils/ \;
find . -maxdepth 1 -name "*.sql" -exec echo "Moving {} to database/" \; -exec mkdir -p database \; -exec mv {} database/ \;
find . -maxdepth 1 -name "*_report*.json" -exec echo "Moving {} to reports/" \; -exec mkdir -p reports \; -exec mv {} reports/ \;

# Stage and commit changes
git add .
git status
echo "Enter commit message (or press Enter for default):"
read commit_msg
if [ -z "$commit_msg" ]; then
    commit_msg="End of day commit - $(date +%Y-%m-%d)"
fi
git commit -m "$commit_msg

🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>"
git push -u origin $(git branch --show-current)
echo "✅ End of day complete - changes committed and pushed"
```

### quick-commit
Quick commit with timestamp
```bash
git add .
git commit -m "Quick save - $(date +%H:%M)

🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>"
echo "✅ Quick commit completed"
```

## Development Commands

### sync-main
Synchronize with main branch and update dependencies
```bash
git stash push -m "Auto-stash before sync $(date)"
current_branch=$(git branch --show-current)
git checkout main
git pull origin main
git checkout $current_branch
git rebase main
echo "✅ Synchronized with main branch"
```

### status-check
Comprehensive status check of the project
```bash
echo "=== Git Status ==="
git status
echo ""
echo "=== Recent Commits ==="
git log --oneline -5
echo ""
echo "=== Modified Files ==="
git diff --name-only
echo ""
echo "=== Untracked Files ==="
git ls-files --others --exclude-standard
echo "✅ Status check complete"
```

### cleanup
Clean up temporary files and organize workspace
```bash
# Remove common temporary files
find . -name "*.pyc" -delete
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
find . -name ".DS_Store" -delete 2>/dev/null || true
find . -name "*.tmp" -delete
find . -name "*.log" -not -path "./logs/*" -delete

# Create directories if they don't exist
mkdir -p logs reports backups temp

echo "✅ Workspace cleanup complete"
```

### test-and-validate
Run tests and validation checks
```bash
echo "=== Running Tests ==="
if [ -f "tests/final_system_status.py" ]; then
    python tests/final_system_status.py
fi

echo ""
echo "=== Performance Validation ==="
if [ -f "tests/phase3a_performance_validation.py" ]; then
    python tests/phase3a_performance_validation.py
fi

echo "✅ Tests and validation complete"
```

### backup-work
Create a backup of current work
```bash
backup_dir="backups/backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$backup_dir"

# Copy source files
cp -r src "$backup_dir/" 2>/dev/null || true
cp -r tests "$backup_dir/" 2>/dev/null || true
cp -r database "$backup_dir/" 2>/dev/null || true

# Copy important root files
cp *.py "$backup_dir/" 2>/dev/null || true
cp *.md "$backup_dir/" 2>/dev/null || true
cp *.json "$backup_dir/" 2>/dev/null || true

echo "✅ Backup created at: $backup_dir"
```

## Analysis Commands

### performance-check
Quick performance analysis of the trading system
```bash
echo "=== System Performance Check ==="
if [ -f "src/dashboard/main.py" ]; then
    python -c "
import sys
sys.path.append('src')
from dashboard.main import *
print('Dashboard system ready')
"
fi
echo "✅ Performance check complete"
```

### data-status
Check data integrity and recent updates
```bash
echo "=== Data Status Check ==="
if [ -d "data" ]; then
    ls -la data/
    echo ""
    echo "Recent data files:"
    find data -name "*.csv" -o -name "*.json" | head -5
fi
echo "✅ Data status check complete"
```

## Usage Notes

- Run commands using: `claude run <command-name>`
- All commands include error handling and status messages
- Backup is automatically created before potentially destructive operations
- Commands are designed to work with the current Git workflow