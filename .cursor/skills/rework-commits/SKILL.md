# Split branch into reviewable commits

Rework a branch into a sequence of small, semantic commits for review.

## Important

- Prepend `GIT_EDITOR=true` to all git commands you run, especially ones looking at diffs, so you avoid getting blocked

## Instructions

1. **Check for uncommitted changes**: Abort if there are any.
2. **Check rebase status**: Verify the branch is rebased on top of `main`. Abort if not.
3. **Save recovery point**: Tell the user the current commit hash in case we need to `git reset --hard` to it later.
4. **Save the original diff**: Save the full git diff to `/tmp/original-diff.patch` before making changes.
5. **Reset to main**: Run `git reset main` to unstage all changes.
6. **Plan the commits**: Read through ALL changes carefully. Plan a logical breakdown into small, sequential, semantic commits. Write a TODO for each in `/tmp/split-todos.md`. Order: database/schema changes first, backend second, frontend last.
7. **Create the commits**: Work through the TODOs one by one. Write excellent commit descriptions for human reviewers.
8. **Validate**: Compare the current diff against `/tmp/original-diff.patch` to ensure no changes were lost or altered.
9. **Cleanup**: Delete temporary files once validation passes.

## Notes

- If validation fails, tell the user and provide the original commit hash for recovery
- Each commit should be self-contained and represent a logical unit of work
- Commit messages should explain the "why" behind the changes
