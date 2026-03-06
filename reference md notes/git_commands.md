# Git Commands Reference Guide

This document provides a comprehensive list of common and useful Git commands categorized by their typical use cases.

## Configuration and Setup

Set up your Git installation, defining your identity and preferences.

*   **`git config --global user.name "[name]"`**: Sets the name you want attached to your commit transactions.
*   **`git config --global user.email "[email address]"`**: Sets the email you want attached to your commit transactions.
*   **`git config --global color.ui auto`**: Enables helpful colorization of command line output.
*   **`git config --list`**: Lists all currently configured Git settings.

## Getting and Creating Projects

Start a new repository or obtain an existing one.

*   **`git init`**: Initializes a new local Git repository in the current directory.
*   **`git clone [url]`**: Downloads a project and its entire version history from a remote repository.

## Basic Snapshotting

Work with files, staging changes, and committing them to history.

*   **`git status`**: Lists all new or modified files to be committed.
*   **`git add [file]`**: Snapshots the file in preparation for versioning (stages the file).
*   **`git add .`** or **`git add -A`**: Stages all modified, new, and deleted files in the current directory and subdirectories.
*   **`git commit -m "[descriptive message]"`**: Records file snapshots permanently in version history.
*   **`git commit -am "[message]"`**: Stages all modified (not new) files and commits them in one step.
*   **`git diff`**: Shows file differences not yet staged.
*   **`git diff --staged`** or **`git diff --cached`**: Shows file differences between staging and the last file version.
*   **`git restore [file]`**: Discards unstaged changes in the working directory (newer alternative to `git checkout -- [file]`).
*   **`git restore --staged [file]`**: Unstages a file, retaining the changes in the working directory (newer alternative to `git reset HEAD [file]`).
*   **`git rm [file]`**: Deletes the file from the working directory and stages the deletion.

## Branching and Merging

Isolate work in branches, integrate changes, and manage context.

*   **`git branch`**: Lists all local branches in the current repository. The current branch has an asterisk `*`.
*   **`git branch -a`**: Lists all local and remote branches.
*   **`git branch [branch-name]`**: Creates a new branch.
*   **`git branch -d [branch-name]`**: Deletes the specified branch (only if merged).
*   **`git branch -D [branch-name]`**: Forcefully deletes the specified branch.
*   **`git checkout [branch-name]`** or **`git switch [branch-name]`**: Switches to the specified branch and updates the working directory.
*   **`git checkout -b [branch-name]`** or **`git switch -c [branch-name]`**: Creates a new branch and switches to it in one step.
*   **`git merge [branch-name]`**: Combines the specified branch's history into the current branch.
*   **`git stash`**: Temporarily stores all modified tracked files.
*   **`git stash pop`**: Restores the most recently stashed files and removes them from the stash list.
*   **`git stash list`**: Lists all stashed changesets.

## Sharing and Updating Projects

Interact with remote repositories to collaborate.

*   **`git remote -v`**: Lists all currently configured remote repositories and their URLs.
*   **`git remote add [alias] [url]`**: Adds a new remote repository with a given alias (usually `origin`).
*   **`git fetch [alias]`**: Downloads all history from the remote tracking branches.
*   **`git merge [alias]/[branch]`**: Combines remote-tracking branch into current local branch.
*   **`git pull`**: Fetches and merges changes on the remote server to your working directory (fetch + merge).
*   **`git push [alias] [branch]`**: Uploads all local branch commits to the remote repository.
*   **`git push -u origin [branch]`**: Pushes a new branch and sets up tracking so future `git pull` and `git push` commands run without arguments.

## Inspection and Comparison

Examine history, commits, and changes.

*   **`git log`**: Lists version history for the current branch.
*   **`git log --oneline`**: Shows history concisely (one line per commit).
*   **`git log --graph --oneline --all`**: Displays a visual graph of branches and commits.
*   **`git log --follow [file]`**: Lists version history for a file, including renames.
*   **`git show [commit]`**: Outputs metadata and content changes of the specified commit.

## Advanced and Rewriting History

Correct mistakes and maintain a clean history. *Use with caution!*

*   **`git commit --amend`**: Replaces the last commit with a new one incorporating current staged changes or a new message.
*   **`git rebase [branch]`**: Reapplies commits on top of another base tip (rewrites history, do not use on shared public branches).
*   **`git reset [commit]`**: Undoes all commits after `[commit]`, preserving changes locally.
*   **`git reset --hard [commit]`**: Discards all history and changes back to the specified commit. *Warning: Permanent data loss.*
*   **`git revert [commit]`**: Creates a new commit that undoes the changes made in the specified commit (safe for public history).
*   **`git cherry-pick [commit]`**: Applies the changes introduced by an existing commit to the current branch.
