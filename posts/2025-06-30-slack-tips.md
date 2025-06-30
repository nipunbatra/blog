---
author: Nipun Batra
badges: true
categories:
- productivity
- tools
date: '2025-06-30'
title: Slack Tips for Better Team Management
toc: true
---

Here are some useful Slack tips to improve your team communication and channel management.

## Adding Members from Source to Destination Channel

When you need to move or copy members from one channel to another, here's a quick way to do it:

1. **In the source channel**: Use `/who` command to get a list of all members
2. **Copy the member list** from the output
3. **In the destination channel**: Use `/invite` followed by pasting the member list

This saves you from manually typing each username and ensures you don't miss anyone when setting up new channels or reorganizing your workspace.

Example:
```
# In source channel
/who

# Copy the usernames from the output, then in destination channel
/invite @user1 @user2 @user3 @user4
```

This tip is especially useful when:
- Creating project-specific channels from larger teams
- Setting up temporary channels for events or sprints
- Reorganizing team structures
- Onboarding new team members to multiple relevant channels

*More Slack tips coming soon...*