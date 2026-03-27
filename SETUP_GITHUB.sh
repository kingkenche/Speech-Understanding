#!/bin/bash

# Speech Understanding Repository Setup Script
# This script initializes git and pushes to GitHub

echo "================================================"
echo "Setting up Speech Understanding GitHub Repository"
echo "================================================"
echo ""

# Step 1: Initialize git repository
echo "Step 1: Initializing git repository..."
git init
git config user.name "Debasis"
git config user.email "your-email@example.com"  # Replace with your email

# Step 2: Add files
echo "Step 2: Adding files to git..."
git add -A
git status

# Step 3: Create initial commit
echo "Step 3: Creating initial commit..."
git commit -m "Initial commit: Speech Understanding Assignment with Q1, Q2, Q3 complete"

# Step 4: Display instructions
echo ""
echo "================================================"
echo "Next Steps:"
echo "================================================"
echo ""
echo "1. Create a new repository on GitHub:"
echo "   - Go to https://github.com/new"
echo "   - Repository name: Speech-Understanding"
echo "   - Description: Advanced Speech Processing Assignment"
echo "   - Set as PRIVATE (recommended) or PUBLIC"
echo "   - DO NOT initialize with README/LICENSE/.gitignore"
echo "   - Click 'Create repository'"
echo ""
echo "2. After creating, you'll see commands like:"
echo "   git branch -M main"
echo "   git remote add origin https://github.com/kingkenche/Speech-Understanding.git"
echo "   git push -u origin main"
echo ""
echo "3. Run these commands to push your code:"
echo "   git branch -M main"
echo "   git remote add origin https://github.com/kingkenche/Speech-Understanding.git"
echo "   git push -u origin main"
echo ""
echo "4. Enter your GitHub credentials:"
echo "   - Username: kingkenche"
echo "   - Password: Use GitHub Personal Access Token (PAT)"
echo ""
echo "5. To create a GitHub Personal Access Token:"
echo "   - Go to https://github.com/settings/tokens"
echo "   - Click 'Generate new token (classic)'"
echo "   - Select 'repo' scope"
echo "   - Copy the token and paste when prompted"
echo ""
echo "================================================"
