# GitHub Repository Setup Guide

## Step 1: Create a New Repository on GitHub

1. Go to https://github.com/new
2. Fill in the repository details:
   - **Repository name**: `Speech-Understanding`
   - **Description**: `Advanced Speech Processing Assignment - Q1, Q2, Q3`
   - **Privacy**: Select **PRIVATE** (unless you want it public)
   - **DO NOT** check "Initialize this repository with a README" (we already have one)
   - **DO NOT** add .gitignore or License (we already have .gitignore)

3. Click **"Create repository"**

## Step 2: Copy the Push Commands

After creating the repository, GitHub will show you commands like:
```
git branch -M main
git remote add origin https://github.com/kingkenche/Speech-Understanding.git
git push -u origin main
```

Copy these commands.

## Step 3: Push Your Code to GitHub

Run these commands in your terminal:

```bash
cd /home/debasis/Shivam/M25CSA028_Assignment-1

# Rename master branch to main (recommended for new repos)
git branch -M main

# Add remote repository
git remote add origin https://github.com/kingkenche/Speech-Understanding.git

# Push code to GitHub
git push -u origin main
```

## Step 4: Handle GitHub Authentication

When prompted for credentials, you have two options:

### Option A: Personal Access Token (Recommended)
1. Go to https://github.com/settings/tokens
2. Click **"Generate new token (classic)"**
3. In the "Select scopes" section:
   - Check **`repo`** (for full control of private repositories)
   - Optionally check **`workflow`** if needed
4. Click **"Generate token"**
5. Copy the token (it will only be shown once)
6. When git prompts for password, paste the token

### Option B: GitHub CLI (easiest)
1. Install GitHub CLI:
   ```bash
   sudo apt-get install gh  # Ubuntu/Debian
   # or brew install gh    # macOS
   ```
2. Authenticate:
   ```bash
   gh auth login
   ```
3. Follow the prompts and select HTTPS
4. Then run the git push command again

### Option C: SSH (if you have SSH keys configured)
If you have SSH keys set up on GitHub, use:
```bash
git remote add origin git@github.com:kingkenche/Speech-Understanding.git
git push -u origin main
```

## Step 5: Verify Push Success

After pushing, you should see:
```
Branch 'main' set up to track remote branch 'main' from 'origin'.
```

Then verify on GitHub:
1. Go to https://github.com/kingkenche/Speech-Understanding
2. You should see all the files: Q1/, Q2/, Q3/, README.md, requirements.txt, etc.

## Full Command Summary

```bash
cd /home/debasis/Shivam/M25CSA028_Assignment-1
git branch -M main
git remote add origin https://github.com/kingkenche/Speech-Understanding.git
git push -u origin main
```

## If You Already Created the Repo

If the remote add fails because it already exists, first remove it:
```bash
git remote remove origin
git remote add origin https://github.com/kingkenche/Speech-Understanding.git
git push -u origin main
```

## Repository Contents

Your GitHub repository will contain:

```
Speech-Understanding/
├── README.md                 # Main documentation
├── requirements.txt          # All dependencies
├── .gitignore               # Git configuration
├── SETUP_GITHUB.sh         # This setup script
│
├── Q1/                      # Multi-Stage Cepstral Feature Extraction
│   ├── mfcc_manual.py
│   ├── leakage_snr.py
│   ├── voiced_unvoiced.py
│   ├── phonetic_mapping.py
│   ├── q1_report.pdf
│   ├── README.md
│   ├── requirements.txt
│   └── results/            # Analysis outputs
│
├── Q2/                      # Paper Implementation
│   ├── train.py
│   ├── eval.py
│   ├── models.py
│   ├── review.pdf
│   ├── q2_readme.md
│   ├── configs/            # Training configurations
│   ├── results/            # Results and plots
│   └── requirements.txt
│
└── Q3/                      # Ethical Auditing
    ├── audit.py
    ├── privacymodule.py
    ├── pp_demo.py
    ├── train_fair.py
    ├── q3_report.pdf
    ├── audit_plots.pdf
    ├── evaluation_scripts/  # FAD/DNSMOS evaluation
    ├── examples/           # Audio samples
    └── requirements.txt
```

## Troubleshooting

### "Repository already exists"
- Your repo might already exist on GitHub
- Delete it from GitHub settings and create a new one
- Or change the repository name

### "Permission denied (publickey)"
- SSH key not configured
- Use HTTPS with Personal Access Token instead

### "The remote repository is empty"
- Make sure you pushed to the correct URL
- Verify with: `git remote -v`

### Large files warning
- Large dataset files are excluded by .gitignore
- Reports and results are included
- If you need to include large datasets, use Git LFS

## Next Steps

1. Create the GitHub repository
2. Run the push commands above
3. Share your repository link with your instructor
4. Create a direct link to your ZIP file if needed:
   - Go to: https://github.com/kingkenche/Speech-Understanding
   - Click **Code** → **Download ZIP**

---

**Questions?** Check git status:
```bash
git status
git remote -v
git log --oneline
```
