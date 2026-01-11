#!/bin/bash

# Git Installation Script for Linux Systems
# Supports Amazon Linux, Ubuntu, CentOS, RHEL, and Debian

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}[HEADER]${NC} $1"
}

# Check if running as root
if [[ $EUID -ne 0 ]]; then
   print_error "This script must be run as root (use sudo)"
   exit 1
fi

print_header "Git Installation Script Starting..."

# Detect the operating system
detect_os() {
    if [[ -f /etc/os-release ]]; then
        . /etc/os-release
        OS=$NAME
        VERSION=$VERSION_ID
    elif [[ -f /etc/redhat-release ]]; then
        OS=$(cat /etc/redhat-release)
    elif [[ -f /etc/debian_version ]]; then
        OS="Debian"
        VERSION=$(cat /etc/debian_version)
    else
        OS=$(uname -s)
        VERSION=$(uname -r)
    fi
    
    print_status "Detected OS: $OS $VERSION"
}

# Install Git based on the operating system
install_git() {
    print_status "Installing Git..."
    
    if command -v dnf &> /dev/null; then
        # Amazon Linux 2023, Fedora, newer CentOS/RHEL
        print_status "Using dnf package manager..."
        dnf update -y
        dnf install -y git git-lfs
        
    elif command -v yum &> /dev/null; then
        # Amazon Linux 2, CentOS, RHEL
        print_status "Using yum package manager..."
        yum update -y
        yum install -y git
        
        # Try to install git-lfs if available
        yum install -y git-lfs || print_warning "git-lfs not available in repositories"
        
    elif command -v apt-get &> /dev/null; then
        # Ubuntu, Debian
        print_status "Using apt package manager..."
        apt-get update
        apt-get install -y git git-lfs
        
    elif command -v zypper &> /dev/null; then
        # openSUSE
        print_status "Using zypper package manager..."
        zypper refresh
        zypper install -y git git-lfs
        
    elif command -v pacman &> /dev/null; then
        # Arch Linux
        print_status "Using pacman package manager..."
        pacman -Syu --noconfirm git git-lfs
        
    else
        print_error "Unsupported package manager. Please install Git manually."
        exit 1
    fi
}

# Configure Git with global settings
configure_git() {
    print_status "Configuring Git with recommended global settings..."
    
    # Set up global Git configuration
    git config --system init.defaultBranch main
    git config --system pull.rebase false
    git config --system core.autocrlf input
    git config --system core.safecrlf true
    git config --system color.ui auto
    git config --system push.default simple
    
    # Create a global gitignore file
    cat > /etc/gitignore_global << 'EOF'
# Compiled source
*.com
*.class
*.dll
*.exe
*.o
*.so

# Packages
*.7z
*.dmg
*.gz
*.iso
*.jar
*.rar
*.tar
*.zip

# Logs and databases
*.log
*.sql
*.sqlite

# OS generated files
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# IDE files
.vscode/
.idea/
*.swp
*.swo
*~

# Node modules
node_modules/

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
.env

# Temporary files
*.tmp
*.temp
EOF

    # Set the global gitignore
    git config --system core.excludesfile /etc/gitignore_global
    
    print_status "Global Git configuration completed"
}

# Install additional Git tools
install_git_extras() {
    print_status "Installing additional Git tools..."
    
    # Install git-extras if available
    if command -v dnf &> /dev/null; then
        dnf install -y git-extras || print_warning "git-extras not available"
    elif command -v yum &> /dev/null; then
        # Try EPEL repository for git-extras
        yum install -y epel-release || true
        yum install -y git-extras || print_warning "git-extras not available"
    elif command -v apt-get &> /dev/null; then
        apt-get install -y git-extras || print_warning "git-extras not available"
    fi
    
    # Install tig (text-mode interface for Git) if available
    if command -v dnf &> /dev/null; then
        dnf install -y tig || print_warning "tig not available"
    elif command -v yum &> /dev/null; then
        yum install -y tig || print_warning "tig not available"
    elif command -v apt-get &> /dev/null; then
        apt-get install -y tig || print_warning "tig not available"
    fi
}

# Create helpful Git aliases
create_git_aliases() {
    print_status "Creating useful Git aliases..."
    
    # System-wide Git aliases
    git config --system alias.st status
    git config --system alias.co checkout
    git config --system alias.br branch
    git config --system alias.ci commit
    git config --system alias.unstage 'reset HEAD --'
    git config --system alias.last 'log -1 HEAD'
    git config --system alias.visual '!gitk'
    git config --system alias.lg "log --color --graph --pretty=format:'%Cred%h%Creset -%C(yellow)%d%Creset %s %Cgreen(%cr) %C(bold blue)<%an>%Creset' --abbrev-commit"
    git config --system alias.ll "log --oneline --graph --decorate --all"
    git config --system alias.undo "reset --soft HEAD^"
    git config --system alias.amend "commit --amend --no-edit"
    
    print_status "Git aliases created successfully"
}

# Verify installation
verify_installation() {
    print_status "Verifying Git installation..."
    
    if command -v git &> /dev/null; then
        GIT_VERSION=$(git --version)
        print_status "✓ Git installed successfully: $GIT_VERSION"
        
        # Check if git-lfs is available
        if command -v git-lfs &> /dev/null; then
            GIT_LFS_VERSION=$(git lfs version)
            print_status "✓ Git LFS available: $GIT_LFS_VERSION"
        else
            print_warning "Git LFS not installed"
        fi
        
        # Show Git configuration
        print_status "Current Git configuration:"
        git config --system --list | head -10 || true
        
    else
        print_error "Git installation failed"
        exit 1
    fi
}

# Create usage instructions
create_usage_instructions() {
    print_status "Creating usage instructions..."
    
    cat > /opt/git_usage.txt << 'EOF'
Git Usage Instructions
======================

Initial Setup (for each user):
1. Set your name: git config --global user.name "Your Name"
2. Set your email: git config --global user.email "your.email@example.com"

Basic Commands:
- git init                    # Initialize a new repository
- git clone <url>            # Clone a repository
- git add <file>             # Stage changes
- git commit -m "message"    # Commit changes
- git push                   # Push to remote repository
- git pull                   # Pull from remote repository
- git status                 # Check repository status
- git log                    # View commit history

Useful Aliases (already configured):
- git st                     # git status
- git co                     # git checkout
- git br                     # git branch
- git ci                     # git commit
- git lg                     # Pretty log with graph
- git ll                     # One-line log with graph
- git undo                   # Undo last commit (soft reset)
- git amend                  # Amend last commit

For more information: https://git-scm.com/doc
EOF

    print_status "Usage instructions saved to /opt/git_usage.txt"
}

# Main execution
main() {
    detect_os
    install_git
    configure_git
    install_git_extras
    create_git_aliases
    verify_installation
    create_usage_instructions
    
    print_header "Git installation completed successfully!"
    print_status "Usage instructions: /opt/git_usage.txt"
    print_status "Users should run the following to set up their Git identity:"
    echo -e "  ${BLUE}git config --global user.name \"Your Name\"${NC}"
    echo -e "  ${BLUE}git config --global user.email \"your.email@example.com\"${NC}"
}

# Run the main function
main "$@"
