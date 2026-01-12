#!/bin/bash

# Anaconda Installation Script for All Users
# This script downloads and installs Anaconda system-wide

set -e  # Exit on any error

# Configuration
ANACONDA_VERSION="2025.12-1"  # Latest version as of 2024
ANACONDA_INSTALLER="Anaconda3-${ANACONDA_VERSION}-Linux-x86_64.sh"
ANACONDA_URL="https://repo.anaconda.com/archive/${ANACONDA_INSTALLER}"
INSTALL_DIR="/opt/anaconda3"
TEMP_DIR="/tmp"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
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

# Check if running as root
if [[ $EUID -ne 0 ]]; then
   print_error "This script must be run as root (use sudo)"
   exit 1
fi

print_status "Starting Anaconda installation for all users..."

# Update system packages (FIXED VERSION)
print_status "Updating system packages..."
if command -v yum &> /dev/null; then
    # Amazon Linux / CentOS / RHEL
    yum update -y
    # Fix curl conflict by removing curl-minimal first
    yum remove -y curl-minimal || true
    yum install -y wget bzip2
    # Install curl separately if needed
    yum install -y curl || true
elif command -v apt-get &> /dev/null; then
    # Ubuntu / Debian
    apt-get update
    apt-get install -y wget curl bzip2
else
    print_error "Unsupported package manager. Please install wget and bzip2 manually."
    exit 1
fi

# Download Anaconda installer
print_status "Downloading Anaconda installer..."
cd $TEMP_DIR
if [[ -f "$ANACONDA_INSTALLER" ]]; then
    print_warning "Installer already exists, skipping download"
else
    wget -O "$ANACONDA_INSTALLER" "$ANACONDA_URL"
    if [[ $? -ne 0 ]]; then
        print_error "Failed to download Anaconda installer"
        exit 1
    fi
fi

# Verify download
print_status "Verifying installer..."
if [[ ! -f "$ANACONDA_INSTALLER" ]]; then
    print_error "Anaconda installer not found"
    exit 1
fi

# Make installer executable
chmod +x "$ANACONDA_INSTALLER"

# Remove existing installation if it exists
if [[ -d "$INSTALL_DIR" ]]; then
    print_warning "Existing Anaconda installation found at $INSTALL_DIR"
    read -p "Do you want to remove it and reinstall? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_status "Removing existing installation..."
        rm -rf "$INSTALL_DIR"
    else
        print_error "Installation cancelled"
        exit 1
    fi
fi

# Install Anaconda
print_status "Installing Anaconda to $INSTALL_DIR..."
bash "$ANACONDA_INSTALLER" -b -p "$INSTALL_DIR"

if [[ $? -ne 0 ]]; then
    print_error "Anaconda installation failed"
    exit 1
fi

# Set proper permissions
print_status "Setting permissions..."
chown -R root:root "$INSTALL_DIR"
chmod -R 755 "$INSTALL_DIR"

# Configure conda solver to use libmamba (faster dependency resolution)
print_status "Configuring conda to use libmamba solver..."
"$INSTALL_DIR/bin/conda" config --system --set solver libmamba

# Configure conda to avoid TOS issues by using conda-forge
print_status "Configuring conda channels..."
"$INSTALL_DIR/bin/conda" config --system --add channels conda-forge
"$INSTALL_DIR/bin/conda" config --system --set channel_priority strict

# Try to accept TOS if the command exists, otherwise skip
print_status "Handling Anaconda Terms of Service..."
if "$INSTALL_DIR/bin/conda" tos --help &>/dev/null; then
    print_status "Accepting Terms of Service..."
    "$INSTALL_DIR/bin/conda" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
    "$INSTALL_DIR/bin/conda" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true
else
    print_warning "TOS command not available, configuring to use conda-forge only..."
    "$INSTALL_DIR/bin/conda" config --system --remove channels defaults || true
fi



# Create system-wide conda configuration
print_status "Configuring Anaconda for all users..."

# Add conda to system PATH
cat > /etc/profile.d/conda.sh << 'EOF'
# Anaconda3 configuration
export PATH="/opt/anaconda3/bin:$PATH"

# Initialize conda for bash and zsh
if [ -f "/opt/anaconda3/etc/profile.d/conda.sh" ]; then
    . "/opt/anaconda3/etc/profile.d/conda.sh"
fi
EOF

# Make the profile script executable
chmod +x /etc/profile.d/conda.sh

# Initialize conda for all users
print_status "Initializing conda for all shells..."
"$INSTALL_DIR/bin/conda" init bash
"$INSTALL_DIR/bin/conda" init zsh

# Update conda and install common packages
print_status "Updating conda and installing common packages..."
"$INSTALL_DIR/bin/conda" update -y conda
"$INSTALL_DIR/bin/conda" update -y --all

# Install additional useful packages
print_status "Installing additional packages..."
"$INSTALL_DIR/bin/conda" install -y -c conda-forge \
    jupyter

# Configure Jupyter for system-wide access
print_status "Configuring Jupyter..."
"$INSTALL_DIR/bin/jupyter" notebook --generate-config --allow-root


# Set up conda environments directory
mkdir -p /opt/conda-envs
chown -R root:root /opt/conda-envs
chmod 755 /opt/conda-envs

# Clean up
print_status "Cleaning up..."
rm -f "$TEMP_DIR/$ANACONDA_INSTALLER"

print_status "Installation completed successfully!"
print_status "Anaconda is installed at: $INSTALL_DIR"
print_status "Usage instructions: /opt/anaconda3/USAGE.txt"
print_warning "Please reload your shell or run: source /etc/profile.d/conda.sh"

# Display version information
print_status "Installed versions:"
"$INSTALL_DIR/bin/conda" --version
"$INSTALL_DIR/bin/python" --version

print_status "Installation complete! All users can now use Anaconda."
