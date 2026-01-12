#!/bin/bash

# Create System-Wide Conda Environment from environment.yml Script

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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

# Configuration
CONDA_PATH="/opt/anaconda3"  # or /opt/miniconda3
YAML_FILE="$1"
ENV_NAME=""

# Show usage
show_usage() {
    echo "Usage: $0 <environment.yml> [options]"
    echo ""
    echo "Options:"
    echo "  -h, --help          Show this help message"
    echo "  -p, --prefix PATH   Conda installation path (default: /opt/anaconda3)"
    echo "  -n, --name NAME     Override environment name from YAML file"
    echo ""
    echo "Examples:"
    echo "  $0 environment.yml"
    echo "  $0 datascience.yml --name shared-datascience"
    echo "  $0 ml-env.yml --prefix /opt/miniconda3"
    echo ""
    echo "The environment.yml file should contain:"
    echo "  name: environment_name"
    echo "  channels:"
    echo "    - conda-forge"
    echo "    - defaults"
    echo "  dependencies:"
    echo "    - python=3.11"
    echo "    - numpy"
    echo "    - pandas"
    echo "    - pip"
    echo "    - pip:"
    echo "      - some-pip-package"
}

# Parse command line arguments (FIXED VERSION)
POSITIONAL_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_usage
            exit 0
            ;;
        -p|--prefix)
            CONDA_PATH="$2"
            shift 2
            ;;
        -n|--name)
            ENV_NAME="$2"
            shift 2
            ;;
        -*)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
        *)
            POSITIONAL_ARGS+=("$1")
            shift
            ;;
    esac
done

# Set positional arguments
set -- "${POSITIONAL_ARGS[@]}"

# Now assign the YAML file
YAML_FILE="$1"

# Check if we have too many positional arguments
if [[ ${#POSITIONAL_ARGS[@]} -gt 1 ]]; then
    print_error "Too many arguments. Expected only one YAML file."
    print_error "Received: ${POSITIONAL_ARGS[*]}"
    show_usage
    exit 1
fi

# Check arguments
if [[ -z "$YAML_FILE" ]]; then
    print_error "Environment YAML file is required"
    show_usage
    exit 1
fi

# Check if running as root
if [[ $EUID -ne 0 ]]; then
   print_error "This script must be run as root (use sudo)"
   exit 1
fi

# Validate YAML file
validate_yaml_file() {
    print_status "Validating environment YAML file..."
    
    if [[ ! -f "$YAML_FILE" ]]; then
        print_error "YAML file not found: $YAML_FILE"
        exit 1
    fi
    
    # Extract environment name from YAML if not provided
    if [[ -z "$ENV_NAME" ]]; then
        ENV_NAME=$(grep "^name:" "$YAML_FILE" | sed 's/name:[[:space:]]*//' | tr -d '"' | tr -d "'")
        if [[ -z "$ENV_NAME" ]]; then
            print_error "No environment name found in YAML file and none provided with --name"
            print_error "Please add 'name: your_env_name' to the YAML file or use --name option"
            exit 1
        fi
    fi
    
    print_status "Environment name: $ENV_NAME"
    print_status "YAML file validated successfully"
}

# Check if conda is installed
check_conda() {
    if [[ ! -f "$CONDA_PATH/bin/conda" ]]; then
        print_error "Conda not found at $CONDA_PATH"
        print_error "Please install Anaconda/Miniconda first or specify correct path with --prefix"
        exit 1
    fi
    
    CONDA_VERSION=$("$CONDA_PATH/bin/conda" --version)
    print_status "Using $CONDA_VERSION at $CONDA_PATH"
}

# Create environment from YAML
create_environment() {
    print_header "Creating conda environment '$ENV_NAME' from $YAML_FILE"
    
    # Check if environment already exists
    if "$CONDA_PATH/bin/conda" env list | grep -q "^$ENV_NAME "; then
        print_warning "Environment '$ENV_NAME' already exists"
        read -p "Do you want to remove it and recreate? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            print_status "Removing existing environment..."
            "$CONDA_PATH/bin/conda" env remove -n "$ENV_NAME" -y
        else
            print_error "Operation cancelled"
            exit 1
        fi
    fi
    
    # Create environment from YAML file
    print_status "Creating environment from YAML file..."
    
    # If custom name provided, create temporary YAML with new name
    if [[ -n "$2" ]]; then  # Custom name was provided
        TEMP_YAML="/tmp/temp_env_$(date +%s).yml"
        sed "s/^name:.*/name: $ENV_NAME/" "$YAML_FILE" > "$TEMP_YAML"
        YAML_TO_USE="$TEMP_YAML"
    else
        YAML_TO_USE="$YAML_FILE"
    fi
    
    # Create the environment
    "$CONDA_PATH/bin/conda" env create -f "$YAML_TO_USE"
    
    # Clean up temporary file if created
    if [[ -n "$TEMP_YAML" && -f "$TEMP_YAML" ]]; then
        rm -f "$TEMP_YAML"
    fi
    
    if [[ $? -ne 0 ]]; then
        print_error "Failed to create environment from YAML file"
        exit 1
    fi
    
    print_status "Environment created successfully"
}

# Set proper permissions
set_permissions() {
    print_status "Setting permissions for all users..."
    
    ENV_PATH="$CONDA_PATH/envs/$ENV_NAME"
    
    if [[ ! -d "$ENV_PATH" ]]; then
        print_error "Environment directory not found: $ENV_PATH"
        exit 1
    fi
    
    # Set ownership and permissions
    chown -R root:root "$ENV_PATH"
    chmod -R 755 "$ENV_PATH"
    
    # Make sure all users can read and execute
    find "$ENV_PATH" -type d -exec chmod 755 {} \;
    find "$ENV_PATH" -type f -exec chmod 644 {} \;
    find "$ENV_PATH/bin" -type f -exec chmod 755 {} \; 2>/dev/null || true
    
    print_status "Permissions set successfully"
}

# Create activation script for all users
create_activation_script() {
    print_status "Creating activation script..."
    
    cat > "/etc/profile.d/conda-${ENV_NAME}.sh" << EOF
# Conda environment activation script for $ENV_NAME
alias activate-$ENV_NAME='source $CONDA_PATH/bin/activate $ENV_NAME'
alias deactivate-$ENV_NAME='conda deactivate'

# Function to activate environment
activate_${ENV_NAME//-/_}() {
    source $CONDA_PATH/bin/activate $ENV_NAME
}

# Export environment path for easy access
export CONDA_ENV_${ENV_NAME^^}="$CONDA_PATH/envs/$ENV_NAME"
EOF

    chmod +x "/etc/profile.d/conda-${ENV_NAME}.sh"
    
    print_status "Activation script created at /etc/profile.d/conda-${ENV_NAME}.sh"
}

# Create environment update script
create_update_script() {
    print_status "Creating environment update script..."
    
    cat > "/opt/update_conda_env_${ENV_NAME}.sh" << EOF
#!/bin/bash
# Script to update conda environment: $ENV_NAME

set -e

CONDA_PATH="$CONDA_PATH"
ENV_NAME="$ENV_NAME"
YAML_FILE="$YAML_FILE"

echo "Updating conda environment: \$ENV_NAME"

# Check if YAML file exists
if [[ -f "\$YAML_FILE" ]]; then
    echo "Updating from YAML file: \$YAML_FILE"
    "\$CONDA_PATH/bin/conda" env update -n "\$ENV_NAME" -f "\$YAML_FILE"
else
    echo "YAML file not found: \$YAML_FILE"
    echo "Updating all packages in environment..."
    "\$CONDA_PATH/bin/conda" update -n "\$ENV_NAME" --all -y
fi

# Reset permissions
chown -R root:root "\$CONDA_PATH/envs/\$ENV_NAME"
chmod -R 755 "\$CONDA_PATH/envs/\$ENV_NAME"
find "\$CONDA_PATH/envs/\$ENV_NAME" -type d -exec chmod 755 {} \\;
find "\$CONDA_PATH/envs/\$ENV_NAME" -type f -exec chmod 644 {} \\;
find "\$CONDA_PATH/envs/\$ENV_NAME/bin" -type f -exec chmod 755 {} \\; 2>/dev/null || true

echo "Environment updated successfully!"
EOF

    chmod +x "/opt/update_conda_env_${ENV_NAME}.sh"
    print_status "Update script created at /opt/update_conda_env_${ENV_NAME}.sh"
}

# Create usage instructions
create_usage_instructions() {
    print_status "Creating usage instructions..."
    
    cat > "/opt/conda_env_${ENV_NAME}_usage.txt" << EOF
Conda Environment Usage: $ENV_NAME
==================================

Created from: $YAML_FILE
Environment Location: $CONDA_PATH/envs/$ENV_NAME

Activation Methods:
1. Command line: conda activate $ENV_NAME
2. Alias: activate-$ENV_NAME
3. Function: activate_${ENV_NAME//-/_}

Deactivation:
- conda deactivate
- deactivate-$ENV_NAME

Environment Variables:
- CONDA_ENV_${ENV_NAME^^}=$CONDA_PATH/envs/$ENV_NAME

Installed Packages:
$("$CONDA_PATH/bin/conda" list -n "$ENV_NAME" 2>/dev/null | head -20 || echo "Could not list packages")

Management Commands:
- Update environment: sudo /opt/update_conda_env_${ENV_NAME}.sh
- List environments: conda env list
- Remove environment: sudo conda env remove -n $ENV_NAME

To install additional packages:
- conda install -n $ENV_NAME package_name
- $CONDA_PATH/envs/$ENV_NAME/bin/pip install package_name

To export current environment:
- conda env export -n $ENV_NAME > new_environment.yml

Original YAML file content:
$(cat "$YAML_FILE" 2>/dev/null || echo "YAML file not accessible")
EOF

    print_status "Usage instructions saved to /opt/conda_env_${ENV_NAME}_usage.txt"
}

# Verify installation
verify_installation() {
    print_status "Verifying environment installation..."
    
    # Check if environment exists
    if "$CONDA_PATH/bin/conda" env list | grep -q "^$ENV_NAME "; then
        print_status "✓ Environment '$ENV_NAME' created successfully"
        
        # Show environment info
        print_status "Environment location:"
        "$CONDA_PATH/bin/conda" env list | grep "^$ENV_NAME "
        
        # Show installed packages count
        PACKAGE_COUNT=$("$CONDA_PATH/bin/conda" list -n "$ENV_NAME" | wc -l)
        print_status "Installed packages: $PACKAGE_COUNT"
        
        # Test activation
        print_status "Testing environment activation..."
        source "$CONDA_PATH/bin/activate" "$ENV_NAME"
        PYTHON_VERSION=$(python --version 2>&1)
        conda deactivate
        print_status "✓ Python version in environment: $PYTHON_VERSION"
        
    else
        print_error "Environment verification failed"
        exit 1
    fi
}

# Main execution
main() {
    validate_yaml_file
    check_conda
    create_environment
    set_permissions
    create_activation_script
    create_update_script
    create_usage_instructions
    verify_installation
    
    print_header "System-wide conda environment '$ENV_NAME' created successfully!"
    print_status "Created from: $YAML_FILE"
    print_status "All users can activate with: conda activate $ENV_NAME"
    print_status "Or use alias: activate-$ENV_NAME"
    print_status "Usage instructions: /opt/conda_env_${ENV_NAME}_usage.txt"
    print_status "Update script: /opt/update_conda_env_${ENV_NAME}.sh"
    
    print_warning "Users may need to reload their shell or run:"
    print_warning "source /etc/profile.d/conda-${ENV_NAME}.sh"
}

# Run main function
main "$@"
