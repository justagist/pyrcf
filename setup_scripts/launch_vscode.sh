#! /bin/bash
set -e

export COMPOSE_DOCKER_CLI_BUILD=1
export DOCKER_BUILDKIT=1

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )" #gets the current directory

#move to the project root folder
cd "$SCRIPT_DIR"/..

git submodule update --init --recursive

CONTAINER_NAME=${PWD##*/}  
# CONTAINER_NAME="${PWD##*/}_$(date +%Y-%m-%d_%H-%M-%S)"  

echo "stopping existing container" "$CONTAINER_NAME" 
# docker stop "$CONTAINER_NAME" || true 
docker rename "$CONTAINER_NAME" "${CONTAINER_NAME}_$(date +%Y-%m-%d_%H-%M-%S)" || true 

CONTAINER_HEX=$(printf $CONTAINER_NAME | xxd -p | tr '\n' ' ' | sed 's/\\s//g' | tr -d ' ');

#!/bin/bash

if ! dpkg -l | grep -q python3-venv; then
    echo "python3-venv is not installed. Installing..."
    sudo apt-get update
    sudo apt-get install -y python3-venv
else
    echo "python3-venv is already installed."
fi

# Define the directory for the virtual environment
VENV_DIR="venv"

# Check if the virtual environment directory exists
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment in $VENV_DIR..."
    python3 -m venv "$VENV_DIR"
    echo "Activating the virtual environment..."
    source $VENV_DIR/bin/activate
    echo "Installing deps rocker..."
    pip install deps-rocker
    echo "Virtual environment setup and deps rocker installation complete."
else
    echo "Virtual environment already exists in $VENV_DIR."
    echo "Activating the existing virtual environment..."
    source $VENV_DIR/bin/activate
fi

# Run the rocker command with the specified parameters
rocker --nvidia --x11 --user --pull --git --image-name "$CONTAINER_NAME" --name "$CONTAINER_NAME" --volume "${PWD}":/workspaces/"${CONTAINER_NAME}":Z --deps --oyr-run-arg " --detach" ubuntu:22.04 "$@" 

deactivate

#this follows the same convention as if it were opened by a vscode devcontainer
code --folder-uri vscode-remote://attached-container+"$CONTAINER_HEX"/workspaces/"${CONTAINER_NAME}"
