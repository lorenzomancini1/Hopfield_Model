#!/bin/bash

REMOTE=europa  # The name of the remote machine in your .ssh/config file
REMOTE_DIR="~/Git/Hopfield_Model/data/raw"

rsync -avz $REMOTE:$REMOTE_DIR ./

REMOTE=azure
REMOTE_DIR="~/Hopfield_Model/data/raw"

rsync -avz $REMOTE:$REMOTE_DIR ./
