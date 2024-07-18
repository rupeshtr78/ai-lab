#!/bin/bash

ollama show codellama:latest --modelfile > codellama2.modelfile

ollama create rtr-codellama-v01 -f ./codellama2.modelfile
