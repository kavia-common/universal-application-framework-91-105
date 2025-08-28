#!/bin/bash
cd /home/kavia/workspace/code-generation/universal-application-framework-91-105/frontend_app
npm run build
EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
   exit 1
fi

