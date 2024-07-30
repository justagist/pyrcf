#!/bin/bash

mv python_template "$1"
mv python_template.deps.yaml "$1".deps.yaml

find . \( -type d -name .git -prune \) -o \( -type f -not -name 'tasks.json' -not -name 'update_from_template.sh' -not -name 'update_from_template_ours.sh' \) -print0 | xargs -0 sed -i "s/python_template/$1/g"
