#!/bin/bash

# Wait result.js
echo "Wait for result.json"
until [ -f ~/awsim-logs/result.json ]
do
  sleep 5
done

cp ~/awsim-logs/result.json /aichallenge/output/result.json
cp ~/awsim-logs/verbose_result.json /aichallenge/output/verbose_result.json
cat /aichallenge/output/result.json 
cat /aichallenge/output/verbose_result.json 
rm ~/awsim-logs/*.json
echo "Copy Finish!"
