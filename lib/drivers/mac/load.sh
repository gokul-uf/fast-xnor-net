#!/usr/bin/env bash

echo "================================================================================================"
echo "Kext signature"
echo "================================================================================================"
codesign -dvvv PcmMsrDriver.kext
echo "================================================================================================"
echo ""

cp -R PcmMsrDriver.kext /System/Library/Extensions/
chown -R root:wheel /System/Library/Extensions/PcmMsrDriver.kext
kextload /System/Library/Extensions/PcmMsrDriver.kext