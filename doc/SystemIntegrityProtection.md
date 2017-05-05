Disable System Integrity Protection on Mac OS X
=============================================== 

To use Intel PCM on a Mac OS X machine, a `kext` module has to be loaded. To load a module on Mac OS X
operating system, starting from Yosemite and above, a Developer ID Certificate for Signing Kexts is 
required. Unfortunately, even though we have repeatedly requested `kext` signing license from Apple, we have 
not yet obtained one. We have also asked Intel developers to [provide us with a precompiled driver][opcmissue] 
for Mac OS X, but as of the moment of designing the performance skeleton, such module has not been 
provided.

We advise you against disabling System Integrity Protection on long term, since this will render your
machine vulnerable to system attacks. For example, trojan like [Backdoor.MAC.Eleanor][eleanor] thrive
once arbitrary kernel modules can be loaded. Thus make sure that you enable the protection as soon as
you are done using IntelPCM.

Mac OS X (Yosemite)
---------------------

Enable loading unsigned `kexts` (Yosemite):

1. Run `sudo nvram boot-args="kext-dev-mode=1"`
2. Restart the machine.

Disable loading unsigned `kexts` (Yosemite):

1. Run `sudo nvram -d boot-args"`
2. Restart the machine.


Mac OS X (El Capitan / Sierra)
---------------------

As of OS X El Capitan, the kext-dev-mode boot-arg is now obsolete. The only way to work around this issue is
to disable [System Integrity Protection][sipmac] of Mac OS X.
 
**Disable System Integrity Protection:** 
1. Click the  menu.
2. Select Restart...
3. Hold down command-R to boot into the Recovery System.
4. Click the Utilities menu and select Terminal.
5. Type `csrutil disable` and press return.
6. Close the Terminal app.
7. Click the  menu and select Restart.

```shell
Successfully disabled System Integrity Protection. Please restart the machine 
for the changes to take effect.
```

**Enable System Integrity Protection:**
1. Click the  menu.
2. Select Restart...
3. Hold down command-R to boot into the Recovery System.
4. Click the Utilities menu and select Terminal.
5. Type `csrutil enable` and press return.
6. Close the Terminal app.
7. Click the  menu and select Restart.

Troubleshooting:
---------------

Depending on the installation and configuration of your Mac OS X machine, it might be the case that 
when executing `csrutil disable` you get:

```sh
-bash-3.2# csrutil disable
-bash: csrutil: command not found
```

This is because your Mac has an older recovery image installed, or is downloading an older image over
the network. To avoid this issue, we need an updated recovery image.

Installing a recovery image (El Capitan / Sierra):

1. Back up your Mac machine (better be safe than sorry). 
2. Use Disk Utility and create a new partition (you can partition the existing drive). 
3. Recovery partitions are in general small partitions, about ~650 MB.
4. Get a copy of the installer file for your version of OS X.   
5. Download and run [Recovery Partition Creator 4.x][recovery].
6. Follow the instructions as the app proceeds. In-dept tutorial is available [here][tutorialrecovery].
7. Restart the machine and hold down Option (⌥) button. 
8. Choose the recovery partition.
9. Click the Utilities menu and select Terminal.
10. Type `csrutil enable` and press return.
11. Close the Terminal app.
12. Click the  menu and select Restart.

Notes:
- This method will not work on OS X Yosemite since `csrutil` is not available on this OS X version.
- If you need older images of OS X installers, but have not way to download then use the [polybox share][osxinstallers]. 

[opcmissue]: https://github.com/opcm/pcm/issues/4
[sipmac]: https://support.apple.com/en-us/HT204899
[eleanor]: https://labs.bitdefender.com/2016/07/new-mac-backdoor-nukes-os-x-systems/
[recovery]: http://musings.silvertooth.us/downloads-2/
[tutorialrecovery]: http://www.macworld.co.uk/how-to/mac/how-create-mac-recovery-partition-os-x-el-capitan-yosemite-backup-free-3636717/
[osxinstallers]: https://polybox.ethz.ch/index.php/s/HHD5Kh1C3BWN6u8

