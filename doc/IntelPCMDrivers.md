OS support for Intel® Performance Counter Monitor
=================================================

In order to use the Intel PCM tool, we need to enable access to the OS kernel to the MSR registers.

To simplify the process of accessing the MSR registers, we provide you with compiled versions of the
drivers that access the MSR register.

Linux
--------
Linux has a built-in support for accessing the MSRs. If not enabled, load the MSR module:

```sh
modprobe msr
```

This driver reads the MSR data, and as such needs elevated privileges. If `NMI Watchdog` is enabled on your system,
you might have to disable it with:

```sh
echo 0 > /proc/sys/kernel/nmi_watchdog
```

Note that you can run the PCM in usermode if `/dev/cpu/*/msr` are read-write enabled to non-root users. This however is
not a good practices, as it leaves the system vulnerable to many [side channel attacks][sda]. 

[sda]: https://en.wikipedia.org/wiki/Side-channel_attack

Mac OS X (Yosemite / El Capitan / Sierra)
--------

1. Disable [System Integrity Protection][SystemIntegrityProtection] on Mac OS X.
2. To load the Intel PCM driver use:

```sh
cd drivers/mac/
sudo su
./load.sh
```

3. To unload the Intel PCM driver use:

```sh
cd drivers/mac/
sudo su
./unload.sh
```

[SystemIntegrityProtection]: SystemIntegrityProtection.md

Windows (7 / 10)
-------

Windows is supported using third-party libraries & binaries:
 
- `winpmem` [Rekall Memory Forensic Framework][rekall] (Release 1.6.0 Gotthard)
- `winring` [Real Temp 3.70][realtemp]
- `elevate` [Command-Line UAC Elevation Utility][elevate]

[realtemp]: https://www.techpowerup.com/realtemp/
[rekall]: http://www.rekall-forensic.com
[elevate]: http://code.kliu.org/misc/elevate/
 

 
The only limitation in Windows is that the tool must be run with elevated privileges (`Run as administrator`).
