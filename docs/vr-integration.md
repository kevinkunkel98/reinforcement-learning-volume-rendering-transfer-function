# VR headset integration — findings and next steps

Investigation into hooking a VR headset up to this project (viewing the
volume-rendered output in a headset) via [Vrui](https://web.cs.ucdavis.edu/~okreylos/ResDev/Vrui/),
Oliver Kreylos' VR toolkit. Written up so the next attempt (likely on the
Linux machine mentioned below) doesn't have to re-derive any of this.

## Hardware surveyed

Four headsets were available to test against:

| Headset | Tracking | Verdict |
|---|---|---|
| **HTC Vive Pro** | Lighthouse (base stations) | Not viable on macOS. Requires SteamVR/OpenVR, which Valve dropped Mac support for in 2020. The open-source alternative (`libsurvive` for tracking + `Monado` for display/compositing) is Linux/Windows-only per its own docs. Vrui's own "full native HTC Vive support" (2016) was built and tested on Kreylos' Linux lab hardware — no evidence of a macOS port. |
| **Oculus Development Kit (DK1/DK2)** | DK1: orientation only (IMU). DK2: adds positional tracking via external IR camera. | **The one viable candidate on macOS.** Vrui has its own driver (`VRDeviceDaemon/VRDevices/OculusRift.cpp`) that talks to the hardware directly via `libusb`, not the proprietary Oculus SDK/runtime. libusb is genuinely cross-platform (Linux/macOS/Windows), so this driver is not Linux-locked — confirmed by building it (see below). DK2's positional-tracking camera driver (`Video/Linux/OculusRiftDK2VideoDevice.cpp`) *is* Linux-only, so on macOS you'd get DK1-equivalent (orientation-only) tracking even with a DK2. |
| **Sony HMZ-series** ("Personal 3D Viewer") | None | Never a tracked VR device — just a stereoscopic display panel, no head tracking hardware at all. Out of scope for VR use regardless of OS. |
| **HP Windows Mixed Reality** | Inside-out (proprietary) | Not viable anywhere but Windows. Locked to the proprietary Windows Mixed Reality Portal; unlike Vive there's no open-source reimplementation of its tracking, so there's no fallback path at all (worse than the Vive situation). |

**Bottom line:** on this MacBook (Apple M1 Pro, Apple Silicon, macOS 26.6.2), only the Oculus DK1 has a real path forward.

## Vrui builds on macOS — build log

Built Vrui 6.0 from source (`github.com/Doc-Ok/Vrui`) on the M1 Pro Mac to
test feasibility. **It fully builds**, including `VRDeviceDaemon` and the
`libOculusRift.bundle` driver, after 8 fixes — all mechanical, none requiring
a rewrite. In rough order hit:

1. **libusb not detected.** The build's `SYSTEM_PACKAGE_SEARCH_PATHS` defaults
   to `/usr/local /usr` (plus `/opt/local` for MacPorts), missing Homebrew's
   Apple Silicon prefix. Fix: build with
   `SYSTEM_PACKAGE_SEARCH_PATHS="/usr/local /usr /opt/local /opt/homebrew"`.

2. **`VERSION` file collides with libc++'s `<version>` header.** macOS's
   default filesystem (APFS) is case-insensitive, so the repo's root
   `VERSION` file aliases with the C++20 standard header `<version>`, which
   several files transitively include. Broke every translation unit. Fix:
   renamed `VERSION` → `VERSION.txt` (nothing in the build reads it — version
   numbers are hardcoded in the makefile).

3. **`pipe2()` is Linux-only.** `Threads/EventDispatcher.cpp` used it
   unconditionally; macOS/BSD has no equivalent. Fix: `#ifdef __linux__` with
   a `pipe()` + `fcntl(F_SETFL, O_NONBLOCK)` fallback for other platforms.

4. **Stray `.template` disambiguator.** `Misc/VarIntMarshaller.h:106` had
   `sink.template write(seq,numBytes);` with no actual template argument list
   following — legal-but-meaningless on old compilers, a hard error on
   current Clang. Fix: removed the `.template`.

5. **macOS's dylib linker is stricter than Linux's `.so` linker.** Several
   Vrui libraries have circular-ish cross-dependencies (e.g. `libThreads`
   needs `libMisc`'s `MessageLogger`) that Linux's lazier `.so` symbol
   resolution tolerates by default; macOS's `dynamiclib` linking requires
   each library to resolve its own symbols eagerly, so these failed to link
   at all. Fix: added `-undefined dynamic_lookup` to Darwin's `DSOLINKFLAGS`
   in `BuildRoot/SystemDefinitions` (mirrors a flag Vrui's own
   `PLUGINLINKFLAGS` already used for the same reason).

6. **`GLhandleARB` type mismatch — the big one.** Apple's own OpenGL headers
   define `GLhandleARB` as `void*`; this ~2020 codebase (like virtually every
   Linux/Windows OpenGL codebase) treats it as `GLuint`. Touched **66 files**
   throughout the GL rendering support library — looked structural at first.
   Turned out XQuartz's own `glext.h` has a `BUILDING_MESA` macro that
   switches `GLhandleARB` to an integer type on Apple when defined. Fix: one
   line, `EXTRACPPSYSFLAGS += -DBUILDING_MESA` added to the Darwin block in
   `BuildRoot/SystemDefinitions` — fixed all 66 files at once, no per-file
   patching needed.

7. **`AudioFileOpenURL` enum strictness.** `Sound/SoundPlayer.cpp:96` passed
   a raw `0x1` where modern `AudioToolbox.framework` headers require an
   `AudioFilePermissions`-typed value (implicit int→enum no longer allowed).
   Fix: wrapped it, `AudioFilePermissions(0x1)`.

8. **Broken compiler-version detection — root cause of a confusing,
   seemingly random string of `.d: No such file or directory` build
   failures.** `BuildRoot/SystemDefinitions` picks the dependency-file naming
   convention with `expr $(gcc -dumpversion) ">=" "3.0.0"`, using **string**
   (lexicographic) comparison. Modern Clang reports version `21.0.0`;
   comparing `"21.0.0"` against `"3.0.0"` character-by-character, `'2' <
   '3'`, so the check decided the compiler was older than gcc 3.0 and picked
   the wrong (1990s-GCC, current-working-directory) dependency-file
   convention. This only broke plain executable targets (library objects use
   a separate, accidentally-no-op dependency macro, which is why hundreds of
   library files built fine before this surfaced on the first real
   executables). Fix: force the modern convention unconditionally —
   `DEPFILETEMPLATE = '$(patsubst %.o,%.d,$@)'` — right after the broken
   conditional block.

With all 8 fixes: `make` succeeds (exit 0) for the full Vrui tree, and
`ExamplePrograms` builds too (needed two more command-line variable
overrides — `VRUI_MAKEDIR`, `VRUI_LIBDIR`, `VRUI_PACKAGEROOT`, and
`EXTRACINCLUDEFLAGS` — to point at the local, uninstalled build tree instead
of an installed `/usr/local` prefix).

**Running an example app (`ShowEarthModel`) segfaults**, most likely because
Vrui couldn't find its runtime config (normally deployed by `sudo make
install`, which needs interactive `sudo` — not something automatable here).
Not evidence of a deeper platform problem given how far everything else got;
just the next thing to chase.

This was all done in a disposable scratchpad clone, not against this
project's actual dependencies — none of these patches live in this repo.

## Confirmed hardware blocker (tested with the physical DK1)

The DK1 breakout box was physically connected (DVI-via-HDMI-adapter + USB-A +
its own power brick). macOS does detect the tracker at the USB level:

```
Tracker DK   idVendor=0x2833 (Oculus VR)   idProduct=0x0001
```

(via `ioreg -c IOUSBHostDevice -r -l`, searching for the device name/vendor
ID rather than `system_profiler SPUSBDataType`, which returned nothing
useful in this environment).

But a standalone `libusb` probe (open the device, then try to claim its
interface — same two calls Vrui's `OculusRift.cpp` driver makes) shows
exactly why the driver won't work here:

```
RESULT: libusb_open succeeded
Vendor=0x2833 Product=0x0001 NumConfigurations=1
kernel_driver_active(interface 0) = 1
RESULT: libusb_claim_interface FAILED: LIBUSB_ERROR_ACCESS (code -3)
```

macOS's own HID subsystem (`AppleUserUSBHostHIDDevice` / `IOHIDInterface`)
has already exclusively claimed the device as a generic HID input device by
the time `libusb` tries to open it. On Linux, `libusb_detach_kernel_driver()`
would let a userspace program forcibly take the interface back from
whatever kernel driver has it — that call is a documented no-op on macOS's
IOKit-backed libusb implementation, so there is no equivalent escape hatch.
This is **not a configuration problem** — `sudo make install` and the
`VRDeviceDaemon` config wiring below would not change this result.

**This is confirmed, not speculative** (the "untested unknown" from the
original writeup): Vrui's existing Oculus driver, as written, cannot read
the DK1's tracker data on macOS. The only ways around it:

- Write a new Vrui device driver against macOS's native `IOHIDManager` API
  (read the sensor as a standard HID input device through the OS, instead of
  trying to bypass the OS via raw USB) — genuine new development work, not a
  patch to the existing driver.
- Ship a kernel extension to stop macOS's generic HID driver from matching
  this vendor/product ID first, freeing it for `libusb` — requires SIP
  disabled / reduced security mode on Apple Silicon; heavy and fragile for a
  side project.

Neither is a quick fix, so this line of attack stops here on macOS.

## What's left to actually get the DK1 rendering something (on Linux)

The steps below are what remained even *before* the HID-claim test above
killed the macOS path outright. They should still be the right steps on
Linux, where this whole USB-vs-HID conflict is much less likely to bite
(`libusb_detach_kernel_driver()` is fully supported there):

1. `sudo make install` — deploys libraries/headers to `/usr/local` and
   generates the real `Vrui.cfg` from a template.
2. Wire up `Share/OculusRift.cfg` (ships with Vrui) into the
   `VRDeviceDaemon` config so it actually loads the Oculus driver at
   startup.
3. Run `VRDeviceDaemon` as a background process — Vrui apps talk to it over
   a local socket for tracking data, they don't touch the headset directly.
4. Point an app's environment config at the DK1's panel resolution/geometry
   (also templated in `Share/OculusRift.cfg`) so its HDMI output is treated
   as the render target.

## Recommendation

Do this on the Linux machine, not macOS — the DK1 is now a confirmed dead
end on this Mac (not just theoretically difficult), for the exact HID-claim
reason above. None of the eight build fixes earlier in this doc are Linux
problems either — Vrui's own tooling and Kreylos' own Vive/Rift driver work
were done and tested on Linux in the first place. If a Linux attempt is
made, this doc's headset survey table still applies — the Vive Pro becomes
viable too on Linux (either via SteamVR-on-Linux or Vrui's native driver),
which it never was on macOS at all.
