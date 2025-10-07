DewarpingPlayer README
====================

DewarpingPlayer is an application for playing video streams over RTSP and dewarping them in a novel way with OpenGL pixel shaders.

## Screenshots

![Undistort](undistort.png)
![Fisheye](fisheye.png)

## Build

```
git clone git@github.com:MonocleSecurity/DewarpingPlayer.git --recursive
cd DewarpingPlayer
VCPKG_FORCE_SYSTEM_BINARIES=1 cmake -G"Unix Makefiles" .
make
```

## Run

./DewarpingPlayer rtsp://username:password@192.168.0.1/stream
