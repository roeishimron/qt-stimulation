# GEMINI

This project is meant to be run with the following command:

```
docker build . -t experiment-runner && docker run -it --mount type=bind,source=/tmp/.X11-unix,target=/tmp/.X11-unix  --mount type=bind,source=/dev/dri/renderD128,destination=/dev/dri/renderD128  --mount type=bind,source=/dev/snd,destination=/dev/snd --mount type=bind,source=/run/user/$UID/wayland-0,destination=/run/user/1000/wayland-0 --privileged --mount type=bind,source=$PWD/output,target=/app/output experiment-runner
```

## Requirements

*   There must be a local `output` folder for the container to bind to.
*   The code will run an experiment based on qt over wayland.

## Usage

While the default entrypoint is `env_main.py`, it is also possible to run `main.py` by overriding the default at `docker run...`.
