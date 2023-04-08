# sd-webui-latent-script

This is a script to allow generating latents on your CPU (Seeds) which generally result in improved consistency. This can be seen from ComfyUI.

An additional feature (to be added soon) is to disable the mean calculation which unbalances your weights in prompts, favoring the former prompts higher, which also leads to improved consistency.

This is a simple script that uses "monkey patching" to replace functions with other functions. In this case we are replacing `cuda` calls with `cpu` calls when generating latents. We will also be replacing the mean calculation in prompt weights soon.