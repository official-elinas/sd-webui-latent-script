# sd-webui-latent-script

This is a script to allow generating latents on your CPU (Seeds) which generally result in improved consistency. This can be seen from ComfyUI.

It also has an option to disable the mean calculation when using (emphasis:1.1) in the prompts which makes it so your prompts have equal weight, whether their first or last.

This is a simple script that uses "monkey patching" to replace functions with other functions at runtime, without modifying the underlying code itself. In this case we are replacing `cuda` calls with `cpu` calls when generating latents. This does a similar method with disabling the mean calculation in prompts.