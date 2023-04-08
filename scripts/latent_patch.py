import torch
import modules.scripts as scripts
import gradio as gr

from modules import devices
from modules.scripts import AlwaysVisible


class Script(scripts.Script):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.default_randn = devices.randn
        self.default_randn_without_seed = devices.randn_without_seed

    def title(self):
        return "Latent Script"

    def show(self, is_img2img):
        return AlwaysVisible

    def ui(self, is_img2img):
        is_cpu_seed = gr.Checkbox(False, label="Generate Latent (Seed) on CPU for more consistent results")
        # is_mean_disabled = gr.Checkbox(False, label="Disable mean calculation to make weights identical")
        return [is_cpu_seed]

    def process(self, p, is_cpu_seed):

        if is_cpu_seed:
            print("Using CPU seed")

            def bypass_randn(seed, shape):
                torch.manual_seed(seed)
                return torch.randn(shape, device=torch.device('cpu')).to(torch.device('cpu'))

            def bypass_randn_without_seed(shape):
                return torch.randn(shape, device=torch.device('cpu')).to(torch.device('cpu'))

            devices.randn = bypass_randn
            devices.randn_without_seed = bypass_randn_without_seed

        else:
            # restore
            devices.randn = self.default_randn
            devices.randn_without_seed = self.default_randn_without_seed
