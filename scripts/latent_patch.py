import torch
import gradio as gr
import modules.scripts as scripts
from modules import devices
from modules.sd_hijack_clip import FrozenCLIPEmbedderWithCustomWordsBase
from modules.scripts import AlwaysVisible


class Script(scripts.Script):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.default_randn = devices.randn
        self.default_randn_without_seed = devices.randn_without_seed
        self.default_sd_hijack_clip_process_tokens = FrozenCLIPEmbedderWithCustomWordsBase.process_tokens

    def title(self):
        return "Latent Modifier"

    def show(self, is_img2img):
        return AlwaysVisible

    def ui(self, is_img2img):
        is_cpu_seed = gr.Checkbox(False, label="Generate Latent (Seed) on CPU for more consistent results")
        is_mean_disabled = gr.Checkbox(False, label="Disable mean calculation to make weights identical")
        return [is_cpu_seed, is_mean_disabled]

    def process(self, p, is_cpu_seed, is_mean_disabled):

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

        if is_mean_disabled:
            def bypass_process_tokens(self, remade_batch_tokens, batch_multipliers):
                tokens = torch.asarray(remade_batch_tokens).to(devices.device)
                if self.id_end != self.id_pad:
                    for batch_pos in range(len(remade_batch_tokens)):
                        index = remade_batch_tokens[batch_pos].index(self.id_end)
                        tokens[batch_pos, index + 1:tokens.shape[1]] = self.id_pad
                z = self.encode_with_transformers(tokens)
                batch_multipliers = torch.asarray(batch_multipliers).to(devices.device)
                z = z * batch_multipliers.reshape(batch_multipliers.shape + (1,)).expand(z.shape)
                return z

            FrozenCLIPEmbedderWithCustomWordsBase.process_tokens = bypass_process_tokens
        else:
            FrozenCLIPEmbedderWithCustomWordsBase.process_tokens = self.default_sd_hijack_clip_process_tokens
