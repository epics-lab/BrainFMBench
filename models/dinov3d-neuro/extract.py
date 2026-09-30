
import requests
import tempfile
from dinov3.inference.load import load_teacher_model_for_inference

CONFIG_REMOTE = "https://huggingface.co/huggingbrain/Dinov3d-Neuro/resolve/main/config.yaml"
CKPT = "https://huggingface.co/huggingbrain/Dinov3d-Neuro/resolve/main/eval/training_122999/teacher_checkpoint.pth"


def _download_config(remote_path):
    """
    takes a remote path, downloads and returns a local temp path instead
    """
    with requests.get(remote_path, stream=True) as response:
        response.raise_for_status()

    temp_config_file = tempfile.NamedTemporaryFile(suffix=".yaml",
                                                   delete=False).name
    with open(temp_config_file, "wb") as file:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:  # Filter out keep-alive chunks
                file.write(chunk)

    return temp_config_file


def extract(input_dir, output_csv, weights_dir, config_file=None):

    if config_file is None:
        config_file = _download_config(CONFIG_REMOTE)

    model, embed_dim = load_teacher_model_for_inference(
        config_file,
        ckpt,
        device='cpu')

    model.eval()


    return None
