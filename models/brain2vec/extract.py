import os
import csv
import torch
from torch.utils.data import DataLoader
from Brain2Vec.dataloader.loader import NiftiLoader

from Brain2Vec.models.brain2vec import brain2vec_base_patch8_dec512d8b
from Brain2Vec.models.utils import load_brain2vec


def test_dataloader(in_dir: str) -> DataLoader:
    single_loader = NiftiLoader(
        mode="inference",
        in_dir=in_dir,
        augmentation=False,
    )
    dataloader = DataLoader(
        single_loader,
        batch_size=1,
        shuffle=False,
        num_workers=os.cpu_count() // 2,  # type: ignore
    )

    return dataloader

@torch.no_grad()  # type: ignore
def extract(in_dir: str, output_csv: str) -> None:
    """
    Please install Brain2Vec from source:
        git clone https://github.com/CBICA/Brain2Vec
        cd Brain2Vec
        pip install -e .

    We suggest to use a different virtualenv for preprocessing and Brain2Vec to
    minimize dependency errors/warnings.
    """
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    model = load_brain2vec(
        repo_id="cbica/Brain2Vec",
        weights_filename="brain2vec.safetensors",
        model_class=brain2vec_base_patch8_dec512d8b,
        model_kwargs={"img_size": 128},
        device="cuda",
    )

    dataloader = test_dataloader(in_dir)

    model.eval()

    out_features = {}
    for _, (imgs, mrids) in enumerate(dataloader):
        imgs = imgs.to("cuda").float()

        with torch.autocast(device_type="cuda"):
            features, _, _ = model.forward_encoder(imgs, mask_ratio=0.0)
        features = features[:, 0, :].float()
        for i in range(len(mrids)):
            out_features[mrids[i]] = features[i].cpu().tolist()

    dim = len(next(iter(out_features.values())))
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["subject_id"] + [f"f{i}" for i in range(dim)])
        for subject_id, vec in out_features.items():
            writer.writerow([subject_id] + vec)
