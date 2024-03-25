# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
import pyhocon

def write_neus_config_file(output_dir: Path, use_masks: bool=False):
    """Writes a config file with the right input and output paths for NeuS.
    Args:
        output_dir: The directory with the 'cameras_sphere.npz' file and the 'image' and 'mask' folders.
        use_masks: If True write a config that enables the masks
    """
    if use_masks:
        conf = pyhocon.ConfigFactory.parse_file(Path(__file__).parent/'neus_wmask.conf')
    else:
        conf = pyhocon.ConfigFactory.parse_file(Path(__file__).parent/'neus_womask.conf')

    # the input directory for NeuS
    conf.put('dataset.data_dir', str(output_dir.resolve()))
    # the output directory for NeuS
    if use_masks:
        neus_output_dir = output_dir/'exp'/'wmask_sphere'
    else:
        neus_output_dir = output_dir/'exp'/'womask_sphere'
    neus_output_dir.mkdir(exist_ok=True, parents=True)
    conf.put('general.base_exp_dir', str(neus_output_dir.resolve()))

    converter = pyhocon.converter.HOCONConverter()
    if use_masks:
        config_path = output_dir/'wmask.conf'
    else:
        config_path = output_dir/'womask.conf'
    with open(config_path,'w') as f:
        f.write(converter.to_hocon(conf))
