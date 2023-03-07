from os.path import dirname, basename, isfile, join
import glob
from torch_geometric.graphgym.config import cfg

modules = glob.glob(join(dirname(__file__), "*.py"))
__all__ = [
    basename(f)[:-3] for f in modules
    if isfile(f) and not f.endswith('__init__.py')
]

cfg.dataset.protein_filename = 'protein.csv'
cfg.dataset.interaction_filename = 'interaction.csv'
cfg.model.loss_pos_weight = 1.0
