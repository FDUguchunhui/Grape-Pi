from torch_geometric.graphgym.register import register_config


@register_config('protein')
def set_cfg_protein(cfg):
    r'''
    This function sets the default config value for customized options
    :return: customized configuration use by the experiment.
    '''

    cfg.dataset.protein_filename = 'protein.csv'
    cfg.dataset.interaction_filename = 'interaction.csv'
