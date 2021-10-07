import torch.optim as optim

def config_optimizer(cfg, param):
    print(f"Using [{cfg['optimizer']['name']}]: learning rate = {cfg['optimizer']['lr']}, weight_decay = {cfg['optimizer']['weight_decay']}")
    
    if cfg['optimizer']['name'] is 'sgd':
        optimizer = optim.SGD(param, lr=cfg['optimizer']['lr'], momentum=cfg['optimizer']['momentum'], weight_decay=cfg['optimizer']['weight_decay'])
    elif cfg['optimizer']['name'] is 'adam':
        optimizer = optim.Adam(param, lr=cfg['optimizer']['lr'], betas=cfg['optimizer']['beta'], eps=cfg['optimizer']['eps'], weight_decay=cfg['optimizer']['weight_decay'])
    else:
        AssertionError('optimizer can not be recognized.')
    
    return optimizer
