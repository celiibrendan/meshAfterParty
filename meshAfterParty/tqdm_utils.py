"""
To provide a tqdm that can be controlled

Reference Article: https://github.com/tqdm/tqdm/issues/619
"""
#global_enable = False

from tqdm.notebook import tqdm
class _TQDM(tqdm):
    disable = False
    def __init__(self, *argv, **kwargs):
        kwargs['disable'] = self.disable
        if kwargs.get('disable_override', 'def') != 'def':
            kwargs['disable'] = kwargs['disable_override']
        super().__init__(*argv, **kwargs)
tqdm = _TQDM

def practice_tqdm(n=10000):
    for i in tqdm(range(n)):
        i + 5
        
def turn_on_tqdm():
    tqdm.disable=False
    
def turn_off_tqdm():
    tqdm.disable=True
    
    
