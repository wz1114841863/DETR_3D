from mmcv.parallel import DataContainer as DC
from mmdet.datasets.pipelines.formatting import to_tensor
from mmdet.datasets.pipelines.formatting import DefaultFormatBundle \
    as MMDetDefaultFormatBundle
    
from ..builder import PIPELINES


@PIPELINES.register_module()
class FormatBundle(MMDetDefaultFormatBundle):
    """在默认的format bundle中添加一些额外操作
    

    Args:
        MMDetDefaultFormatBundle (_type_): _description_
    """
    def __init__(self, 
                    extra_keys=[],
                    img_to_float=True, 
                    pad_val=dict(img=0, masks=0, seg=255)):
        super().__init__(img_to_float, pad_val)
        self.extra_keys = extra_keys
        
    def __call__(self, results):
        """

        Args:
            results (_type_): _description_
        """
        results = super(FormatBundle, self).__call__(results)
        assert isinstance(self.extra_keys, (list, tuple))
        if self.extra_keys:
            for key in self.extra_keys:
                if key not in results:
                    continue
                results[key] = DC(to_tensor(results[key]))
        return results