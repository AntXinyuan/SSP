from mmcv.runner import Hook, HOOKS
from mmcv.parallel import is_module_wrapper

@HOOKS.register_module()
class RecordEpochIterHook(Hook):

    def before_run(self, runner):
        model = runner.model
        if is_module_wrapper(model):
            model = model.module
        model.bbox_head.runner_info['max_epochs'] = runner.max_epochs
        model.bbox_head.runner_info['max_iters'] = runner.max_iters

    def before_epoch(self, runner):
        model = runner.model
        if is_module_wrapper(model):
            model = model.module
        model.bbox_head.runner_info['epoch'] = runner.epoch
        model.bbox_head.runner_info['dataset'] = (runner.data_loader.dataset.__class__.__name__[:-7], 
                                                  runner.data_loader.dataset.CLASSES, 
                                                  runner.data_loader.dataset.PALETTE)

    def before_iter(self, runner):
        model = runner.model
        if is_module_wrapper(model):
            model = model.module
        #model.head.runner_info['epoch'] = runner.epoch
        model.bbox_head.runner_info['iter'] = runner.iter
        model.bbox_head.runner_info['inner_iter'] = runner.inner_iter

@HOOKS.register_module()
class FreezeLayersHook(Hook):
        def __init__(self, start_epoch=None, freeze_names=None):
            self.start_epoch = start_epoch
            self.freeze_names = freeze_names

        def before_train_epoch(self, runner):
            start_epoch = self.start_epoch if self.start_epoch is not None else runner.max_epochs // 2
            model = runner.model
            if is_module_wrapper(model):
                model = model.module
            if runner.epoch > start_epoch:
                for name, param in runner.model.named_parameters():
                    if self.freeze_names is not None:
                        for fz_name in self.freeze_names:
                            if name.find(fz_name) != -1:
                                runner.logger.info(f'freeze {name}')
                                param.requires_grad = False
                    else:
                        runner.logger.info(f'unfreeze {name}')