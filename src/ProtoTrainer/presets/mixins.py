import torch
import json

from os import PathLike

from src.ProtoTrainer.core.generic.base_mixin import BaseMixin
from src.ProtoTrainer.core.context import Context, Priority
from pathlib import Path
from rich import print


# TODO: load model with best evaluation metric
class SaveAndLoadMixin(BaseMixin):

    class _Cfg:
        @classmethod
        def get_members(cls) -> list[str]:
            return [getattr(cls, obj_str) for obj_str in  dir(cls) if not obj_str.startswith("__") and not obj_str.startswith("_")
                    and isinstance(getattr(cls, obj_str), str)]

    class Keys(_Cfg):
        model_state_dict = "model"
        compiled_model_state_dict = "compiled_model"
        optimizer_state_dict = "optimizer"
        scheduler_state_dict = "scheduler"
        ema_state_dict = "ema"
        global_step = "global_step"
        total_steps = "total_steps"
        current_step_in_batch = "current_step_in_batch"
        current_epoch = "current_epoch"
        total_epochs = "total_epochs"
        state_saved_in = "state_saved_in"

    class Configs(_Cfg):
        state_dir_name = "states"
        state_dict_file_name = "state_dict"

    @property
    def state_dir(self) -> Path:
        """
        Returns: the directory used to store the trainer state, including model weights
        """
        return self.project_dir / SaveAndLoadMixin.Configs.state_dir_name

    @Context.on_epoch_step_end(priority=Priority.HIGH)
    def _save_current_epoch(self, current_step, *args, **kwargs):
        self.current_epoch = current_step

    @Context.on_train_step_end(priority=Priority.HIGH)
    def _save_current_step(self, current_step, *args, **kwargs):
        self.global_step = current_step
        self.current_step_in_batch = current_step % self.total_steps_per_epoch

    def save_state(self, current_step, *args, **kwargs):
        """
        Saves current state of the trainer instance, including `model`, optimizer, project specifications and
        , if provided, scheduler and ema
        Args:
            current_step: the current training step
            *args: optional positional arguments
            **kwargs: optional keyword arguments
        """
        self.state_dir.mkdir(exist_ok=True, parents=True)
        # check whether model is saved epoch or stepwise
        state_saved_in = "epoch" if current_step == self.current_epoch else "step"
        state_dict = {SaveAndLoadMixin.Keys.model_state_dict: self.accelerator.unwrap_model(self.model).state_dict(),
                      SaveAndLoadMixin.Keys.optimizer_state_dict: self.optimizer.state_dict(),
                      SaveAndLoadMixin.Keys.total_steps: self.total_steps,
                      SaveAndLoadMixin.Keys.total_epochs: self.epochs,
                      SaveAndLoadMixin.Keys.global_step: self.global_step,
                      SaveAndLoadMixin.Keys.current_step_in_batch: self.current_step_in_batch,
                      SaveAndLoadMixin.Keys.current_epoch: self.current_epoch,
                      SaveAndLoadMixin.Keys.state_saved_in: state_saved_in,
                      SaveAndLoadMixin.Keys.scheduler_state_dict: None if self.scheduler is None \
                          else self.scheduler.state_dict(),
                      SaveAndLoadMixin.Keys.ema_state_dict: None if self.ema_model is None \
                          else self.ema_model.state_dict()}

        print(f"Saving state to {self.state_dir}")
        self.accelerator.save(state_dict,
                self.state_dir / f"{SaveAndLoadMixin.Configs.state_dict_file_name}_{current_step:010d}.pth")

    def _load_state_dict(self, obj_to_load, state_dict_coll: dict, state_key: str):
        state_dict = state_dict_coll.get(state_key)
        if state_dict is not None:
            obj_to_load.load_state_dict(state_dict)
            return True
        else:
            return False

    def load_state(self, f: str | Path | PathLike | None, load_latest: bool = True, use_old_dir: bool = True):
        """

        Args:
            f:
            load_latest:
            use_old_dir:

        """
        if f is None:
            # load from current working dir, which might not be initialized
            if not self.accelerator_initialized:
                raise RuntimeError("No project directory specified. Load the trainer state after compiling or "
                                   "give a valid path to trainer.setup_accelerator()")
            f = self.state_dir
        # redundant but catches many if else cases
        f = Path(f).resolve()
        if f.is_dir():
            fp = list(f.glob('**/*.pth'))
            if not len(fp):
                raise FileNotFoundError(f"Could not find any '.pth' file in the given directory {f}")
            if len(fp) > 1:
                if load_latest:
                    fp.sort(key= lambda _f: int(_f.stem.split("_")[-1]))
                else:
                    raise ValueError("Ambiguous state: can't infer correct file with 'f' = None and 'load_latest' = False")
            _fp = fp[-1]
        elif f.is_file():
            _fp = f
        else:
            raise ValueError(f"{f} is neither a file nor a directory")

        # at this point _fp points to a valid old project path
        if use_old_dir:
            index = _fp.parts.index(SaveAndLoadMixin.Configs.state_dir_name)
            new_fp = Path(*_fp.parts[:index])
            self.project_dir = new_fp
            # remap paths
            self.tracker.logging_dir = self.logging_dir
            self.tracker.tracker.log_dir = self.logging_dir
            self.accelerator.project_configuration.project_dir = self.project_dir
            self.accelerator.init_trackers(self.logging_dir)

        loadable_keys = [SaveAndLoadMixin.Keys.model_state_dict, SaveAndLoadMixin.Keys.optimizer_state_dict,
                         SaveAndLoadMixin.Keys.scheduler_state_dict, SaveAndLoadMixin.Keys.ema_state_dict]
        loadable_objects = [self.model, self.optimizer, self.scheduler, self.ema_model]
        state_dict: dict = torch.load(_fp, map_location='cpu')
        # TODO: Dynamic key mapping
        # TODO: provide trainables interface e.g objects that have a load_state_dict function
        success = [self._load_state_dict(obj, state_dict, key) for obj, key in zip(loadable_objects, loadable_keys)]
        current_step_in_batch = state_dict.get(SaveAndLoadMixin.Keys.current_step_in_batch)
        current_epoch = state_dict.get(SaveAndLoadMixin.Keys.current_epoch)
        start_step = state_dict.get(SaveAndLoadMixin.Keys.global_step)
        if current_step_in_batch is not None:
            self.train_dataloader = self.accelerator.skip_first_batches(
                self.train_dataloader, current_step_in_batch)
        if current_epoch is not None:
            self.start_epoch = current_epoch
        if start_step is not None:
            self.start_step = start_step

        state_dict_print = {key: value for key, value in state_dict.items() if key not in loadable_keys}
        state_dict_print.update({f"{key} included": _success for _success, key in zip(success, loadable_keys)})
        missing_keys = [key for key in self.Keys.get_members() if key not in state_dict]
        unexpected_keys = [key for key in state_dict if key not in self.Keys.get_members()]
        state_dict_print.update({"Missing keys": missing_keys, "Unexpected keys": unexpected_keys})
        print("Loading trainer state with following parameters:")
        print(json.dumps(state_dict_print, indent=4))
