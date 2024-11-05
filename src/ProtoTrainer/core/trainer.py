import operator
import os
import datetime
import json
import torch

from accelerate import Accelerator
from accelerate.tracking import TensorBoardTracker
from torch.utils.data import DataLoader
from functools import wraps, partial
from rich import print
from tqdm.auto import tqdm
from pathlib import Path
from typing import Protocol, Callable, final
from abc import abstractmethod, ABC, ABCMeta
from datasets import Dataset
from ProtoTrainer.core.generic.interface import TrainerInterface
from ProtoTrainer.core.context import Context, execute_for_step

from accelerate.utils.dataclasses import (
    PrecisionType,
    DeepSpeedPlugin,
    FullyShardedDataParallelPlugin,
    MegatronLMPlugin,
    RNGType,
    LoggerType,
    ProjectConfiguration,
    GradientAccumulationPlugin,
    DynamoBackend,
    KwargsHandler)
from accelerate.tracking import GeneralTracker

from ProtoTrainer.core.generic.base_mixin import BaseMixin


# TODO: accelerate.utils.set_seed() for deterministic seed behaviour across GPUs
# TODO: prepare ema_model for parallelized inference, need to .unwrap() in save_model too
# TODO: context class: Done afaik
# TODO: functional interface, including context, and execute params
#       -> ADD save_state and various other functions
#       -> What to do with stateful function ?
# TODO: preset interface for auxiliary functions that want to use Trainer internal attributes
# TODO: separate Context and ContextManager and do __init__(type(Context)) -> ?
# TODO: need to rework the context stuff cause its awful...
# TODO: optimizer pre and post step have "step" as arg, remaining has "current_step", need no adapt




class OnTrainStepEndProtocol(Protocol):
    """
    The protocol for all functions that are registered with `on_train_step_end`.

    Expected signature is:
    ::
        def func(current_step: int, processed_batch: torch.Tensor, batch: dict) -> None:
    """

    def __call__(self,
                 current_step: int,
                 processed_batch: torch.Tensor | None,
                 batch: dict or None):
        ...


class OnEpochStepEndProtocol(Protocol):
    """
    The protocol for all functions that are registered with `on_epoch_step_end`.

    Expected signature is:
    ::
        def func(current_step: int, train_dataloader: Dataloader, validation_dataloader: Dataloader) -> None:

    """

    def __call__(self,
                 current_step: int,
                 train_dataloader: DataLoader | None,
                 validation_dataloader: DataLoader | None):
        ...


class _TrainerMeta(type):
    #def __call__(cls, *args, **kwargs):
        #cls.__mixin_collection = []
        #instance = super().__call__(*args, **kwargs)
        #for base in cls.__mro__:
            #if issubclass(base, BaseMixin) and base is not BaseMixin and base is not cls:
                #cls.__mixin_collection.append(base)
        #return instance

    def __new__(cls, name, bases, attrs):

        def init_mixins(mixin_list: list[type]):
            def _init_mixins(self):
                for mixin in mixin_list:
                    mixin.__init__(self)
            return _init_mixins

        mixins = [base for base in bases if issubclass(base, BaseMixin) and base is not BaseMixin and base is not cls]
        attrs['mixins'] = []
        if mixins:
            attrs['mixins'] = mixins
        attrs["init_mixins"] = init_mixins(attrs["mixins"])
        return super().__new__(cls, name, bases, attrs)

class TrainerMeta(_TrainerMeta, ABCMeta):
    pass


class BaseTrainer(TrainerInterface, metaclass=TrainerMeta):
    """
    Accelerator based generic trainer class.

    Attributes:
        tracker: The tracker accelerator is initialized with. Defaults to a *tensorboard* tracker
        device: The device the trainer is placed at. Defaults to GPU based training
        accelerator: The initialized accelerator instance
        model: The initialized model instance
        optimizer: The initialized optimizer instance
        scheduler: The initialized scheduler instance if given
        ema_model: The weight-averaged model instance if used
        train_dataloader: The dataloader used to train the model
        validation_dataloader: The dataloader used to validate the model, if given
        project_dir: The active working directory of the trainer instance.
        logging_dir: The directory used to store tensorboard event files
        epochs: The number of epochs to train the model
        total_steps: The total number of steps to train the model
        total_steps_per_epoch: The number of steps per epoch
        accelerator_initialized: If accelerator is fully initialized
        compiled: If the trained instance is compiled
        cfg: Any configuration given during initialization

    """

    def __init__(self,
                 cfg: dict | None = None):
        """
        Args:
            cfg: Additional config file to access external data
        """
        # Accelerator setup
        self.__accelerator_args = None

        # misc
        self.cfg = cfg
        self._loss_dict = None

        # setup callable stacks
        self._context_manager = Context()

        # names
        self._default_project_dir = "logs"
        self._default_tensorboard_dir = "tensorboard"
        self.__mixin_interface_function = "post_init"

    def init(self):
        """
        Function to initialize trainer attributes that are dependent on runtime attributes e.g. ``self.project_dir`` or
        ``self.total_steps_per_epoch``.Trainer attributes accessed in `init` are fully defined.
        """
        pass

    def setup_accelerator(
            self,
            device_placement: bool = True,
            split_batches: bool = False,
            mixed_precision: PrecisionType | str | None = "fp16",
            gradient_accumulation_steps: int = 1,
            cpu: bool = False,
            deepspeed_plugin: DeepSpeedPlugin | None = None,
            fsdp_plugin: FullyShardedDataParallelPlugin | None = None,
            megatron_lm_plugin: MegatronLMPlugin | None = None,
            rng_types: list[str | RNGType] | None = None,
            log_with: str | LoggerType | GeneralTracker | list[
                str | LoggerType | GeneralTracker] | None = "tensorboard",
            project_dir: str | os.PathLike | None = None,
            project_config: ProjectConfiguration | None = None,
            gradient_accumulation_plugin: GradientAccumulationPlugin | None = None,
            dispatch_batches: bool | None = None,
            even_batches: bool = True,
            step_scheduler_with_optimizer: bool = True,
            kwargs_handlers: list[KwargsHandler] | None = None,
            dynamo_backend: DynamoBackend | str | None = "inductor"
    ):
        """
        Args:
            device_placement (`bool`, *optional*, defaults to `True`):
                Whether the accelerator should put objects on device (tensors yielded by the dataloader, model,
                etc...).
            split_batches (`bool`, *optional*, defaults to `False`):
                Whether the accelerator should split the batches yielded by the dataloaders across the devices. If
                `True` the actual batch size used will be the same on any kind of distributed processes, but it must be a
                round multiple of the `num_processes` you are using. If `False`, actual batch size used will be the one set
                in your script multiplied by the number of processes.
            mixed_precision (`str`, *optional*):
                Whether to use mixed precision training. Choose from 'no','fp16','bf16 or 'fp8'. Will default to the
                value in the environment variable `ACCELERATE_MIXED_PRECISION`, which will use the default value in the
                accelerate config of the current system or the flag passed with the `accelerate.launch` command. 'fp8'
                requires the installation of transformers-engine.
            gradient_accumulation_steps (`int`, *optional*, default to 1):
                The number of steps that should pass before gradients are accumulated. A number > 1 should be combined with
                `Accelerator.accumulate`. If not passed, will default to the value in the environment variable
                `ACCELERATE_GRADIENT_ACCUMULATION_STEPS`. Can also be configured through a `GradientAccumulationPlugin`.
            cpu (`bool`, *optional*):
                Whether to force the script to execute on CPU. Will ignore GPU available if set to `True` and force
                the execution on one process only.
            deepspeed_plugin (`DeepSpeedPlugin`, *optional*):
                Tweak your DeepSpeed related args using this argument. This argument is optional and can be configured
                directly using *accelerate config*
            fsdp_plugin (`FullyShardedDataParallelPlugin`, *optional*):
                Tweak your FSDP related args using this argument. This argument is optional and can be configured directly
                using *accelerate config*
            megatron_lm_plugin (`MegatronLMPlugin`, *optional*):
                Tweak your MegatronLM related args using this argument. This argument is optional and can be configured
                directly using *accelerate config*
            rng_types (list of `str` or [`~utils.RNGType`]):
                The list of random number generators to synchronize at the beginning of each iteration in your prepared
                dataloaders. Should be one or several of:

                - `"torch"`: the base torch random number generator
                - `"cuda"`: the CUDA random number generator (GPU only)
                - `"xla"`: the XLA random number generator (TPU only)
                - `"generator"`: the `torch.Generator` of the sampler (or batch sampler if there is no sampler in your
                  dataloader) or of the iterable dataset (if it exists) if the underlying dataset is of that type.

                Will default to `["torch"]` for PyTorch versions <=1.5.1 and `["generator"]` for PyTorch versions >= 1.6.
            log_with (list of `str`, [`~utils.LoggerType`] or [`~tracking.GeneralTracker`], *optional*):
                A list of loggers to be setup for experiment tracking. Should be one or several of:

                - `"all"`
                - `"tensorboard"`
                - `"wandb"`
                - `"comet_ml"`
                If `"all"` is selected, will pick up all available trackers in the environment and initialize them. Can
                also accept implementations of `GeneralTracker` for custom trackers, and can be combined with `"all"`.
            project_config (`ProjectConfiguration`, *optional*):
                A configuration for how saving the state can be handled.
            project_dir (`str`, `os.PathLike`, *optional*):
                A path to a directory for storing data such as logs of locally-compatible loggers and potentially saved
                checkpoints. If no directory is specified, a default directory will be created.
            dispatch_batches (`bool`, *optional*):
                If set to `True`, the dataloader prepared by the Accelerator is only iterated through on the main process
                and then the batches are split and broadcast to each process. Will default to `True` for `DataLoader` whose
                underlying dataset is an `IterableDataset`, `False` otherwise.
            even_batches (`bool`, *optional*, defaults to `True`):
                If set to `True`, in cases where the total batch size across all processes does not exactly divide the
                dataset, samples at the start of the dataset will be duplicated so the batch can be divided equally among
                all workers.
            step_scheduler_with_optimizer (`bool`, *optional`, defaults to `True`):
                Set `True` if the learning rate scheduler is stepped at the same time as the optimizer, `False` if only
                done under certain circumstances (at the end of each epoch, for instance).
            kwargs_handlers (`list[KwargHandler]`, *optional*)
                A list of `KwargHandler` to customize how the objects related to distributed training or mixed precision
                are created. See [kwargs](kwargs) for more information.
            dynamo_backend (`str` or `DynamoBackend`, *optional*, defaults to `"no"`):
                Set to one of the possible dynamo backends to optimize your training with torch dynamo.
            gradient_accumulation_plugin (`GradientAccumulationPlugin`, *optional*):
                A configuration for how gradient accumulation should be handled, if more tweaking than just the
                `gradient_accumulation_steps` is needed.
        """
        self.__accelerator_args = {**locals()}
        self.__accelerator_args.pop("self")
        if log_with != "tensorboard":
            raise NotImplementedError("Currently only `tensorboard` logging is supported.")
        if project_dir is not None:
            self.__accelerator_init(**self.__accelerator_args)
        else:
            # postpone init and dynamically create the project dir
            print("No project directory specified. Default directory will be used.")

    def __accelerator_init(self, **kwargs):
        self.accelerator = Accelerator(
            **kwargs
        )
        if self.accelerator.is_main_process:
            os.makedirs(kwargs["project_dir"], exist_ok=True)
            print(f"Project directory defined at {kwargs['project_dir']}.")
            log_dir = Path(kwargs['project_dir'], self._default_tensorboard_dir)
            print(f"Logging tensorboard data to: {str(log_dir)}")
            self.accelerator.init_trackers(project_name=str(log_dir))
            self.tracker: TensorBoardTracker | GeneralTracker = self.accelerator.get_tracker(kwargs["log_with"])
            self.device = self.accelerator.device
            self.project_dir = Path(kwargs["project_dir"])
            self.accelerator_initialized = True

    @property
    def logging_dir(self):
        return self.project_dir / self._default_tensorboard_dir

    @property
    def on_train_step_end(self) -> list[Callable[[int, torch.Tensor, dict], None]]:
        """
        The callback stack containing all functions that are executed after the train step.
        """
        return self._context_manager.on_train_step_end

    @property
    def on_epoch_step_end(self) -> list[Callable[[int], None]]:
        """
        The callback stack containing all functions that are executed after one epoch.
        """
        return self._context_manager.on_epoch_step_end

    @abstractmethod
    def preprocess(self, batch: dict) -> torch.Tensor:
        """
        Preprocessing function to convert one batch to a
        torch Tensor representation for the forward pass
         """
        raise NotImplementedError("Preprocess function must be implemented in subclass")

    @abstractmethod
    def compute_loss(self, processed_batch) -> torch.Tensor | dict[torch.Tensor]:
        """Objective function to be used during training"""
        raise NotImplementedError("Compute loss function must be implemented in subclass")

    def _create_progress_bar(self):
        return tqdm(
            total=self.total_steps - self.start_step + 1,
            dynamic_ncols=True,
            disable=not self.accelerator.is_main_process)

    def _count_parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def _compute_relative_stepsize(self, step: float | int, step_size: str) -> int:
        total_steps = self.epochs if step_size == "epoch" else self.total_steps_per_epoch
        return int(total_steps * step) if isinstance(step, float) else total_steps

    def _decorate_remaining_functions(self):
        # get all user defined functions
        # all functions that are not added to any callback stack get ignored
        funcs = self._context_manager.retrieve_decorated_functions(self)
        # TODO: doesn't include mixin functions, need to be fixed
        step_wise = [Context.on_train_step_end.name,
                     Context.optimizer_post_step.name, Context.optimizer_pre_step.name]
        epoch_wise = [Context.on_epoch_step_end.name]
        for func in funcs:
            func_body = getattr(self, func)
            context = self._context_manager.get_function_context(func_body, raise_exception=True)
            if context in step_wise:
                step_size = "step"
            elif context in epoch_wise:
                step_size = "epoch"
            else:
                raise RuntimeError(f"Can't deduct step size.")
            # get the n step params
             # check if deferred decoration, apply decorator on instance level
            to_decorate = getattr(func_body, "to_decorate", None)
            if to_decorate is not None:
                kwargs = to_decorate["execute_for_step"]["kwargs"]
                kwargs["n"] = self._compute_relative_stepsize(kwargs["n"], step_size=step_size)
                # func_body is already bound, and thus passes self to the decorator
                # need to unbind first
                # we have function objects, so we need to decorate the underlying function without losing the object
                #func_body = execute_for_step(**kwargs)(getattr(type(self), func))
                func_body._function = execute_for_step(**kwargs)(func_body._function)
                # then rebind
                #func_body = func_body.__get__(self, type(self))
                #setattr(self, func, func_body)
            # add to the correct stack
            self._context_manager.register(func_body)

    @final
    def compile_trainer(
            self,
            epochs: int,
            model: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
            train_dataloader: DataLoader,
            validation_dataloader: DataLoader | None = None,
            scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
            use_ema: bool = False,
            verbose: bool = True) -> None:
        """
        Initializes all runtime dependent attributes. Must be called after `setup_accelerator` and before `train`.        Args:

        Args:
            epochs: total epochs to train the model.
            model: model to be trained
            optimizer: optimizer to be used
            train_dataloader: dataset used for training
            validation_dataloader: dataset used for validation
            scheduler: learning rate scheduler
            use_ema: whether exponential moving average weights of the model shall be computed
            verbose: whether to output a detailed configuration to the console

        Raises:
            RuntimeError: if accelerator is not initialized
            ValueError: if the underlying dataset of the data loader is not of type `Dataset`
        """

        # setup
        if not self.accelerator_initialized:
            if self.__accelerator_args is None:
                raise RuntimeError("Please initialize accelerator first with 'setup_accelerator(...)'")
            else:
                _d: Dataset = train_dataloader.dataset
                if not isinstance(_d, Dataset):
                    raise ValueError("The underlying dataset must be of type `Dataset`")
                project_name = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
                dir_path = Path(os.getcwd()) / self._default_project_dir / _d.builder_name / _d.config_name / project_name
                self.__accelerator_args["project_dir"] = dir_path
                self.__accelerator_init(**self.__accelerator_args)

        # don't pass as list, otherwise inductor won't get called properly

        # noinspection PyTypeChecker
        model, optimizer, scheduler, train_dataloader, validation_dataloader = self.accelerator.prepare(
            model, optimizer, scheduler, train_dataloader, validation_dataloader)
        self.epochs = epochs
        self.model = model.to(self.device)
        # scale the lr linearly to account for multi gpu training
        for parameter_group in optimizer.param_groups:
            parameter_group['lr'] *= self.accelerator.num_processes
        self.optimizer = optimizer
        if scheduler is not None:
            self.scheduler = scheduler
            if self.__accelerator_args["step_scheduler_with_optimizer"]:
                # append did not insert a priority
                self._context_manager.optimizer_post_step.append(self.update_scheduler)
                self._context_manager.on_train_step_end.append(self.log_scheduler)
            else:
                self._context_manager.on_epoch_step_end.append(self.update_scheduler)
                self._context_manager.on_epoch_step_end.append(self.log_scheduler)

        if use_ema:
            self.ema_model = torch.optim.swa_utils.AveragedModel(
                self.model,
                multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.999))
            self.ema_model.requires_grad_(False)
            self._context_manager.on_epoch_step_end.append(self.update_ema)

        self.train_dataloader: DataLoader = train_dataloader
        self.validation_dataloader: DataLoader = validation_dataloader
        self.total_steps_per_epoch = len(train_dataloader) // self.accelerator.gradient_accumulation_steps
        self.total_steps = self.total_steps_per_epoch * self.epochs
        self.init()
        self.init_mixins()
        # update deferred decorators
        self._decorate_remaining_functions()
        # run any mixin post inits
        #for cls in type(self)._Tr
        self.compiled = True

        if verbose:
            if self.accelerator.is_main_process:
                if self.cfg is not None:
                    print(f"\nPassed Trainer config: {json.dumps(self.cfg, indent=4)}")
                print(f"\nNumber of parameters: {self._count_parameters():,}")
                print(f"\nDataset information:\n"
                      f"    Train: {train_dataloader.dataset}")
                val_params = validation_dataloader.dataset if validation_dataloader is not None else ("No validation "
                                                                                                      "dataset given")
                print(f"    Validation: {val_params}")
                print(f"\nSteps per epoch: {self.total_steps_per_epoch}")
                print(f"Number of total training steps: {self.total_steps}\n")

    def update_scheduler(self, step):
        self.scheduler.step()

    def log_scheduler(self, current_step, *args, **kwargs):
        self.tracker.log({"train/lr": self.scheduler.get_last_lr()[0]}, step=current_step)

    def update_ema(self, epoch, *args, **kwargs):
        # print("\n Updating EMA model")
        self.ema_model.update_parameters(self.model)

    def _process_loss(self, processed_batch):
        loss = self.compute_loss(processed_batch=processed_batch)
        if isinstance(loss, dict):
            retloss = sum(loss.values())
            self._loss_dict = {"train/"+key: value.item() for key, value in loss.items()}
        else:
            retloss = loss
            self._loss_dict = {"train/loss": retloss.item()}
        return retloss

    @Context.on_train_step_end
    def log_loss(self, current_step, *args, **kwargs):
        self.accelerator.log(self._loss_dict, step=current_step)

    def train(self):
        global_step = self.start_step
        progress_bar = self._create_progress_bar()
        self.model.train()

        # Main Loop
        for epoch in range(self.start_epoch, self.epochs + 1):
            progress_bar.set_description(f"Training: Epoch {epoch}/{self.epochs}")
            for batch in self.train_dataloader:
                data = self.preprocess(batch)
                with self.accelerator.accumulate(self.model):
                    loss = self._process_loss(processed_batch=data)
                    self.accelerator.backward(loss)
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self._context_manager.optimizer_pre_step.run(step=global_step)
                    self.optimizer.step()
                    self._context_manager.optimizer_post_step.run(step=global_step)
                    self.optimizer.zero_grad()

                # execute the step dependant functions
                if self.accelerator.is_main_process:
                    self._context_manager.on_train_step_end.run(
                        current_step=global_step, processed_batch=data, batch=batch)
                    global_step += 1
                    progress_bar.update(1)

            if self.accelerator.is_main_process:
                self._context_manager.on_epoch_step_end.run(current_step=epoch)

        self.accelerator.end_training()

    def preprocess_validation_data(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("No preprocess function for validation data implemented")

    def inference(self, processed_batch: torch.Tensor, processed_target: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("No inference function implemented")

    def eval(self):
        self.model.eval()
        if self.validation_dataloader is None:
            raise RuntimeError("No validation dataloader given during model.compile")
        for batch in self.validation_dataloader:
            processed_batch, processed_target = self.preprocess_validation_data(batch)
            prediction = self.inference(processed_batch, processed_target)
            all_predictions, all_targets = self.accelerator.gather_for_metrics((prediction, processed_target))

