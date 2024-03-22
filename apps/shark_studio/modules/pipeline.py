from shark.iree_utils.compile_utils import (
    get_iree_compiled_module,
    load_vmfb_using_mmap,
    clean_device_info,
    get_iree_target_triple,
)
from apps.shark_studio.web.utils.file_utils import (
    get_checkpoints_path,
    get_resource_path,
)
from apps.shark_studio.modules.shared_cmd_opts import (
    cmd_opts,
)
from iree import runtime as ireert
from pathlib import Path
import gc
import os


class SharkPipelineBase:
    # This class is a lightweight base for managing an
    # inference API class. It should provide methods for:
    # - compiling a set (model map) of torch IR modules
    # - preparing weights for an inference job
    # - loading weights for an inference job
    # - utilites like benchmarks, tests

    def __init__(
        self,
        model_map: dict,
        base_model_id: str,
        static_kwargs: dict,
        device: str,
        import_mlir: bool = True,
    ):
        self.model_map = model_map
        self.pipe_map = {}
        self.static_kwargs = static_kwargs
        self.base_model_id = base_model_id
        self.triple = get_iree_target_triple(device)
        self.device, self.device_id = clean_device_info(device)
        self.import_mlir = import_mlir
        self.iree_module_dict = {}
        self.tmp_dir = get_resource_path(os.path.join("..", "shark_tmp"))
        if not os.path.exists(self.tmp_dir):
            os.mkdir(self.tmp_dir)
        self.tempfiles = {}
        self.pipe_vmfb_path = ""

    def get_compiled_map(self, pipe_id, submodel="None", init_kwargs={}) -> None:
        # First checks whether we have .vmfbs precompiled, then populates the map
        # with the precompiled executables and fetches executables for the rest of the map.
        # The weights aren't static here anymore so this function should be a part of pipeline
        # initialization. As soon as you have a pipeline ID unique to your static torch IR parameters,
        # and your model map is populated with any IR - unique model IDs and their static params,
        # call this method to get the artifacts associated with your map.
        self.pipe_id = self.safe_name(pipe_id)
        self.pipe_vmfb_path = Path(
            os.path.join(get_checkpoints_path(".."), self.pipe_id)
        )
        self.pipe_vmfb_path.mkdir(parents=False, exist_ok=True)
        if submodel == "None":
            print("\n[LOG] Gathering any pre-compiled artifacts....")
            for key in self.model_map:
                self.get_compiled_map(pipe_id, submodel=key)
        else:
            self.pipe_map[submodel] = {}
            self.get_precompiled(self.pipe_id, submodel)
            ireec_flags = []
            if submodel in self.iree_module_dict:
                return
            elif "vmfb_path" in self.pipe_map[submodel]:
                return
            elif submodel not in self.tempfiles:
                print(
                    f"\n[LOG] Tempfile for {submodel} not found. Fetching torch IR..."
                )
                if submodel in self.static_kwargs:
                    init_kwargs = self.static_kwargs[submodel]
                for key in self.static_kwargs["pipe"]:
                    if key not in init_kwargs:
                        init_kwargs[key] = self.static_kwargs["pipe"][key]
                self.import_torch_ir(submodel, init_kwargs)
                self.get_compiled_map(pipe_id, submodel)
            else:
                ireec_flags = (
                    self.model_map[submodel]["ireec_flags"]
                    if "ireec_flags" in self.model_map[submodel]
                    else []
                )

                weights_path = self.get_io_params(submodel)

                self.iree_module_dict[submodel] = get_iree_compiled_module(
                    self.tempfiles[submodel],
                    device=self.device,
                    frontend="torch",
                    mmap=True,
                    external_weight_file=weights_path,
                    extra_args=ireec_flags,
                    write_to=os.path.join(self.pipe_vmfb_path, submodel + ".vmfb"),
                )
        return

    def get_io_params(self, submodel):
        if "external_weight_file" in self.static_kwargs[submodel]:
            # we are using custom weights
            weights_path = self.static_kwargs[submodel]["external_weight_file"]
        elif "external_weight_path" in self.static_kwargs[submodel]:
            # we are using the default weights for the HF model
            weights_path = self.static_kwargs[submodel]["external_weight_path"]
        else:
            # assume the torch IR contains the weights.
            weights_path = None
        return weights_path

    def get_precompiled(self, pipe_id, submodel="None"):
        if submodel == "None":
            for model in self.model_map:
                self.get_precompiled(pipe_id, model)
        vmfbs = []
        for dirpath, dirnames, filenames in os.walk(self.pipe_vmfb_path):
            vmfbs.extend(filenames)
            break
        for file in vmfbs:
            if submodel in file:
                self.pipe_map[submodel]["vmfb_path"] = os.path.join(
                    self.pipe_vmfb_path, file
                )
        return

    def import_torch_ir(self, submodel, kwargs):
        torch_ir = self.model_map[submodel]["initializer"](
            **self.safe_dict(kwargs), compile_to="torch"
        )
        if submodel == "clip":
            # clip.export_clip_model returns (torch_ir, tokenizer)
            torch_ir = torch_ir[0]

        self.tempfiles[submodel] = os.path.join(
            self.tmp_dir, f"{submodel}.torch.tempfile"
        )

        with open(self.tempfiles[submodel], "w+") as f:
            f.write(torch_ir)
        del torch_ir
        gc.collect()
        return

    def load_submodels(self, submodels: list):
        for submodel in submodels:
            if submodel in self.iree_module_dict:
                print(f"\n[LOG] {submodel} is ready for inference.")
                continue
            if "vmfb_path" in self.pipe_map[submodel]:
                weights_path = self.get_io_params(submodel)
                # print(
                #     f"\n[LOG] Loading .vmfb for {submodel} from {self.pipe_map[submodel]['vmfb_path']}"
                # )
                self.iree_module_dict[submodel] = {}
                (
                    self.iree_module_dict[submodel]["vmfb"],
                    self.iree_module_dict[submodel]["config"],
                    self.iree_module_dict[submodel]["temp_file_to_unlink"],
                ) = load_vmfb_using_mmap(
                    self.pipe_map[submodel]["vmfb_path"],
                    self.device,
                    device_idx=0,
                    rt_flags=[],
                    external_weight_file=weights_path,
                )
            else:
                self.get_compiled_map(self.pipe_id, submodel)
        return

    def unload_submodels(self, submodels: list):
        for submodel in submodels:
            if submodel in self.iree_module_dict:
                del self.iree_module_dict[submodel]
                gc.collect()
        return

    def run(self, submodel, inputs):
        if not isinstance(inputs, list):
            inputs = [inputs]
        inp = [
            ireert.asdevicearray(
                self.iree_module_dict[submodel]["config"].device, input
            )
            for input in inputs
        ]
        return self.iree_module_dict[submodel]["vmfb"]["main"](*inp)

    def safe_name(self, name):
        return name.replace("/", "_").replace("-", "_").replace("\\", "_")

    def safe_dict(self, kwargs: dict):
        flat_args = {}
        for i in kwargs:
            if isinstance(kwargs[i], dict) and "pass_dict" not in kwargs[i]:
                flat_args[i] = [kwargs[i][j] for j in kwargs[i]]
            else:
                flat_args[i] = kwargs[i]

        return flat_args
