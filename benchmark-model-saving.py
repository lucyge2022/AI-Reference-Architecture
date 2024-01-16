"""
A script to benchmarking saving a model to a uri from torch.save.

Example usage:
python3 benchmark-model-saving.py -a posix -p /mnt/localfolder/saved_model --etcd localhost
"""
import argparse
import logging
import time
from logging.config import fileConfig
from enum import Enum
import requests
import urllib.parse
import requests
from requests.adapters import HTTPAdapter
import os, sys

import torch
import torchvision
import torchvision.transforms as transforms

#from alluxio import AlluxioFileSystem 

log_conf_path = "./conf/logging.conf"
fileConfig(log_conf_path, disable_existing_loggers=True)
_logger = logging.getLogger("ModelCheckpointBenchmark")
# Explicitly disable the PIL.TiffImagePlugin logger as it also uses
# the StreamHandler which will overrun the console output.
logging.getLogger("PIL.TiffImagePlugin").disabled = True

class APIType(Enum):
    POSIX = "posix"
    ALLUXIO = "alluxio_fake"
    S3 = "s3"


def get_args():
    parser = argparse.ArgumentParser(description="Benchmark PyTorch Model Saving")

    parser.add_argument(
        "-a",
        "--api",
        help="The API to use. default is posix",
        choices=[e.value for e in APIType],
        default=APIType.POSIX.value,
    )
    parser.add_argument(
        "-p",
        "--path",
        help="Local POSIX PATH if API type is POSIX, full ufs path if "
        "ALLUXIO/S3 API (e.g.s3://ref-arch/imagenet-mini/val)",
        default="./saved_model/model_sample",
    )
    parser.add_argument(
        "--etcd",
        help="Alluxio API require ETCD hostname",
        default="localhost",
    )

    return parser.parse_args()

class CustomReader:
   def __init__(self, s):
       self.data_src = s
        


class VirtualFileWriter:
    PUT_PAGE_URL_FORMAT = "http://{worker_host}:{http_port}/v1/file/{file_id}/dummy/{page_index}"

    def __init__(self, path):
        self.path = path
        self.session = requests.Session()
        concurrency = 64
        adapter = HTTPAdapter(
            pool_connections=concurrency, pool_maxsize=concurrency
        )
        self.session.mount("http://", adapter)
        self.buf = []
        self.file_id = urllib.parse.quote(self.path, safe="")
        self.page_id_counter = 1

    def write(self, s):
        for i in range(0,s.nbytes,4096):
            response = requests.put(
               self.PUT_PAGE_URL_FORMAT.format(
                   worker_host="localhost", http_port=28080,
                   file_id=self.file_id, page_index=self.page_id_counter
               ), data=s[i:i+4096])
            '''
            response = self.session.put(
            self.PUT_PAGE_URL_FORMAT.format(
                worker_host="localhost", http_port=28080,
                file_id=self.file_id, page_index=self.page_id_counter
            ), data=s)
            '''
            self.page_id_counter += 1
            #print("s len:{},cur_len:{},range:{}-{}".format(
            #    s.nbytes, len(s[i:i+4096]), i, i+4096))
            if response.status_code != 200:
                _logger.info(f"response:{response.reason}")
                _logger.info(f"Upload failed {response.status_code} fileid:{self.file_id} pageid:{self.page_id_counter}")
                sys.exit(-1)

    def flush(self):
        print("FLUSHED!")

class ModelCheckpointBenchmark:
    def __init__(
        self,
        api,
        path
    ):
        self.api = api
        self.path = path
        self.model = None
        self.counter = 0

    def prepare_model(self):
        #model = torchvision.models.resnet18(pretrained=True)
        from transformers import BertModel, BertTokenizer, BertConfig
        model = BertModel.from_pretrained("bert-base-uncased", torchscript=True)
        _logger.info(
            f"Current time after loading the model: {time.perf_counter()}"
        )

        # Parallelize training across multiple GPUs
        #model = torch.nn.DataParallel(model)
        return model 


    def run(self):
        path = self.path + "_" + str(self.counter)
        self.counter += 1
        if not self.model:
            self.model = self.prepare_model()
        start_time = time.perf_counter()
        if self.api == APIType.POSIX.value:
            torch.save(self.model.state_dict(), path)
        elif self.api == APIType.ALLUXIO.value:
            virtual_file_writer = VirtualFileWriter(path)
            torch.save(self.model.state_dict(), virtual_file_writer)
        end_time = time.perf_counter()
        _logger.info(f"Model checkpointing time in {end_time - start_time:0.4f} seconds")

    def cleanup(self):
        _logger.info(f"Cleaning up... path exists?:{self.path}:{os.path.exists(self.path)}")
        if self.api == APIType.POSIX.value:
            if os.path.exists(self.path):
                os.remove(self.path)
                _logger.info(f"removed file:{self.path}")


class ResnetTrainer:
    def __init__(
        self,
        input_path="/mnt/alluxio/fuse/imagenet-mini/train",
        output_path="./resnet-imagenet-model.pth",
        profiler_log_path="./log/resnet",
        num_epochs=3,
        batch_size=128,
        num_workers=16,
        learning_rate=0.001,
        profiler_enabled=False,
    ):
        _logger.info(f"Start time: {time.perf_counter()}")

        self.input_path = input_path
        self.output_path = output_path
        self.profiler_log_path = profiler_log_path
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.learning_rate = learning_rate
        self.profiler_enabled = profiler_enabled

        self.train_loader = None
        self.model = None

        self.device = self._check_device()

    def run_trainer(self):
        self.train_loader = self._create_data_loader()
        self.model = self._load_model()
        self._train()
        self._save_model()

    def _create_data_loader(self):
        transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(degrees=45),
                transforms.ColorJitter(
                    brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5
                ),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                ),
            ]
        )

        train_dataset = torchvision.datasets.ImageFolder(
            root=self.input_path, transform=transform
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

        return train_loader

    def _load_model(self):
        model = torchvision.models.resnet18(pretrained=True)
        _logger.info(
            f"Current time after loading the model: {time.perf_counter()}"
        )

        # Parallelize training across multiple GPUs
        model = torch.nn.DataParallel(model)

        # Set the model to run on the device
        model = model.to(self.device)
        _logger.info(
            f"Current time after loading the model to device: {time.perf_counter()}"
        )

        return model

    def _train(self):
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate
        )

        start_time = time.perf_counter()

        profiler = None
        if self.profiler_enabled:
            profiler = torch.profiler.profile(
                schedule=torch.profiler.schedule(
                    wait=0, warmup=0, active=1, repeat=1
                ),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    self.profiler_log_path
                ),
            )
            profiler.start()

        for epoch in range(self.num_epochs):
            batch_start = time.perf_counter()
            for inputs, labels in self.train_loader:
                # Move input and label tensors to the device
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                batch_end = time.perf_counter()
                _logger.debug(
                    f"Loaded input and labels to the device in "
                    f"{batch_end - batch_start:0.4f} seconds"
                )

                # Zero out the optimization
                optimizer.zero_grad()

                # Forward pass
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)

                # Backward pass
                loss.backward()
                optimizer.step()

                batch_start = time.perf_counter()

            _logger.info(
                f"Epoch {epoch + 1}/{self.num_epochs}, Loss: {loss.item():.4f} "
                f"at the timestamp {time.perf_counter()}"
            )

            if self.profiler_enabled:
                profiler.step()

        _logger.info(f"Finished Training, Loss: {loss.item():.4f}")

        end_time = time.perf_counter()
        _logger.info(f"Training time in {end_time - start_time:0.4f} seconds")

        if self.profiler_enabled:
            profiler.stop()

    def _save_model(self):
        torch.save(self.model.state_dict(), self.output_path)
        _logger.info(f"Saved PyTorch Model State to {self.output_path}")

    def _check_device(self):
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        _logger.debug(f"Using {device}")

        return device


if __name__ == "__main__":
    args = get_args()
    model_checkpoint_benchmark = ModelCheckpointBenchmark(
        api=args.api, path=args.path
    )
    start = time.time()
    while time.time() - start < 5 * 60:
        model_checkpoint_benchmark.run()
        #break
        #model_checkpoint_benchmark.cleanup()
