"""
A script to benchmarking saving a model to a uri from torch.save.

Example usage:
python3 benchmark-model-saving.py -a posix -p /mnt/localfolder/saved_model 
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
    ALLUXIO_REST = "alluxio_rest"


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
        "ALLUXIO_REST API (e.g.s3://ref-arch/imagenet-mini/val)",
        default="./saved_model/model_sample",
    )
    parser.add_argument(
        "--workerhost",
        help="workerhost for ALLUXIO_REST API",
        default="localhost",
    )
    parser.add_argument(
        "--iteration",
        help="number of iterations to run",
        default=200,        
    )

    return parser.parse_args()

class CustomReader:
   def __init__(self, s):
       self.data_src = s
        


class VirtualFileWriter:
    PUT_PAGE_URL_FORMAT = "http://{worker_host}:{http_port}/v1/file/{file_id}/dummy/{page_index}"

    def __init__(self, path, workerhost):
        self.path = path
        self.workerhost = workerhost
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
        # now that current http api has drawbacks of a unmodifiable max http body content
        # of 4k, here we chunk it up to make each http request contain only 4k
        # of data, remove this once this restriction is lifted.
        for i in range(0,s.nbytes,4096):
            response = requests.put(
               self.PUT_PAGE_URL_FORMAT.format(
                   worker_host=self.workerhost, http_port=28080,
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
            '''
            print("s len:{},cur_len:{},range:{}-{}".format(
                s.nbytes, len(s[i:i+4096]), i, i+4096))
            '''
            if response.status_code != 200:
                _logger.error(f"response:{response.reason}")
                _logger.error(f"Upload failed {response.status_code} fileid:{self.file_id} pageid:{self.page_id_counter}")

    def flush(self):
        print("FLUSHED!")

class ModelCheckpointBenchmark:
    def __init__(
        self,
        api,
        path,
        workerhost
    ):
        self.api = api
        self.path = path
        self.workerhost = workerhost
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
        path = self.path
        #path = self.path + "_" + str(self.counter)
        #self.counter += 1
        if not self.model:
            self.model = self.prepare_model()
        start_time = time.perf_counter()
        if self.api == APIType.POSIX.value:
            torch.save(self.model.state_dict(), path)
        elif self.api == APIType.ALLUXIO_REST.value:
            virtual_file_writer = VirtualFileWriter(pathi, self.workerhost)
            torch.save(self.model.state_dict(), virtual_file_writer)
        end_time = time.perf_counter()
        _logger.info(f"Model checkpointing time in {end_time - start_time:0.4f} seconds")

    def cleanup(self):
        _logger.info(f"Cleaning up... path exists?:{self.path}:{os.path.exists(self.path)}")
        if self.api == APIType.POSIX.value:
            if os.path.exists(self.path):
                os.remove(self.path)
                _logger.info(f"removed file:{self.path}")


if __name__ == "__main__":
    args = get_args()
    model_checkpoint_benchmark = ModelCheckpointBenchmark(
        api=args.api, path=args.path, workerhost=args.workerhost
    )
    test_iteration = args.iteration
    for _ in range(test_iteration):
        model_checkpoint_benchmark.run()
