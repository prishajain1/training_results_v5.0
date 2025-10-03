# Copyright (c) 2023-2025, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import logging
import math
import sys
import os
import torch
import torch.cuda.nvtx as nvtx

import mlperf_logging.mllog.constants as mllog_constants
from mlperf_common.frameworks.hugectr import HCTRCommunicationHandler
from mlperf_common.logging import MLLoggerWrapper
from mpi4py import MPI

import hugectr
import mlperf_logger
import sharding

TRAIN_NUM_SAMPLES = 4195197692
EVAL_NUM_SAMPLES = 89137319
TABLE_SIZE_ARRAY = [
    40000000, 39060, 17295, 7424, 20265, 3, 7122, 1543, 63, 40000000,
    3067956, 405282, 10, 2209, 11938, 155, 4, 976, 14, 40000000, 40000000,
    40000000, 590152, 12973, 108, 36,
]
MULTI_HOT_SIZES = [
    3, 2, 1, 2, 6, 1, 1, 1, 1, 7, 3, 8, 1, 6, 9, 5, 1, 1, 1, 12, 100, 27, 10,
    3, 1, 1,
]
NUM_TABLE = len(TABLE_SIZE_ARRAY)
NUM_DENSE = 13

mllogger = MLLoggerWrapper(HCTRCommunicationHandler(), value=None)
mllogger.start(key=mllog_constants.INIT_START)

parser = argparse.ArgumentParser(description="HugeCTR DCN V2 model script.")
parser.add_argument(
    "--mode",
    help="benchmark_train_step, benchmark_inference, full_train",
    type=str,
    choices=["benchmark_train_step", "benchmark_inference", "full_train"],
    default="full_train",
)
parser.add_argument(
    "--optimizer",
    help="Optimizer to use",
    type=str,
    choices=["adagrad", "sgd"],
    default="adagrad",
)
parser.add_argument(
    "--batchsize",
    help="Batch size for training",
    type=int,
    default=8192,
)
parser.add_argument(
    "--batchsize_eval",
    help="Batch size for evaluation/inference",
    type=int,
    default=16384,
)
parser.add_argument(
    "--max_eval_batches",
    help="The number of evaluation batches to use",
    type=int,
    default=None,
)
parser.add_argument(
    "--lr",
    help="Learning rate",
    type=float,
    default=0.005,
)
parser.add_argument(
    "--eps",
    help="Epsilon value for Adagrad",
    type=float,
    default=1e-8,
)
parser.add_argument(
    "--init_accu",
    help="Initial accumulator value for Adagrad",
    type=float,
    default=0.0,
)
parser.add_argument(
    "--warmup_steps",
    help="Warmup steps for benchmark modes",
    type=int,
    default=50,
)
parser.add_argument(
    "--benchmark_steps",
    help="Number of steps to run for benchmark modes",
    type=int,
    default=200,
)
parser.add_argument(
    "--decay_start",
    help="Decay start",
    type=int,
    default=0,
)
parser.add_argument(
    "--decay_steps",
    help="Decay steps",
    type=int,
    default=0,
)
parser.add_argument(
    "--use_mixed_precision",
    action="store_true",
)
parser.add_argument(
    "--scaler",
    help="Loss scaling constant",
    type=float,
    default=1.0,
)
parser.add_argument(
    "--enable_tf32_compute",
    action="store_true",
)
parser.add_argument(
    "--disable_algorithm_search",
    help="Disables GEMM algorithm search for fully connected layers",
    dest="use_algorithm_search",
    action="store_false",
)
parser.add_argument(
    "--gen_loss_summary",
    help="Compute loss summary during training (loss = 0 if not set)",
    action="store_true",
)
parser.add_argument(
    "--max_iter",
    help="Number of training iterations to run for full_train mode",
    type=int,
    default=None,
)
parser.add_argument(
    "--display_interval",
    help="Display throughput stats every number of iterations",
    type=int,
    default=100,
)
parser.add_argument(
    "--eval_interval",
    help="Evaluate every number of iterations given",
    type=int,
    default=None,
)
parser.add_argument(
    "--auc_threshold",
    help="AUC threshold to reach to stop training",
    type=float,
    default=0.80275,
)
parser.add_argument(
    "--sharding_plan",
    help="Sharding plan to use",
    type=str,
    choices=["round_robin", "uniform", "auto", "hier_auto"],
    default="round_robin",
)

parser.add_argument(
    "--dp_sharding_threshold",
    help="threshold fo DP sharding in GiB.",
    type=float,
    default=0,
)

parser.add_argument(
    "--num_gpus_per_node",
    help="The numer of GPUs per node",
    type=int,
    default=8,
)
parser.add_argument(
    "--mem_comm_bw_ratio",
    help="The ratio between the communication and the memory bw of the system",
    type=float,
    default=3.35e12 / 450e9,
)
parser.add_argument(
    "--mem_comm_work_ratio",
    help="The ratio between the communication and the memory work of the network",
    type=float,
    default=8 / 2,
)
parser.add_argument(
    "--memory_cap_for_embedding",
    help="The amount of memory can be used for storing embedding in GB",
    type=float,
    default=60,
)
parser.add_argument(
    "--ev_size",
    help="The width of the embedding vector",
    type=int,
    default=128,
)
parser.add_argument(
    "--seed",
    help="The global seed for training.",
    type=int,
    default=0,
)
parser.add_argument(
    "--minimum_training_time",
    help="If set this vable bigger than 0, even hit the target AUC, training will continue until reach the minumum_training_time(minutes)",
    type=int,
    default=0,
)
parser.add_argument(
    "--train_data",
    help="Full path of training data file in container default is '/data/train_data.bin'",
    type=str,
    default="/data/train_data.bin",
)
parser.add_argument(
    "--val_data",
    help="Full path of evaluation data file in container, default is '/data_val/val_data.bin'",
    type=str,
    default="/data_val/val_data.bin",
)

args = parser.parse_args()
comm = MPI.COMM_WORLD
num_nodes = comm.Get_size()
rank = comm.Get_rank()
num_gpus = num_nodes * args.num_gpus_per_node
is_rank_zero = rank == 0



if args.mode == "full_train" and args.minimum_training_time > 0:
    args.max_iter = 1000000

logging.basicConfig(level=logging.INFO, format="%(message)s")

# Dependent parameters (if not set)
if args.mode == "full_train":
    iter_per_epoch = TRAIN_NUM_SAMPLES / args.batchsize
    if args.max_iter is None: args.max_iter = math.ceil(iter_per_epoch)
    if args.eval_interval is None: args.eval_interval = math.floor(0.05 * iter_per_epoch)
    if args.max_eval_batches is None: args.max_eval_batches = math.ceil(EVAL_NUM_SAMPLES / args.batchsize_eval)
    iter_per_epoch = math.ceil(iter_per_epoch)
elif args.mode == "benchmark_train_step":
    iter_per_epoch = 1
    args.max_eval_batches = 1
elif args.mode == "benchmark_inference":
    iter_per_epoch = 0
    args.max_eval_batches = 1

mllogger.mlperf_submission_log(mllog_constants.DLRM_DCNv2, num_nodes, "NVIDIA")
mlperf_logger.param_info(mllogger, args)

shard_matrix, shard_strategy = sharding.generate_plan(
    TABLE_SIZE_ARRAY, MULTI_HOT_SIZES, num_nodes, num_gpus, args, is_rank_zero
)

logging_callback = mlperf_logger.LoggingCallback(
    mllogger, args.auc_threshold, iter_per_epoch, args.batchsize)
if args.mode == "full_train":
    logging_callback.minimum_training_time = args.minimum_training_time

solver = hugectr.CreateSolver(
    model_name=mllog_constants.DLRM_DCNv2, seed=args.seed,
    max_eval_batches=args.max_eval_batches,
    batchsize_eval=args.batchsize_eval, batchsize=args.batchsize,
    vvgpu=[[x for x in range(args.num_gpus_per_node)] for _ in range(num_nodes)],
    repeat_dataset=True, lr=args.lr,
    warmup_steps=args.warmup_steps if args.mode == "full_train" else 0,
    decay_start=args.decay_start, decay_steps=args.decay_steps, decay_power=2.0, end_lr=0.0,
    use_mixed_precision=args.use_mixed_precision, enable_tf32_compute=args.enable_tf32_compute,
    scaler=args.scaler, use_cuda_graph=True,
    gen_loss_summary=args.gen_loss_summary if args.mode != "benchmark_inference" else False,
    train_intra_iteration_overlap=True, train_inter_iteration_overlap=True,
    eval_intra_iteration_overlap=False, eval_inter_iteration_overlap=True,
    all_reduce_algo=hugectr.AllReduceAlgo.NCCL, grouped_all_reduce=True,
    num_iterations_statistics=20, perf_logging=False, drop_incomplete_batch=True,
    use_embedding_collection=True, use_algorithm_search=args.use_algorithm_search,
    training_callbacks=[logging_callback] if args.mode == "full_train" else [],
)


optimizer = None
if args.optimizer == "adagrad":
    optimizer = hugectr.CreateOptimizer(optimizer_type=hugectr.Optimizer_t.AdaGrad, update_type=hugectr.Update_t.Global, initial_accu_value=args.init_accu, epsilon=args.eps)
elif args.optimizer == "sgd":
    optimizer = hugectr.CreateOptimizer(optimizer_type=hugectr.Optimizer_t.SGD, update_type=hugectr.Update_t.Local, atomic_update=True)

if args.mode == "benchmark_inference":
    reader_source = [args.val_data]; reader_num_samples = EVAL_NUM_SAMPLES
else: # 'benchmark_train_step' or 'full_train'
    reader_source = [args.train_data]; reader_num_samples = TRAIN_NUM_SAMPLES

reader = hugectr.DataReaderParams(
    data_reader_type=hugectr.DataReaderType_t.RawAsync, source=reader_source, eval_source=args.val_data,
    check_type=hugectr.Check_t.Non, num_samples=reader_num_samples, eval_num_samples=EVAL_NUM_SAMPLES,
    cache_eval_data=1, slot_size_array=TABLE_SIZE_ARRAY,
    async_param=hugectr.AsyncParam(num_threads=1, num_batches_per_thread=16, shuffle=False, multi_hot_reader=True, is_dense_float=True),
)

model = hugectr.Model(solver, reader, optimizer)
model.add(hugectr.Input(label_dim=1, label_name="label", dense_dim=NUM_DENSE, dense_name="dense", data_reader_sparse_param_array=[hugectr.DataReaderSparseParam("data{}".format(i), MULTI_HOT_SIZES[i], True, 1) for i in range(NUM_TABLE)]))
embedding_table_list = []
for i in range(NUM_TABLE): embedding_table_list.append(hugectr.EmbeddingTableConfig(name=str(i), max_vocabulary_size=TABLE_SIZE_ARRAY[i], ev_size=args.ev_size))
comm_strategy = hugectr.CommunicationStrategy.Hierarchical if num_nodes > 1 and args.sharding_plan=='hier_auto' else hugectr.CommunicationStrategy.Uniform
ebc_config = hugectr.EmbeddingCollectionConfig(use_exclusive_keys=True, comm_strategy=comm_strategy)
ebc_config.embedding_lookup(table_config=embedding_table_list, bottom_name=["data{}".format(i) for i in range(NUM_TABLE)], top_name="sparse_embedding", combiner=["sum" for _ in range(NUM_TABLE)])
ebc_config.shard(shard_matrix=shard_matrix, shard_strategy=shard_strategy)
model.add(ebc_config)
compute_config = hugectr.DenseLayerComputeConfig(async_wgrad=True, fuse_wb=False)
model.add(hugectr.DenseLayer(layer_type=hugectr.Layer_t.MLP, bottom_names=["dense"], top_names=["mlp1"], num_outputs=[512, 256, 128], act_type=hugectr.Activation_t.Relu, compute_config=compute_config))
model.add(hugectr.DenseLayer(layer_type=hugectr.Layer_t.Concat, bottom_names=["sparse_embedding", "mlp1"], top_names=["concat1"]))
model.add(hugectr.DenseLayer(layer_type=hugectr.Layer_t.MultiCross, bottom_names=["concat1"], top_names=["interaction1"], projection_dim=512, num_layers=3, compute_config=compute_config))
model.add(hugectr.DenseLayer(layer_type=hugectr.Layer_t.MLP, bottom_names=["interaction1"], top_names=["mlp2"], num_outputs=[1024, 1024, 512, 256, 1], activations=[hugectr.Activation_t.Relu, hugectr.Activation_t.Relu, hugectr.Activation_t.Relu, hugectr.Activation_t.Relu, hugectr.Activation_t.Non], compute_config=compute_config))
model.add(hugectr.DenseLayer(layer_type=hugectr.Layer_t.BinaryCrossEntropyLoss, bottom_names=["mlp2", "label"], top_names=["loss"]))

nvtx.range_push("Model Compile")
model.compile()
nvtx.range_pop()
nvtx.range_push("Model Summary")
model.summary()
nvtx.range_pop()

if args.mode == 'full_train':
    num_columns = 1 + NUM_DENSE + sum(MULTI_HOT_SIZES)
    mllogger.event(key=mllog_constants.TRAIN_SAMPLES, value=mlperf_logger.get_row_count(args.train_data, num_columns, 4), metadata={mllog_constants.EPOCH_NUM: 0.0})
    mllogger.event(key=mllog_constants.EVAL_SAMPLES, value=mlperf_logger.get_row_count(args.val_data, num_columns, 4), metadata={mllog_constants.EPOCH_NUM: 0.0})
    model.fit( max_iter=args.max_iter, display=args.display_interval, eval_interval=args.eval_interval, snapshot=2000000, snapshot_prefix="dlrm")
elif args.mode == 'benchmark_train_step':
    model.start_data_reading()
    nvtx.range_push("train warmup")
    if args.warmup_steps > 0:
        for i in range(args.warmup_steps): 
            nvtx.range_push(f"Train_Warmup_{i}")
            model.train()
            nvtx.range_pop()
    nvtx.range_pop()
    nvtx.range_push("DLRM_Train_Step_Benchmark")
    for i in range(args.benchmark_steps): 
        nvtx.range_push(f"Training_{i}")
        model.train()
        nvtx.range_pop()
    nvtx.range_pop()
elif args.mode == 'benchmark_inference':
    model.start_data_reading()
    nvtx.range_push("inference warmup")
    if args.warmup_steps > 0:
        for i in range(args.warmup_steps): 
            nvtx.range_push(f"Inference_Warmup_{i}")
            model.eval()
            nvtx.range_pop()
    nvtx.range_pop()
    nvtx.range_push("DLRM_Inference_Benchmark")
    for i in range(args.benchmark_steps): 
        nvtx.range_push(f"Inference_{i}")
        model.eval()
        nvtx.range_pop()
    nvtx.range_pop()

if is_rank_zero: print("DEBUG: Script finished"); sys.stdout.flush()
