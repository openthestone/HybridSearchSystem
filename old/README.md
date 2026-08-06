# npu-project-v1

昇腾 NPU 混合检索系统（向量 + 属性过滤）。

## 文档

- [运行说明](docs/run.md)
- [数据集说明](docs/dataset.md)

## 快速开始

```bash
cd /root/xihe_v1
./run.sh -v s -p 0 -c 1   # serial
./run.sh -v p -p 0 -c 1   # parallel
```

## 数据

大文件（`*.bin`、`datasets/`、`buckets/` 等）不入库，见 [docs/dataset.md](docs/dataset.md)。
