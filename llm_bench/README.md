# llm-bench

轻量级 LLM Benchmark Serve 工具：零 vLLM 依赖（仅 `aiohttp` + `numpy` + `matplotlib`），针对火山方舟 Ark 压测 **TTFT / TPOT / E2E / Cache 命中率**。

> 在 `ark-model-benchmark` Skill 中，请优先运行
> `bash ../scripts/run.sh --model "<模型名称>"`。该包装器会从 Ark CLI 自动注入
> 后付费 Profile、Base URL 和 API Key，无需手工设置下文的 `ARK_*` 环境变量。

## 安装

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 快速开始

```bash
source .venv/bin/activate   # 若尚未激活虚拟环境

export ARK_BASE_URL="https://ark.cn-beijing.volces.com/api/v3"
export ARK_API_KEY="..."
# 也兼容 OpenAI 命名：OPENAI_BASE_URL / OPENAI_API_KEY（未设置 ARK_* 时自动回退读取）

# 连通性自检（1 个请求，打印 TTFT / usage / cached_tokens）
python bench.py connectivity --model deepseek-v4-flash-ga-260731
```

自检通过后，可以直接用 run 脚本跑方舟服务配置下的两种压测模式：

```bash
./run_prefix.sh     # 模式 A：多用户共享同一批固定前缀
./run_multiturn.sh  # 模式 B：会话内逐轮累加历史
```

两种模式的详细语义见「模式说明」，可调参数见「压测参数」。

## 标准用法

按你的目标选一条路径即可，三条路径共用同一套工具和报告：

| 我想…… | 做法 |
|---|---|
| 只压方舟 | `./run_prefix.sh` / `./run_multiturn.sh` |
| 方舟 vs 第三方对比 | `PEER=<名字> ./run_prefix.sh`（详见下文「双侧压测」） |
| 只压第三方服务 | 覆盖主目标坐标后照常跑（详见下文「单独压第三方」） |

### 单侧压测（只压方舟）

```bash
./run_prefix.sh
./run_multiturn.sh

# 覆盖参数：环境变量前置即可（完整参数见「压测参数」）
MAX_CONCURRENCY=20 PREFIX_LEN=12000 SUFFIX_LEN=2000 NUM_REQUESTS=200 \
./run_prefix.sh

# multiturn：想让命中率曲线更平滑可提到 50 个 session（见「样本量与统计口径」）
MAX_CONCURRENCY=10 NUM_SESSIONS=50 MAX_TURNS=20 \
./run_multiturn.sh
```

### 双侧压测（方舟 vs 第三方，双边同发）

`prefix` / `multiturn` 均支持同步压测多个模型：同一份样本数据、同一时间窗内并发发起，指标口径完全一致，报告自动出对比表。

```bash
# 1. 准备对比目标配置：cp peers/_template.env peers/<名字>.env，按注释填空
#    （自带 aliyun.env / deepseek.env 两个样例，填上自己的 key 即可用）
ls peers/

# 2. PEER=<名字> 选定对比目标后启动
PEER=aliyun ./run_prefix.sh
PEER=aliyun ./run_multiturn.sh

# 覆盖参数写法相同
MAX_CONCURRENCY=20 PREFIX_LEN=12000 SUFFIX_LEN=2000 NUM_REQUESTS=200 \
PEER=aliyun ./run_prefix.sh
```

### 单独压测第三方服务

工具走 OpenAI 兼容 **Chat Completions** 协议（流式，`POST /chat/completions`），任何兼容该协议的服务都能压。做法很简单：**主目标坐标（`ARK_BASE_URL` / `ARK_API_KEY` / `MODEL`）指向第三方**，其余流程不变。

```bash
# 1. 先连通性自检（1 个请求，验证地址 / 密钥 / 模型 ID 可用）
python bench.py connectivity --model <第三方模型ID> \
  --base-url <第三方API地址> --api-key <第三方KEY>

# 2. 通过后正式压测
python bench.py prefix --model <第三方模型ID> \
  --base-url <第三方API地址> --api-key <第三方KEY>

# 3. 用 run 脚本同理：export 三个主目标变量后照常 ./run_prefix.sh
```

> 提示：第三方不认方舟的 `reasoning_effort=none` 时（自检会告警「思考未关闭」），可用 `--reasoning-effort ""` 置空、`--enable-thinking false` / `--thinking disabled` 等参数适配厂商差异，详见 `--help`。


## 压测参数

跑 run 脚本时用环境变量覆盖默认值（直接调 bench.py 则用对应 `--` 长参数，见 `--help`）：

| 环境变量 | 模式 | 对应参数 | 默认值 | 含义 |
|---|---|---|---|---|
| `MODEL` | 通用 | `--model` | `deepseek-v4-flash-ga-260731` | 方舟侧（主）模型 ID |
| `PREFIX_LEN` | prefix | `--prefix-len` | 12000 | 共享前缀长度（token，主变量） |
| `SUFFIX_LEN` | prefix | `--suffix-len` | 2000 | 每请求独立后缀长度（token） |
| `NUM_PREFIXES` | prefix | `--num-prefixes` | 10 | 前缀池个数（决定命中率上限） |
| `NUM_REQUESTS` | prefix | `--num-requests` | 200 | 总请求数（建议为 `NUM_PREFIXES` 整数倍） |
| `INITIAL_LEN` | multiturn | `--initial-len` | 3000 | Turn1 长输入 token 数 |
| `QUESTION_LEN` | multiturn | `--question-len` | 256 | Turn2+ 每轮短追问 token 数 |
| `NUM_SESSIONS` | multiturn | `--num-sessions` | 10 | 总会话数（样本量） |
| `MAX_TURNS` | multiturn | `--max-turns` | 20 | 每会话轮数 |
| `MAX_CONCURRENCY` | 通用 | `--max-concurrency` | 5 | 并发上限（prefix 单位是请求；multiturn 单位是 session） |
| `OUTPUT` | 通用 | `--output` | `reports/<mode>_<时间戳>/result_<mode>.json` | 结果 JSON 路径 |
| `PEER` / `COMPARE[_1..3]` | 通用 | `--compare` | 不设即单边 | 第三方对比目标（见「双侧压测」） |

## 模式说明

**multiturn**：session = 一条完整对话链，会话内多轮严格串行（并发单位是 session）。Turn1 塞入 `--initial-len` 长输入模拟文档/长上下文，Turn2+ 每轮只有 `--question-len` 的短追问，上下文由模型回答逐轮累积；Turn_i 的前缀 = Turn_{i−1} 的全部内容，命中率随轮次爬升，Turn1 冷启动未命中属预期。

**prefix**：不同请求复用同一批固定前缀（模拟多用户共享系统提示词），后缀各不相同。每前缀首次请求冷启动未命中属预期；全并发下前几个复用序号可能因 cache 未及时写入而低于稳态值，看命中率稳态请关注复用序号较大的曲线段。`--num-requests` 建议取 `--num-prefixes` 的整数倍。

## 输出与报告

每次压测产出独立目录 `reports/<mode>_<时间戳>/`，多次运行互不混杂：`report_<mode>.md`（Markdown 报告 + PNG 曲线）+ `result_<mode>.json`（逐请求原始数据）；控制台同步打印总览。实测样例（prefix 双侧对比报告开头的对比表）：

> 建议以 avg / median 为主要对比口径；P99 的解读需满足样本量条件，见「样本量与统计口径」。

| 模型 | 成功/失败 | TTFT avg (ms) | TTFT P99 (ms) | TPOT avg (ms) | E2E avg (ms) | 加权Cache命中率 |
|---|---|---|---|---|---|---|
| deepseek-v4-flash-ga-260731 | 200/0 | 2442.7 | 3663.8 | 9.5 | 5626.1 | 85.29% |
| DeepSeek官方 | 199/1 | 730.8 | 1616.0 | 10.5 | 4308.2 | 84.78% |

## 样本量与统计口径

统计样本是「一次生成」事件：**prefix 的样本数 = 总请求数，multiturn 的样本数 = 会话数 × 轮数**。默认参数下两者均为 200 个样本，专为**均值口径**设计：

- **avg / median**：200 个样本已足够稳定，默认规模直接可信，这是推荐客户关注的口径。
- **P95**：约 300~500 个样本起比较稳。
- **P99**：需要 1000+ 个样本；200 个样本下 P99 之后只剩约 2 个数据点，波动大，仅供参考。若确需解读 P99，把样本量提到 1000 以上（prefix：`NUM_REQUESTS=1000 NUM_PREFIXES=20`；multiturn：`NUM_SESSIONS=50`）。

另外，同参数多跑几次看跨 run 一致性，比单次加大样本量更能反映真实波动（每次 run 的 seed 已写入报告，可复现）。无论样本量多大，对比不同批次的结果时务必保持 `MAX_CONCURRENCY` 一致。

multiturn 的命中率曲线按轮序分层（每个轮序点由所有 session 平均而来），session 越少曲线越毛糙；想让曲线更平滑可提到 50 个 session，均值指标本身不受影响。

## 说明

- 语料来自本地 `data/sharegpt_pool.txt`（约 12MB / 263 万 tokens，随工具自带，完全离线、不联网下载）；压测规模超出池容量时自动循环拼接补足并告警。
- 默认关闭思维链，保证 TTFT/TPOT 口径干净
- 不做任何 warmup，保持冷启动 + 全并发真实形态；失败请求不重试、剔除时延统计、计入失败数（对齐 vLLM：429 也计为失败；例外：multiturn 会话内 429 指数退避重试最多 3 次）。
- 语料采样 seed 每轮随机生成并写入报告，可复现输入。
- 输出封顶（`--max-completion-tokens`）prefix 默认 512 / multiturn 默认 1024：multiturn 的回答会累积进上下文、驱动轮次增长，上限需要更宽松；prefix 各请求独立、输出即弃，512 足够且省配额。
