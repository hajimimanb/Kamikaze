# 统一观测/决策记录契约（schema）v1

本文件是全体成员共享的数据契约。**任何偏离必须先在本文件修订并全队同步**。

## 1. 牌的编码

- kind: 0..33（0-8=万1-9, 9-17=筒1-9, 18-26=索1-9, 27-30=东南西北, 31-33=白發中）
- tile136: 0..135，tile136 = kind*4 + copy（copy 0..3 为同一牌种的4张）
- 赤宝牌固定：赤5万=16、赤5筒=52、赤5索=88（即 kind 4/13/22 的 copy 0）
- 转换工具函数（34计数数组、136数组、含赤标记）放在 `C:/agentwork/src/riichi/tiles.py`，先由 engine-developer 实现，data-engineer 与 ml-engineer 直接复用。

## 2. 每决策点记录（JSON，一行一条）

```json
{
  "game_id": "tenhou log id 或 majsoul uuid",
  "source": "tenhou|majsoul",
  "seat": 0,            // 当前决策者座位 0-3（0=东家/起家）
  "round": 0,           // 0..7: E1..E4, S1..S4（-1=四局终了后的西入等一律不采样）
  "honba": 0,           // 本场数
  "riichi_sticks": 0,   // 场上立直棒数
  "wall_left": 70,      // 牌山剩余张数（含王牌14张，即墙剩余摸牌数）
  "dora_indicators": [20],      // 已翻出的宝牌指示牌 tile136 列表（含杠宝）
  "scores": [25000, 25000, 25000, 25000],  // 四人当前点棒（按座位序）
  "oya": 0,             // 亲家座位
  "hand": [0,4,8,16,55,89, ...],   // 当前决策者手牌 tile136（摸牌后的决策点含刚摸的牌）
  "melds": [            // 四家副露，按座位序；每家为数组
    [],
    [{"type":"pon","tiles":[20,21,23],"from":0,"red":false}, ...],
    [], []
  ],
  "discards": [         // 四家牌河，按座位序；每家为数组（按时间序）
    [{"tile":100,"tsumogiri":true,"riichi":false}, ...],
    [], [], []
  ],
  "n_kan": 4,           // 场上总杠数（0-4）
  "riichi_declared": [false,false,false,false],  // 各家是否已立直
  "last_event": {"type":"discard","seat":1,"tile":50},  // 触发本决策的上一事件
  "legal_actions": {
    "discard": [16,17,18],        // 可切 tile136 列表
    "riichi": [],                 // 可立直打出的牌（非门清/非听为空）
    "chow": [["tiles":[...]], ...],   // 可吃组合（含所吃牌）
    "pon": [["tiles":[...]], ...],
    "kan": [["tiles":[...]], ...],    // 含明杠/加杠（暗杠作为自摸后可选动作之一）
    "ron": false,                 // 是否可荣和
    "tsumo": false                // 是否可自摸
  },
  "label": {"type":"discard","tile":16}  // 人类实际动作（SL 标签）: discard/riichi/chow/pon/kan/ron/tsumo/ryuukyoku
  // 若 type=riichi: {"type":"riichi","tile":16}；chow/pon/kan: {"type":"chow","tiles":[...]}
}
```

## 2.1 字段补充约定（2026-08-25 ml-engineer ↔ data-engineer 已确认，以此为准）
- `last_event`：**摸牌后切牌决策点必须为** `{"type":"draw"|"tsumo","seat":<当前决策者座位>,"tile":<刚摸的 tile136>}`；鸣牌/杠/和牌机会决策点为触发事件（discard/pon/kan/riichi 等）。训练端 tsumogiri（摸切）基线指标依赖 draw/tsumo 事件类型。
- `legal_actions.discard`：可切 tile136 **完整列表**（同种 4 张都在手时逐张列出）；特征编码器据此生成 34 维合法掩码。
- `hand` 在摸后决策点含刚摸的牌（14 张），与 §2 注释一致，实际输出必须如此。
- 输出路径：`data/processed/tenhou/records.jsonl`（zstd/分片可；分片时用约定 `data/processed/tenhou/records-*.jsonl` 通配）。ml-engineer 的张量化分片落 `data/processed/tenhou/shards/`（见 §5.1）。
- 和牌/流局等无弃牌 label 的样本（ron/tsumo/ryuukyoku）：ml-engineer 端弃牌头不对其监督（仅各二元头在人类有该选择权时监督）。
- discard/riichi 标签带显式 `"tsumogiri": true/false`（2026-08-25 captain 增强）：
  discard 时 `label.tile == last_event.tile 且 last_event.type=="draw"` 为 true；
  riichi 声明时刻一律 false（手切）；鸣牌后的切牌为 false；立直后的强制摸切
  discard 正常判 draw 匹配。旧分片未回刷（可由 last_event 推导），新增日期即生效。

## 3. 采样规则（SL 训练）
- 只采样人类**拥有真实选择权**的决策点：切牌（摸切或手切，含立直判定时刻）、可鸣牌/杠时、可和牌时（和牌样本单独加权，见下）。
- 被其他玩家抢胡/抢杠而跳过的点**不采样**（无选择）。
- 流局不产生 label 样本，但整局用于统计。
- 和牌样本（ron/tsumo）按 1:1 加入（即不降权），立直样本保留（其弃牌标签为打出的牌 + 立直标记）。
- 明暗杠：杠后摸牌决策点正常采样。

## 4. 可解释性要素（决策记录附带，由 engine-developer 实现计算函数供三队复用）
每条记录附 `"explain": {...}` 可选块（初始为空，函数在 `src/riichi/explain.py`）：
- shanten（向听数）、ukeire（有效进张列表及张数，含可见牌扣除）、
- 对每家牌河的现物/筋/壁安全度（危险牌集合），
- 手牌役种期望（役种/宝牌/赤宝/门清加成粗略值）。

## 5. 输出目录约定
- 天凤原始 XML: `C:/agentwork/data/raw/mjlog/{yyyy}/{logid}.xml`（gzip 存储可自行决定）
- 天凤决策集: `C:/agentwork/data/processed/tenhou/records.jsonl`（zstd/分片均可）
- 雀魂: `C:/agentwork/data/raw/majsoul/...` 与 `C:/agentwork/data/processed/majsoul/...`
- 环境与模型代码: `C:/agentwork/src/riichi/`（共享）、`C:/agentwork/src/env/`、`C:/agentwork/src/model/`
- 测试: `C:/agentwork/tests/`（pytest）

## 5.1 张量化训练分片（ml-engineer 产出，train_sl.py 消费）
- 由 `src/model/preprocess.py` 从 records.jsonl(.gz) 多进程离线生成，zstd 压缩落 `data/processed/tenhou/shards/`。输入支持 gzip 分片（data-engineer 的 records-*.jsonl.gz 直接吃）。
- **训练分区强制**：`--game-ids-file data/processed/tenhou/splits/train_games.txt` 白名单过滤；eval_holdout（log id 日期>=20260801）永不进训练分片。
- 每分片 `shard-{seq:06d}.npz.zst`：npz 内含 uint8 特征 `feat` (N,126,34)（=features.py mode="full" 通道序，通道清单/语义/量化 scale 见 `docs/channel_layout.json`）、逐通道缩放 `scale` (126,)（=max/255）、弃牌掩码 `dmask` (N,34)、`discard` (N,)、`drawn` (N,)、各头 `{head}_y`/`{head}_cand` (N,)（255=该头在此样本不可用/无 label）。
- 解码：x = feat_u8 * scale（float32）；mode="simple" = feat 前 17 通道（simple ⊂ full 且通道序一致）。
- 训练端按分片随机序加载 + 片内置换洗牌 + 背景预取；实测吞吐记录在 logs/preprocess_*.log。

## 6. 已知天凤端点（data-engineer 直接可用）
- 文件索引: https://tenhou.net/sc/raw/list.cgi （近7天）与 list.cgi?old （历史，含 `2025/...` 年份前缀路径）
- 凤凰桌对局列表: https://tenhou.net/sc/raw/dat/scc{yyyymmdd}{hh}.html.gz（内容含 `log={id}` 链接；四凤南=大厅码 00a9，三凤南=00b9）
- 牌谱 XML: https://tenhou.net/0/log/?{logid}（返回 mjlog XML）
- 大厅码解析: log id 第 13-17 位 hex，bit0x08=半庄（否则东风战），bit0x10=三人麻；凤凰四麻南=00a9
- 凤凰四麻南日产量 ~430 局（2026-08-24 实测）；可回溯至 2009 年

## 7. 已知雀魂端点（majsoul-researcher）
- 数据镜像: https://5-data.amae-koromo.com/ （另 1-data、4-data），路径 `api/v2/pl4/...`，需 `authorization: Bearer <cap token>`
- 模式 id: 王座=16、玉=12、王东=15（`games/{end_ms}/{start_ms}?limit=100&descending=true&mode=16`）
- CAP: POST https://akcap.pikapika.me/14f343ec68/challenge -> 解 80 个 sha256-pow（种子=token+i, i从1起；target 为 hex 前缀）-> POST .../redeem {token, solutions:[nonce...], instr}；instrumentation 为 deflate-raw 压缩脚本，需在沙箱 iframe 中执行并把结果作为 instr 回传
- 官方牌谱页: https://game.maj-soul.com/1/?paipu={uuid}（HTML 内嵌 JSON）
- 工作区已有半成品: `C:/agentwork/src/cap_fetch.py`（可读可改，整体重写优先）


## 7.5. MJAI 协议输出要求（任务书强制，v1.1 追加）
- 天凤与雀魂两路管道除 records.jsonl 外，还必须输出 MJAI 事件流：每局一个 JSONL（如 data/processed/{source}/mjai/{game_id}.jsonl）。
- 事件类型参照 mjai-simulator 协议：start_game / start_kyoku / tsumo / dahai / chi / pon / daiminkan / ankan / kakan / riichi / hora / ryukyoku / end_kyoku / end_game；牌用 MJAI 记法（1-9m/p/s, 1-7z；赤5 记 5mr/5pr/5sr；未知/宝牌指示牌按协议用 ? 或实际牌）。
- pip 镜像有 mjai==0.2.1 可安装用于校验协议合法性（能 import 并解析即可）。
- records.jsonl 由同一事件流派生，二者必须一致（reviewer 会交叉核对）。
- ml-engineer: features.py 增加 mode="simple"（仅手牌+宝牌+基础通道，课程学习第一阶段摸切模型用）与 mode="full" 开关。

## 8. 环境与工具约束（全员必读）
- 所有工具（bash/write/read/edit/glob/grep/web_search 等）只能在 `run_code` 程序内调用
- Python 一律用 venv: `C:/agentwork/.venv/Scripts/python.exe`；跑 python 前 `export PYTHONIOENCODING=utf-8`
- pip 用清华镜像（venv 的 pip.ini 已配置）；长任务必须后台运行写日志（bash 调用有超时，不要前台跑长命令）
- 覆盖写文件前先 read 该文件（文件工具要求）；JS/Python 模板串里 `\n` 转义注意
- 新装依赖写入 `C:/agentwork/requirements.txt` 追加