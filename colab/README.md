# Colab 層：VM 端套件

Bot 的 Colab 層在一台 Colab A100（40 GB）上跑 Ollama：`qwen3.8:27b-mtp-q4_K_M` 負責對話、視覺與思考，`clef-flash:9b`（Cloudflare 的決策模型，從 Qwen3.5-9B 微調，Q8_0，`/v1/systemone`；2026-10-16 起取代 `nimble:9b`）負責路由。Ollama 版本由 bot 在每次開機時決定：GitHub 最新的正式版（至少 `versions.json` 的 `min_version`，0.40.4），查不到或不能用時用 `versions.json` 釘住的備援版本（0.40.5）。兩個模型都跟著 tag 走：每次開機拉 tag 目前指向的版本，digest 只記錄、不擋開機（見〈模型 tag 與 digest〉）。這個目錄是 **VM 上跑的程式**；bot 端（控制器、launcher、註冊、請求路徑）在 `src/services/colab/`。設計理由、條款風險與成本保護見 [ADR-0010](../docs/adr/0010-colab-ollama-tier.md)。

| 檔案 | 用途 |
| --- | --- |
| `bootstrap.py` | 由 launcher 以獨立 process 啟動：裝好 Ollama、拉模型、啟動 auth proxy 與 cloudflared，最後送出簽章過的 `ready`。每一步寫進 `/content/state.json`。`--stub-upstream` 執行 DRYRUN 用的假 Ollama |
| `auth_proxy.py` | `127.0.0.1:8081` 的 bearer token 反向代理（tunnel → proxy → Ollama），內含 watchdog |
| `register.py` | bot 與 VM 必須算得一模一樣的部分：金鑰推導、簽章、事件內容、env 檔格式、readiness probe 檢查。只用標準函式庫 |
| `versions.json` | Ollama 的備援版本（下載網址、sha256、大小）與最低版本 `min_version`、cloudflared 的下載網址、sha256 與大小、兩個模型（角色、最後驗證過的 digest `verified_digest`）、GPU 下限。launcher 上傳的是它的副本，`ollama` 一項換成這次挑到的版本 |
| `proxy-requirements.lock` | auth proxy 的 aiohttp 與它全部的依賴，每個檔案都有 sha256（由 `proxy-requirements.in` 以 uv 產生，版本與 bot 映像的 `uv.lock` 相同、`uv.lock` 列出的每個檔案 hash 都在其中，測試會檢查；重新鎖定的指令見 [DEPLOY.md](../DEPLOY.md)〈依賴鎖定與本機驗證〉） |

`colab.VM_FILES` 列出 launcher 要上傳到 `/content` 的檔案。上傳後它們以同層模組互相 import（`python /content/bootstrap.py`）；在 repo 裡則是 `colab` 套件，所以 bot 與測試可以直接 import，而且 **不需要 aiohttp**（`auth_proxy` 只在真的開始服務時才載入 aiohttp）。測試：`tests/test_colab_vm_scripts.py`。

## 端到端流程

1. **需求**：有人觸發 bot（已通過限流與擋訊息檢查），而 Colab 沒開（`MODEL_PATH_PREFERENCE=colab`），或管理員下 `!colab on`。在 Colab READY 之前，**每則訊息都立刻由 NVIDIA → OpenRouter 回覆，不會等 Colab**。
2. **開機**（每分鐘的 cron 控制器）：CU 餘額扣掉「一次沒人用的 session」（開機逾時加一次閒置，A100 約 2.7 CU）仍 ≥ `COLAB_MIN_CU_BALANCE`（10），且今天用量加上它仍不超過 `COLAB_DAILY_CU_CAP`（30 CU）時，先把新的 `nbh`（UUID）寫進 Firestore，再 assign 一台 A100（**絕不用 high-mem**）。
3. **launcher**：先決定這次的 Ollama：問 GitHub `releases/latest`（不是草稿也不是 pre-release），`ollama-linux-amd64.tar.zst` 有 GitHub 下載網址、`sha256:` digest 與大小，而且版本 ≥ `min_version` 就用它（INFO `Colab launch: Ollama <版本>, the latest release`）；HTTP 錯誤、10 秒沒回應、被限流、沒有 digest 或格式不對就用備援（WARNING 寫原因）。結果依 session（`nbh`）記在 bot 的程序內：同一個 session 重試 launch 一定上傳同一個版本（否則重試挑到別的版本時，已在跑的 bootstrap 會被 probe 拒絕），每個新 session 都重新問一次 GitHub（GitHub 曾在發佈一天後重建 0.40.5 的 asset，舊的 sha256 最多只影響一個 session）；也記在 session 狀態（`ollama_version`）。接著用 Contents API 上傳 `VM_FILES`（`versions.json` 的 `ollama` 換成挑到的版本）；用 `execute(silent=True, store_history=False)` 寫出 `/content/.bootstrap.env`（0600）；再以 `Popen(start_new_session=True)` 啟動 `bootstrap.py`，確認 `BOOTSTRAP_LAUNCHED` 標記。**密鑰絕不經過 `colab exec --env`**（它會把值寫進 history）。
4. **bootstrap**（約 4–8 分鐘），`state.json` 的 `step` 依序為：

   | step | 做什麼 |
   | --- | --- |
   | `env` | 讀取並**立刻刪除** env 檔，取得 `/content/bootstrap.lock`（auth proxy 會繼承這個鎖，同一台 VM 上第二個 bootstrap 會安靜結束） |
   | `python` | 執行環境缺少或版本不同時，依 `proxy-requirements.lock` 安裝 proxy 的套件：`pip install --require-hashes --no-deps --only-binary=:all:`（每個 wheel 都比對 sha256、開機時不解析依賴、不從原始碼建置）。proxy 持有這個 session 的金鑰，所以它執行的程式和 Ollama、cloudflared 一樣都要驗證 |
   | `zstd` | 沒有 zstd 就 `apt-get install -y zstd`（失敗時先 `apt-get update` 再試一次） |
   | `gpu` | `nvidia-smi`：VRAM ≥ 38000 MiB、驅動 ≥ 550（用數字判斷，不比對字串 "80"） |
   | `ollama_install` | 下載 `versions.json` 指定的 Ollama tarball（launcher 挑的版本）、驗證大小與 sha256（比 `size` 長就立刻中止、不重試，不會塞滿磁碟；連線提早結束當成網路錯誤重試；sha256 不符就失敗：被竄改或沒下載完都過不了）、`zstd -dc \| tar -x -C /usr` |
   | `ollama_start` | `ollama serve`（`OLLAMA_HOST=127.0.0.1:11434`、`KEEP_ALIVE=-1`、`CONTEXT_LENGTH=32768`、`MAX_LOADED_MODELS=2`、`NUM_PARALLEL=1`、`FLASH_ATTENTION=1`、`NO_CLOUD=1`、`MAX_QUEUE=8`、`LOAD_TIMEOUT=10m`，`OLLAMA_ORIGINS` 不設），等 `/api/version` 回 `versions.json` 的版本 |
   | `pull` | 先 qwen3.8 再 clef-flash 拉兩個 tag（各最多 3 次，間隔 15、45 秒），再記下 `/api/tags` 的 digest（INFO，和 `verified_digest` 不同時寫 `tag moved since it was verified`，不會失敗）：qwen3.8 拉不到就失敗；clef-flash 拉不到只記一筆 ERROR 並繼續開機（原因放進 `ready` 的 `error`） |
   | `warmup` | **一定先** `POST /api/chat`（空訊息、`keep_alive:-1`，要回 `done_reason:"load"`）載入 qwen3.8，**再**用一次很小的 `/v1/systemone` 載入 clef-flash（順序見〈VRAM〉）；接著記錄 VRAM：`/api/ps` 每個模型的 `size_vram`／`size` 與 `nvidia-smi` 的 `memory.used`，寫進 `state.json` 的 `vram` 與 `ready` 的 `gpu`。clef-flash 載入失敗或沒有完全在 GPU 上時只記一筆 ERROR 並繼續開機；載入 clef-flash 時若 Ollama 踢掉了 qwen3.8，就卸載 clef-flash、單獨重新載入 qwen3.8，不用 clef-flash 繼續開機（見〈VRAM〉）。原因都放進 `ready` 的 `error` |
   | `proxy` | 啟動 `auth_proxy.py`，確認帶 token 回 200、不帶 token 回 401 |
   | `probe` | 透過 proxy 跑一次和 bot 相同的 readiness probe（`register.probe_problems`，qwen3.8 必須完全在 VRAM） |
   | `tunnel` | 下載 cloudflared 2026.9.3 並驗證大小與 sha256；移開 `~/.cloudflared/config.yaml`；quick tunnel 從 metrics 的 `/quicktunnel` 讀網址，named tunnel 用 `TUNNEL_TOKEN`（只放環境變數，不進 argv）。90 秒沒連上就改 `--protocol http2` 再試一次 |
   | `tunnel_check` | 從 VM 經公開網址呼叫一次 `/api/version`（失敗只記 log，仍會註冊） |
   | `register` | 送出簽章過的 `ready`（`tunnel_url`、版本、實際的 digest、GPU 與 VRAM，路由模型缺席時加上 `error` 寫原因）；5xx 與 422（bot 暫時解析不到新 tunnel 的網域）會重送，最多共送 4 次（間隔 5、10、20 秒，每次重新簽章），其他 4xx 直接失敗 |
   | `ready` | 完成；bootstrap 結束，Ollama、proxy、cloudflared 繼續執行 |

   **任何一步失敗**：`state.json` 寫入錯誤 → 送出簽章過的 `failed`（含 step 與錯誤）→ 停掉這次啟動的 process → VM 自我釋放（`POST http://$TBE_RUNTIME_ADDR/unassign`）→ 結束。env 檔無效時沒有可信的金鑰可簽章，只寫 `state.json` 並自我釋放。
5. **READY**：bot 驗證簽章與 `nbh`、檢查 `tunnel_url`（只接受 `https://<子網域>.trycloudflare.com` 或 `COLAB_TUNNEL_HOST`，並拒絕解析到非公開 IP 的網域），再帶 bearer token 經 tunnel 跑 readiness probe（10 秒內）：`/api/version` 至少是 `min_version`，而且就是 launcher 這次挑的版本（2026-10-16 之前開機、沒記錄版本的 session 只看最低版本）、`/api/ps` 列出 qwen3.8 且完全在 VRAM（`size_vram == size`，否則 probe 失敗）、`/api/tags` 列出 qwen3.8（digest 不比對）、qwen3.8 的 `/api/show` think 值是 `[false, "low", "medium", "xhigh"]`（新的 Ollama 若改了思考等級，這一項會擋下）。通過才 READY，INFO 記下實際的 Ollama 版本與每個模型的 digest（tag 動過會寫 `tag moved since it was verified`）。clef-flash 沒拉到、沒載入或沒有完全在 VRAM 時仍會 READY，但這個 session 不用它路由，並記一筆 ERROR `Colab session READY without its router: …`，寫出 probe 看到的情況（沒拉到、沒載入，或 VRAM 幾 MiB）與 VM 在 `ready` 的 `error` 回報的原因（例如拉取時的 HTTP 412）。之後每分鐘的健康檢查同樣要求這個 Ollama 版本與 qwen3.8 已載入；發現 clef-flash 被踢出 VRAM 或沒有完全在 VRAM 時，只讓它在這個 session 剩下的時間停用（WARNING `health check found clef-flash:9b not loaded`，接著第一則要路由的訊息記一筆 ERROR `clef-flash:9b off for this Colab session: …`，兩者都只代表 session 中途被踢；READY 時就缺席的不會再記 WARNING 或 ERROR，路由層只記一行 INFO `(reported at READY)`），不會把整個 Colab 層標成不健康。
6. **使用中**：每個 Colab 模型各一個 `asyncio.Semaphore(1)`，模型忙碌就直接跳過 Colab 層（Ollama 對已載入的忙碌模型不會回 503，只會排隊）。控制器在需求還新鮮時每 5 分鐘在同一個 kernel 跑一次 heartbeat cell（印出 `state.json` 前 300 字）：Colab 只把 kernel 活動算成使用中，tunnel 流量不算，約 22–25 分鐘沒有 kernel 活動就會被回收。
7. **釋放**：閒置 `COLAB_IDLE_TTL_SECONDS`（900）或滿 `COLAB_MAX_SESSION_SECONDS`（39600）時由控制器 unassign。VM 裡的 watchdog 是第二道保險（見下）。

## 協定摘要

完整定義在 `register.py` 的 docstring，bot 端的對應是 `src/services/colab/protocol.py`；`tests/test_colab_vm_scripts.py` 會交叉檢查兩邊的金鑰、簽章、模型清單、最低 Ollama 版本與上限完全一致，並讓 bot 的 `ColabRegistrar` 與 readiness probe 實際處理 VM 送出的事件（經過真正的 auth proxy 與 DRYRUN stub）。

- **金鑰**（Firestore 不存任何密鑰，bot 隨時可重算）：`reg_key = hex(HMAC_SHA256(COLAB_SHARED_SECRET, "reg:" + nbh))`；`proxy_token = base64url(HMAC_SHA256(COLAB_SHARED_SECRET, "proxy:" + nbh))`，去掉 `=`。
- **註冊**：`POST {BOT_URL}/internal/colab/register`，JSON ≤ 8 KB：`{event: ready|failed|stopping, nbh, tunnel_url?, versions?, digests?, gpu?, step?, error?}`（`ready` 帶 `error` 時是路由模型缺席的原因，bot 截短後記在 ERROR log）；header `X-Colab-Signature: t=<unix 秒>,v1=<hex HMAC_SHA256(reg_key, f"{t}.{原始 body}")>`（`reg_key` 以其 hex 文字當 key）。bot 拒絕時間差 > 300 秒、`nbh` 不符、簽章錯誤、重放與過大的 body。
- **env 檔** `/content/.bootstrap.env`：每行 `KEY=VALUE`，不加引號；key 有 `COLAB_REG_KEY`、`COLAB_PROXY_TOKEN`、`BOT_URL`、`NBH`、`IDLE_TTL`、`MAX_SESSION`、`TUNNEL_MODE`（quick｜named）、`TUNNEL_TOKEN`（named）、`TUNNEL_HOST`（named 必填，等於 `COLAB_TUNNEL_HOST`：named tunnel 無法自己回報網址，而 bot 只接受帶 `tunnel_url` 的 `ready`）、`OPENCODE_ENABLED`、`DRYRUN`。`register.render_bootstrap_env` / `parse_bootstrap_env` 是寫與讀的標準實作。
- **`/content/state.json`**：`{step, ts, error}`，warm-up 之後再加 `vram`（`chat_on_gpu`／`router_on_gpu`：`/api/ps` 的 `size_vram/size`，單位 MiB；`used_mib`：`nvidia-smi` 的 `memory.used`），絕不含密鑰；錯誤截到 200 字，整份不超過 heartbeat 印出的 300 字（放不下時錯誤再截短）。`ready` 事件的 `gpu` 也帶同樣三個欄位，bot 只拿來顯示（`/health/detail` 的 `colab.gpu`），判斷一律用自己 probe 看到的數字。
- **auth proxy**：`hmac.compare_digest` 比對 bearer（錯誤回 401）；白名單 `GET/HEAD /api/version`、`GET /api/ps`、`GET /api/tags`、`POST /api/show`、`POST /api/chat`、`POST /v1/systemone`，`OPENCODE_ENABLED=1` 時再加 `POST /v1/chat/completions`、`GET /v1/models`，其餘（pull、push、delete、create、copy、blobs…）一律 404；轉送時移除 Authorization、Origin、Cookie 與 hop-by-hop header，`Host` 改成 `127.0.0.1:11434`；串流直接轉送；body 上限 16 MB，上游 150 秒逾時。**放棄的請求會在 VM 上真的停下來**：bot 取消請求（鏈的時間上限、回合期限、關機）時會關掉連線，cloudflared 隨即關掉它到 proxy 的連線；`serve()` 以 aiohttp 的 `handler_cancellation` 執行，handler 因此被取消、關掉通往 Ollama 的連線，Ollama 就停止生成，`auth_proxy.log` 記一行 `POST /api/chat cancelled after N s: the client went away`。VM 的檔案是開機時才上傳，所以 proxy 的改動要到下一次 Colab 開機才生效。**只有** `/api/chat`、`/v1/systemone`、`/v1/chat/completions` 會更新閒置計時，health 與 probe 不算。
- **watchdog**（proxy 內，每 30 秒）：閒置超過 `IDLE_TTL + 120` 秒、存活超過 `MAX_SESSION + 120` 秒，或 Ollama 重啟一次後仍然掛掉 → `state.json` 寫 `stopping` → 簽章過的 `stopping` 事件 → `POST http://$TBE_RUNTIME_ADDR/unassign`（預期 200）→ 結束。任何一步出錯都不會讓 watchdog 停擺：單次檢查出錯只記 log、下一輪照跑；連續 3 次出錯、釋放流程本身出錯（例如磁碟滿了寫不了 `state.json`），或 watchdog 的 task 整個掛掉，都直接送出上面那個 unassign POST（不寫 `state.json`、不送事件）然後結束。**不呼叫** `google.colab.runtime.unassign()`：它在非 kernel 的 process 裡送出 POST 後會在 `eval_js` 失敗。

## 設定與操作

第一次設定（專用帳號、產生 token、`.env`、部署、CPU 乾跑、第一次 A100、對所有人開啟）、重新產生 token 與排查的順序在 [DEPLOY.md](../DEPLOY.md)〈Colab 層〉；設計理由、條款風險與成本保護在 [ADR-0010](../docs/adr/0010-colab-ollama-tier.md)。和 VM 端有關的只有兩件事：

- **DRYRUN**（`COLAB_DRY_RUN=true`）：控制器改申請 CPU runtime，VM 以 `DRYRUN=1` 走完整個控制流程（assign、上傳、silent env、launcher 標記、register、heartbeat、watchdog 自我釋放、孤兒對帳）。它跳過 GPU、Ollama 與模型，改啟動假的上游（`bootstrap.py --stub-upstream`，log 在 `stub_upstream.log`）：`/api/version`、`/api/ps`、`/api/tags`、`/api/show` 回傳能通過 readiness probe 的固定內容，推論路徑一律回 Ollama 的忙碌 503，所以這段期間真的使用者訊息會立刻改走 API 層，不會收到假回覆。
- **named tunnel**：Cloudflare 上建立的 tunnel 要把公開網址指到 `http://127.0.0.1:8081`（auth proxy）；launcher 以 `TUNNEL_MODE=named`、`TUNNEL_TOKEN`、`TUNNEL_HOST` 傳給 VM（見〈協定摘要〉的 env 檔）。

## 成本

- A100 40 GB 約 **5.37 CU/小時**（第三方數字；Google 未公布，以 assign 後 `ccu-info` 的 `consumptionRateHourly` 為準，控制器會記錄在狀態文件上）。
- **一次啟動 ≈ 2–2.7 CU**：開機 4–8 分鐘加閒置 15 分鐘，約 0.35–0.5 小時。計費從 assign 開始，不是模型載入完成才開始。
- **絕不用 high-mem**：`--high-mem` 會拿到 80 GB 的 A100，約 6.77 CU/小時。
- 保護措施：CU 餘額低於 10 不開機、每日上限 30 CU、閒置 15 分鐘或滿 11 小時釋放、watchdog 後援、任何啟動失敗都自我釋放、`!colab off` 與 `COLAB_ENABLED=false` 可一鍵關閉。DRYRUN 的 CPU runtime 也會用掉少量 CU。

## 疑難排解

先看 `!colab status`（狀態、最後錯誤、冷卻時間、CU 餘額與速率）。VM 還在時，heartbeat 會印出 `state.json`；VM 上的 log 在 `/content/bootstrap.log`、`ollama.log`（DRYRUN 是 `stub_upstream.log`）、`auth_proxy.log`、`cloudflared.log`。

| 現象 | 原因 | 處理 |
| --- | --- | --- |
| assign 回 **412** | `TooManyAssignmentsError`：同時開太多 runtime，或暫時的用量／容量限制 | 自動冷卻 5→10→20→60 分鐘。檢查專用帳號上是否有瀏覽器開著的 runtime（`colab sessions`） |
| assign 回 **503** | 暫時沒有 A100 | 自動冷卻後再試；常見於尖峰時段 |
| assign 回 **400** | 方案沒有 A100 權限或額度 | DISABLED。確認 Colab 方案與 CU 餘額 |
| assign 回 **401／403**、訊息含 **DENYLISTED** 或 abusive activity | token 權限錯誤，或帳號因疑似濫用被封鎖 | DISABLED。DENYLISTED 時先停用 Colab 層（`COLAB_ENABLED=false`），不要換帳號硬闖 |
| `invalid_grant`、`invalid_client`、`Reauthentication is needed` | refresh token 失效（`invalid_client`：憑證檔裡的 client 不對） | 照 [DEPLOY.md](../DEPLOY.md)〈Colab 層〉的〈重新產生 token〉重新產生 |
| `state.json` 停在 `gpu` | 拿到的不是 40 GB A100，或驅動 < 550 | VM 已自我釋放；若一直發生，回報並檢查 Colab 的 GPU 配置 |
| bot log 有 WARNING `Colab launch: Ollama 0.40.5, the pinned fallback (the latest release is not used: …)` | GitHub API 查不到、限流（未驗證的請求每小時 60 次）、最新版低於 `min_version`，或 asset 沒有 sha256 | 不影響使用：這個 session 用備援版本開機，下一個 session 再查。若一直是「older than」，表示 `min_version`（以及不低於它的備援版本）比 GitHub 的最新版還新，通常是 `versions.json` 填錯，檢查它 |
| `pull`：`pulling … failed 3 times: HTTP 412 …` | 模型的 config 要求更新的 Ollama（`requires`），registry 對舊版 Ollama 回 412（2026-10-15 nimble 就是這樣） | qwen3.8 拉不到時開機失敗並自我釋放；clef-flash 拉不到時照常 READY、不用它路由（bot 的 ERROR 會寫出這個原因）。把 `versions.json` 的 `min_version` 提高到那個 `requires`，必要時一併更新備援版本（見〈模型 tag 與 digest〉） |
| bot log 有 ERROR `Colab session READY without its router: clef-flash:9b …`，`/health/detail` 的 `colab.off_gpu` 或 `!colab status` 列出 `clef-flash:9b` | clef-flash 沒拉到（`not pulled`）、沒載入（`not loaded`：載入失敗，或開機時為了 qwen3.8 卸載了它）、或 Ollama／llama-server 只把它部分放在 GPU（`not wholly in VRAM`，附 MiB）；`the VM says:` 後面是 VM 自己的原因 | 不影響開機與 qwen3.8：這個 session 的路由改由規則與 API 路由決定。依原因處理：412 見上一列；VRAM 不夠照〈VRAM〉記下數字（`ollama.log` 的 `evicting`／`fits`）後處理 |
| bot log 有 WARNING `health check found clef-flash:9b not loaded`，接著第一則要路由的訊息記 ERROR `clef-flash:9b off for this Colab session: …` | session 中途 clef-flash 被踢出 VRAM（例如 qwen3.8 的 runner 當掉後重新載入） | 這個 session 剩下的時間不用它路由，qwen3.8 照常；下一個 session 會重新載入 |
| 健康檢查失敗：`Ollama X, expected Y (the release this session was launched with)` 或 `expected 0.40.4 or later` | VM 上跑的不是這個 session 開機時挑的 Ollama，通常是新版 bot 部署前就在跑的舊 session（例如 2026-10-16 前的 0.35.0） | 連續 3 次後排空釋放，下次開機會重新挑版本。部署前先 `!colab off` 可以避免 |
| `probe`：`qwen3.8… is not loaded` 或 `is not fully on the GPU` | qwen3.8 只有部分在 GPU，或卸載 clef-flash 後 qwen3.8 仍載入失敗 | 開機失敗並自我釋放。看 `state.json` 的 `vram` 與 `ollama.log`（`evicting`／`fits`），照〈VRAM〉處理 |
| `probe`：`think values are …` | 新的 Ollama 改了 qwen3.8 的思考等級 | 開機失敗並自我釋放（連續 3 次後 Colab 層停用）。確認新版的行為後更新 `versions.json` 的 `thinking` 與 `src/providers/model_profiles.py` 的 `THINK_LEVELS`。壞掉的是最新版時，**只提高 `min_version` 跳不過它**：`validate_versions` 要求備援版本不低於 `min_version`，備援又不可能比最新版新，`versions.json` 會驗證失敗（bot 也載入不了）。要等修好的新版發佈，再把 `min_version` 與備援版本（網址、sha256、`size`）一起提高到它；在那之前只能改程式 |
| `tunnel`：90 秒內沒連上 | Colab 擋 UDP（QUIC） | 已自動改 `--protocol http2` 重試；兩者都失敗時看 `cloudflared.log` |
| `register`：HTTP 403／409 | 這台 VM 已不是目前的 session（`nbh` 不符或狀態已變） | VM 會自我釋放，不需處理 |
| `register`：HTTP 503、`readiness probe failed` | bot 經 tunnel 探測失敗 | VM 會重試；持續失敗就看 `!colab status` 的 problems |
| Colab 回應 **524** | 非串流呼叫超過 Cloudflare 的 125 秒 | bot 自動改走 API 層；把 `COLAB_DEEP_THINK` 調低 |

## 模型 tag 與 digest

2026-10-16 起兩個模型都**跟著 tag 走**：每次開機拉 tag 目前指向的版本，digest 不擋任何東西。`versions.json` 的 `verified_digest` 與 `src/providers/model_profiles.py` 的 `verified_digest`（測試檢查兩邊一致）只記錄**最後一次有人手動確認過**的 digest；開機時 VM 與 bot 都會把實際的 digest 記在 INFO log，不同時寫 `tag moved since it was verified`，`/health/detail` 的 `colab.digests`／`colab.tags_moved` 與 `!colab status` 也會顯示。

取捨：以前釘 digest 時，tag 一被重新發佈 qwen3.8 就開不了機、nimble 就不能路由；而 2026-10-14 nimble 重新發佈成要求 Ollama 0.40.4 之後，釘住的版本連拉都拉不到（registry 對舊版 Ollama 回 412）。跟著 tag 走讓 session 照常運作，代價是可能跑到沒人測過的模型版本：它不會讓任何檢查失敗，只會反映在回答、路由與 log 的 digest 上，所以看到 `tag moved since it was verified` 時照下面確認一次。

- **只換 `MODEL_ROLES` 不行**：路由層的模型、`versions.json` 與 model profile 必須一起換。VM 沒拉的模型在 `/v1/systemone` 回 404，而 404 會把整個 Colab 層（包括 qwen3.8 對話）標成不健康，直到下一次健康檢查通過。
- **確認新的 digest**：以 tag 取回 manifest，它的 sha256 就是 Ollama 0.40.5 在 `/api/tags` 顯示的 digest（tag 是 manifest list 時，這個 Accept header 拿到的是 llamacpp 那個 child，正好是 `/api/tags` 顯示的值）；再看 layers 與 config 改了什麼（模型層、projector、params、template）：

  ```bash
  curl -s -H 'Accept: application/vnd.docker.distribution.manifest.v2+json' \
    https://registry.ollama.ai/v2/library/clef-flash/manifests/9b | tee /tmp/manifest.json | shasum -a 256
  python3 -m json.tool /tmp/manifest.json
  # config blob（application/vnd.docker.container.image.v1+json 那一層）：看 "requires"
  curl -sL https://registry.ollama.ai/v2/library/clef-flash/blobs/sha256:<config 的 digest>
  ```

  **一定要看 config 的 `requires`**：它比 `versions.json` 的 `min_version` 新時，舊版 Ollama 拉不到這個模型（registry 回 412；用 curl 拉會是 200，看不出來）。這時把 `min_version` 提高到 `requires`，備援版本也要至少是它（從 GitHub release 取 `ollama-linux-amd64.tar.zst` 的網址、`digest` 的 sha256 與 `size`，sha256 最好在釘版前再核對一次：GitHub 曾在發佈後一天重新上傳 0.40.5 的 asset）。小的 layer（params、config）都可用 `https://registry.ollama.ai/v2/library/<模型>/blobs/sha256:<digest>` 取回。確認後把新的 digest 寫進兩個檔案的 `verified_digest`、跑測試、部署；路由模型換了 checkpoint 時，也要用 `scripts/evals/routing_eval.py` 重新調整 `COLAB_ROUTER_*` 門檻。

## VRAM

A100 40 GB 上同時放 qwen3.8（32K context）與 clef-flash（Q8_0，模型自己的 `num_ctx` 16384，沒有另做小 context 的衍生模型），餘裕比 nimble 時更小，而且**沒有實測過**：

- Ollama 載入新模型時，若預估大小超過目前空閒 VRAM 的 80% 就會踢掉已載入的模型。它對 clef-flash 的預估約 **13.0 GiB**（GGUF 9,657,260,192 bytes、16384 的 KV cache 約 2 GiB，加上決策模型 batch 等於整個 context 的 2 GiB），所以 qwen3.8 載入後至少要剩 **約 16.2 GiB**（nimble 時是約 13.3 GiB）。預估沒算進 0.92 GB 的 vision projector、16 個序列的 recurrent state 與 16384 token 的 compute buffer。
- 所以 warm-up **一定先載入 qwen3.8 再載入 clef-flash**：反過來的話 qwen3.8（預估約 22.4 GiB）要在 clef-flash 已佔用時找到約 28 GiB 空閒，幾乎放不下。預估放不下時 Ollama 會踢掉閒置的 qwen3.8：bootstrap 會發現 `/api/ps` 沒有 qwen3.8，就卸載 clef-flash（`/api/generate`、`keep_alive: 0`）、單獨重新載入 qwen3.8，記一筆 ERROR 後不用 clef-flash 繼續開機。預估通過但實際記憶體不夠時，0.40.4 起的 llama-server 會把 clef-flash 部分放到 CPU：`/api/ps` 的 `size_vram < size`，下面的保護會讓它這個 session 不路由。
- session 中 qwen3.8 單獨被重新載入（例如它的 runner 當掉）而 clef-flash 還在時，同樣放不下，Ollama 會踢掉閒置的 clef-flash；健康檢查發現它不見了，只讓它在這個 session 剩下的時間停用（WARNING `health check found clef-flash:9b not loaded`），qwen3.8 對話照常，也不會再呼叫 clef-flash（重新載入它可能又踢掉 qwen3.8）。watchdog 重啟 Ollama 時一樣先載入 qwen3.8；只有 qwen3.8 沒回來才算重啟失敗。
- 保護措施：readiness probe 要求 qwen3.8 `size_vram == size`（否則開機失敗）；clef-flash 沒有完全在 VRAM 或沒有載入時 session 照樣 READY，但這個 session 不用它路由（READY 時一筆 ERROR 寫出 MiB 數字或原因，`/health/detail` 的 `colab.off_gpu`，只在 READY 時列出）。`OLLAMA_MAX_LOADED_MODELS` 維持 2，bot 從不送 `num_ctx`（送了會觸發重載；`/v1/systemone` 也沒有 options 可以改）。
- `COLAB_ROUTER_ENABLED=false` 只關掉 bot 這邊的路由層，**不會**讓 VM 少拉或少載 clef-flash（約 10.6 GB 下載與約 13 GiB VRAM）；有上面的保護，它不會影響 qwen3.8 對話。要讓 VM 不放它，只能把路由模型從 `versions.json` 與 model profile 一起拿掉。
- **首次實跑一定要量**：READY 後記下 `/health/detail` 的 `colab.gpu`（`chat_on_gpu`、`router_on_gpu`、`used_mib`）與 `colab.off_gpu`，或 heartbeat 印出的 `state.json` 的 `vram`，以及 `ollama.log` 的排程訊息（`fits`／`evicting`），把 qwen3.8 與 clef-flash 的實際大小（`/api/ps` 的 `size_vram`）、`memory.used` 與剩餘 VRAM 寫回這一節與 ADR-0010。clef-flash 放不下或只部分在 GPU 時（READY 的 ERROR 會寫），可以考慮把 qwen3.8 的 `OLLAMA_CONTEXT_LENGTH`（`bootstrap.py`，32768）調低，或另做小 context 的衍生模型（`/api/create` 加 `num_ctx`；它是否保留決策能力與 projector 還沒驗證）。

## 首次實跑時確認（verify on first run）

- **路由模型是 `clef-flash:9b`**（2026-10-16 你決定，理由與取捨見 ADR-0010 的 2026-10-16 修訂）：Cloudflare 的決策模型，從 Qwen3.5-9B 微調，Apache-2.0。tag 是 manifest list，Ollama 0.40.5 的 `/api/tags` 顯示它的 llamacpp child `73e026ff389b28464bb26f8978f3164851a76832f0d339032747e2c56f98ea29`（2026-10-16 核對）：Q8_0 GGUF 9,657,260,192 bytes、vision projector 921,704,928 bytes、params `{"num_ctx":16384}`、config `"requires":"0.40.4"`，沒有 SYSTEM layer（prompt 的模板在 GGUF 裡）。Ollama 0.40.4 起由 llama.cpp 回答 `/v1/systemone`：所有欄位一起放進一個 prompt（`STATE:` 加上 `FIELD`／`ID`／`TYPE`／`INSTRUCTION`／`ALLOWED OPTIONS`），choice 的信心值改成 (最高機率 − 1/N) / (1 − 1/N)。第一次 A100 實跑時確認：`scripts/live_gate.py --path colab` 的 `colab.router` 路由決定合理、`/api/ps` 的 `context_length` 是 16384、一次呼叫的延遲遠低於路由層的 6 秒、VRAM（見〈VRAM〉）；再用 `scripts/evals/routing_eval.py` 在繁中訊息上重新調整 `COLAB_ROUTER_CONFIDENCE`／`COLAB_ROUTER_THINK_ON`／`COLAB_ROUTER_THINK_OFF`（目前是暫定值 0.0／0.85／0.15；信心門檻 0.0 是一律採用機率最高的選項，因為實際訊息的信心值只有 0.39–0.43，0.90 讓每一則都交給 API router）。
- **Ollama**：bot 挑的版本記在 log（`Colab launch: Ollama …`）與 READY 那一行。備援 0.40.5 的 tarball sha256 已與 release asset 的 digest 核對（2026-10-16）；最新版的 sha256 直接用 GitHub 回報的 asset digest。確認 qwen3.8 的 MTP 與思考等級在新版仍正常（probe 會檢查 `/api/show`；MTP 的速度要看回答時間）。
- **cloudflared 2026.9.3**：sha256 已與 release asset 的 digest 核對。
- `qwen3.8:27b-mtp-q4_K_M` 的 digest 在 2026-10-16 以 0.40.5 的 client 重新核對仍是 `22130167…`（仍是單一 manifest，不是 list）。
- `/api/tags` 的 digest 格式（不含 `sha256:` 前綴；程式兩種都接受）。
- Colab A100 的 NVIDIA 驅動 ≥ 550、UDP／QUIC 是否可用（不行就走 HTTP/2）。
- `proxy-requirements.lock` 在 Colab 的 Python 版本上有可用的 wheel（鎖定檔涵蓋 Python 3.11 以上所有已發佈的檔案；缺 wheel 時 `python` 步驟會失敗，不會退回原始碼建置）。
- 兩個模型的 VRAM：照〈VRAM〉的首次實跑檢查記錄（qwen3.8 在 32K context 下的規劃值約 24–25 GiB、clef-flash 預估約 13 GiB，都還沒實測）。
- NDJSON 串流經 quick tunnel 是否即時；bot 放棄請求後，`auth_proxy.log` 是否出現 `cancelled after N s`（proxy 收到斷線後一定會中止 Ollama 的請求，測試以真正的 `serve()` 驗證；實跑要確認的是 Cloudflare 與 cloudflared 會把斷線傳到 proxy）。
- heartbeat 是否確實讓 runtime 不被回收。

## opencode 擴充

兩種做法，都需要 64K context：把 `bootstrap.py` 的 `OLLAMA_SERVE_ENV["OLLAMA_CONTEXT_LENGTH"]` 改成 `65536`。bot 從不在請求裡送 `num_ctx`，所以 qwen3.8 自然用這個值、不會觸發模型重載（clef-flash 有自己的 `num_ctx` 16384，不受影響）；但 qwen3.8 的 KV cache 會多約 2 GiB，依〈VRAM〉的預估 clef-flash 很可能就放不下了，先照首次實跑的數字確認。

1. **推薦：另開一個 session。** 用 `colab ssh --proxy-mode` 把 VM 的 11434 port-forward 到 Mac，opencode 設 `baseURL: http://localhost:11434/v1`。不需要公開網址，SSE 串流也正常；和 bot 的 session 完全分開，互不影響閒置計時與忙碌閘門。
2. **共用 bot 的 proxy。** 設 `OPENCODE_ENABLED=1`，proxy 會多開 `POST /v1/chat/completions` 與 `GET /v1/models`（一樣要 bearer token），opencode 的請求也算活動。quick tunnel 不支援 SSE，所以這個做法需要 named tunnel。注意：qwen3.8 一次只服務一個請求，opencode 使用中時 bot 的忙碌閘門會跳過 Colab 層；而 bot 控制器的閒置計時只看 LINE 的需求，opencode 的流量不會阻止控制器在 15 分鐘後釋放 session。
