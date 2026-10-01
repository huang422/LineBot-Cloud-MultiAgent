# Colab 層：VM 端套件

Bot 的 Colab 層在一台 Colab A100（40 GB）上跑 Ollama 0.35.0：`qwen3.8:27b-mtp-q4_K_M` 負責對話、視覺與思考，`nimble:9b`（Bespoke Nimble 9B，Q8_0，`/v1/systemone`）負責路由。這個目錄是 **VM 上跑的程式**；bot 端（控制器、launcher、註冊、請求路徑）在 `src/services/colab/`。設計理由、條款風險與成本保護見 [ADR-0010](../docs/adr/0010-colab-ollama-tier.md)。

| 檔案 | 用途 |
| --- | --- |
| `bootstrap.py` | 由 launcher 以獨立 process 啟動：裝好 Ollama、拉模型、啟動 auth proxy 與 cloudflared，最後送出簽章過的 `ready`。每一步寫進 `/content/state.json`。`--stub-upstream` 執行 DRYRUN 用的假 Ollama |
| `auth_proxy.py` | `127.0.0.1:8081` 的 bearer token 反向代理（tunnel → proxy → Ollama），內含 watchdog |
| `register.py` | bot 與 VM 必須算得一模一樣的部分：金鑰推導、簽章、事件內容、env 檔格式、readiness probe 檢查。只用標準函式庫 |
| `versions.json` | 釘住的版本：Ollama 與 cloudflared 的下載網址與 sha256、兩個模型的 digest、GPU 下限 |
| `proxy-requirements.lock` | auth proxy 的 aiohttp 與它全部的依賴，每個檔案都有 sha256（由 `proxy-requirements.in` 以 uv 產生，版本與 hash 和 bot 映像的 `requirements.lock` 相同，測試會檢查） |

`colab.VM_FILES` 列出 launcher 要上傳到 `/content` 的檔案。上傳後它們以同層模組互相 import（`python /content/bootstrap.py`）；在 repo 裡則是 `colab` 套件，所以 bot 與測試可以直接 import，而且 **不需要 aiohttp**（`auth_proxy` 只在真的開始服務時才載入 aiohttp）。測試：`tests/test_colab_vm_scripts.py`。

## 端到端流程

1. **需求**：有人觸發 bot（已通過限流與擋訊息檢查），而 Colab 沒開（`MODEL_PATH_PREFERENCE=colab`），或管理員下 `!colab on`。在 Colab READY 之前，**每則訊息都立刻由 NVIDIA → OpenRouter 回覆，不會等 Colab**。
2. **開機**（每分鐘的 cron 控制器）：CU 餘額扣掉「一次沒人用的 session」（開機逾時加一次閒置，A100 約 2.7 CU）仍 ≥ `COLAB_MIN_CU_BALANCE`（10），且今天用量加上它仍不超過 `COLAB_DAILY_CU_CAP`（30 CU）時，先把新的 `nbh`（UUID）寫進 Firestore，再 assign 一台 A100（**絕不用 high-mem**）。
3. **launcher**：用 Contents API 上傳 `VM_FILES`；用 `execute(silent=True, store_history=False)` 寫出 `/content/.bootstrap.env`（0600）；再以 `Popen(start_new_session=True)` 啟動 `bootstrap.py`，確認 `BOOTSTRAP_LAUNCHED` 標記。**密鑰絕不經過 `colab exec --env`**（它會把值寫進 history）。
4. **bootstrap**（約 4–8 分鐘），`state.json` 的 `step` 依序為：

   | step | 做什麼 |
   | --- | --- |
   | `env` | 讀取並**立刻刪除** env 檔，取得 `/content/bootstrap.lock`（auth proxy 會繼承這個鎖，同一台 VM 上第二個 bootstrap 會安靜結束） |
   | `python` | 執行環境缺少或版本不同時，依 `proxy-requirements.lock` 安裝 proxy 的套件：`pip install --require-hashes --no-deps --only-binary=:all:`（每個 wheel 都比對 sha256、開機時不解析依賴、不從原始碼建置）。proxy 持有這個 session 的金鑰，所以它執行的程式和 Ollama、cloudflared 一樣都要驗證 |
   | `zstd` | 沒有 zstd 就 `apt-get install -y zstd`（失敗時先 `apt-get update` 再試一次） |
   | `gpu` | `nvidia-smi`：VRAM ≥ 38000 MiB、驅動 ≥ 550（用數字判斷，不比對字串 "80"） |
   | `ollama_install` | 下載 Ollama 0.35.0 tarball、驗證 sha256、`zstd -dc \| tar -x -C /usr` |
   | `ollama_start` | `ollama serve`（`OLLAMA_HOST=127.0.0.1:11434`、`KEEP_ALIVE=-1`、`CONTEXT_LENGTH=32768`、`MAX_LOADED_MODELS=2`、`NUM_PARALLEL=1`、`FLASH_ATTENTION=1`、`NO_CLOUD=1`、`MAX_QUEUE=8`、`LOAD_TIMEOUT=10m`，`OLLAMA_ORIGINS` 不設），等 `/api/version` 回 0.35.0 |
   | `pull` | 先 qwen3.8 再 nimble 拉兩個 tag（各最多 3 次，間隔 15、45 秒），再比對 `/api/tags` 的 digest 與 `versions.json`：qwen3.8 拉不到或不符就失敗；nimble 拉不到或不符只記一筆 ERROR 並繼續開機（見〈釘住的 digest〉） |
   | `warmup` | **一定先** `POST /api/chat`（空訊息、`keep_alive:-1`，要回 `done_reason:"load"`）載入 qwen3.8，**再**用一次很小的 `/v1/systemone` 載入 nimble（順序見〈VRAM〉）；接著記錄 VRAM：`/api/ps` 每個模型的 `size_vram`／`size` 與 `nvidia-smi` 的 `memory.used`，寫進 `state.json` 的 `vram` 與 `ready` 的 `gpu`。nimble 載入失敗或沒有完全在 GPU 上時只記一筆 ERROR 並繼續開機；載入 nimble 時若 Ollama 踢掉了 qwen3.8，就卸載 nimble、單獨重新載入 qwen3.8，不用 nimble 繼續開機（見〈VRAM〉） |
   | `proxy` | 啟動 `auth_proxy.py`，確認帶 token 回 200、不帶 token 回 401 |
   | `probe` | 透過 proxy 跑一次和 bot 相同的 readiness probe（`register.probe_problems`，qwen3.8 必須完全在 VRAM） |
   | `tunnel` | 下載 cloudflared 2026.9.3 並驗證 sha256；移開 `~/.cloudflared/config.yaml`；quick tunnel 從 metrics 的 `/quicktunnel` 讀網址，named tunnel 用 `TUNNEL_TOKEN`（只放環境變數，不進 argv）。90 秒沒連上就改 `--protocol http2` 再試一次 |
   | `tunnel_check` | 從 VM 經公開網址呼叫一次 `/api/version`（失敗只記 log，仍會註冊） |
   | `register` | 送出簽章過的 `ready`（`tunnel_url`、版本、實際的 digest、GPU 與 VRAM）；5xx 與 422（bot 暫時解析不到新 tunnel 的網域）會重送，最多共送 4 次（間隔 5、10、20 秒，每次重新簽章），其他 4xx 直接失敗 |
   | `ready` | 完成；bootstrap 結束，Ollama、proxy、cloudflared 繼續執行 |

   **任何一步失敗**：`state.json` 寫入錯誤 → 送出簽章過的 `failed`（含 step 與錯誤）→ 停掉這次啟動的 process → VM 自我釋放（`POST http://$TBE_RUNTIME_ADDR/unassign`）→ 結束。env 檔無效時沒有可信的金鑰可簽章，只寫 `state.json` 並自我釋放。
5. **READY**：bot 驗證簽章與 `nbh`、檢查 `tunnel_url`（只接受 `https://<子網域>.trycloudflare.com` 或 `COLAB_TUNNEL_HOST`，並拒絕解析到非公開 IP 的網域），再帶 bearer token 經 tunnel 跑 readiness probe（10 秒內）：`/api/version` 是 0.35.0、`/api/ps` 列出 qwen3.8 且完全在 VRAM（`size_vram == size`，否則 probe 失敗）、`/api/tags` 列出 qwen3.8 且 digest 與釘住的相同、qwen3.8 的 `/api/show` think 值是 `[false, "low", "medium", "xhigh"]`。通過才 READY。bot 記下 probe 看到的 digest 與 `/api/ps` 大小；nimble 沒拉到、沒載入、digest 與釘住的不同（或沒記錄到）、或沒有完全在 VRAM 時仍會 READY，但這個 session 不用 nimble 路由（見〈釘住的 digest〉與〈VRAM〉）。之後每分鐘的健康檢查只要求 qwen3.8 已載入；發現 nimble 不見了（被踢出 VRAM）或沒有完全在 VRAM 時，只讓 nimble 在這個 session 剩下的時間停用，不會把整個 Colab 層標成不健康。
6. **使用中**：每個 Colab 模型各一個 `asyncio.Semaphore(1)`，模型忙碌就直接跳過 Colab 層（Ollama 對已載入的忙碌模型不會回 503，只會排隊）。控制器在需求還新鮮時每 5 分鐘在同一個 kernel 跑一次 heartbeat cell（印出 `state.json` 前 300 字）：Colab 只把 kernel 活動算成使用中，tunnel 流量不算，約 22–25 分鐘沒有 kernel 活動就會被回收。
7. **釋放**：閒置 `COLAB_IDLE_TTL_SECONDS`（900）或滿 `COLAB_MAX_SESSION_SECONDS`（39600）時由控制器 unassign。VM 裡的 watchdog 是第二道保險（見下）。

## 協定摘要

完整定義在 `register.py` 的 docstring，bot 端的對應是 `src/services/colab/protocol.py`；`tests/test_colab_vm_scripts.py` 會交叉檢查兩邊的金鑰、簽章、釘版與上限完全一致，並讓 bot 的 `ColabRegistrar` 與 readiness probe 實際處理 VM 送出的事件（經過真正的 auth proxy 與 DRYRUN stub）。

- **金鑰**（Firestore 不存任何密鑰，bot 隨時可重算）：`reg_key = hex(HMAC_SHA256(COLAB_SHARED_SECRET, "reg:" + nbh))`；`proxy_token = base64url(HMAC_SHA256(COLAB_SHARED_SECRET, "proxy:" + nbh))`，去掉 `=`。
- **註冊**：`POST {BOT_URL}/internal/colab/register`，JSON ≤ 8 KB：`{event: ready|failed|stopping, nbh, tunnel_url?, versions?, digests?, gpu?, step?, error?}`；header `X-Colab-Signature: t=<unix 秒>,v1=<hex HMAC_SHA256(reg_key, f"{t}.{原始 body}")>`（`reg_key` 以其 hex 文字當 key）。bot 拒絕時間差 > 300 秒、`nbh` 不符、簽章錯誤、重放與過大的 body。
- **env 檔** `/content/.bootstrap.env`：每行 `KEY=VALUE`，不加引號；key 有 `COLAB_REG_KEY`、`COLAB_PROXY_TOKEN`、`BOT_URL`、`NBH`、`IDLE_TTL`、`MAX_SESSION`、`TUNNEL_MODE`（quick｜named）、`TUNNEL_TOKEN`（named）、`TUNNEL_HOST`（named 必填，等於 `COLAB_TUNNEL_HOST`：named tunnel 無法自己回報網址，而 bot 只接受帶 `tunnel_url` 的 `ready`）、`OPENCODE_ENABLED`、`DRYRUN`。`register.render_bootstrap_env` / `parse_bootstrap_env` 是寫與讀的標準實作。
- **`/content/state.json`**：`{step, ts, error}`，warm-up 之後再加 `vram`（`chat_on_gpu`／`router_on_gpu`：`/api/ps` 的 `size_vram/size`，單位 MiB；`used_mib`：`nvidia-smi` 的 `memory.used`），絕不含密鑰；錯誤截到 200 字，整份不超過 heartbeat 印出的 300 字（放不下時錯誤再截短）。`ready` 事件的 `gpu` 也帶同樣三個欄位，bot 只拿來顯示（`/health/detail` 的 `colab.gpu`），判斷一律用自己 probe 看到的數字。
- **auth proxy**：`hmac.compare_digest` 比對 bearer（錯誤回 401）；白名單 `GET/HEAD /api/version`、`GET /api/ps`、`GET /api/tags`、`POST /api/show`、`POST /api/chat`、`POST /v1/systemone`，`OPENCODE_ENABLED=1` 時再加 `POST /v1/chat/completions`、`GET /v1/models`，其餘（pull、push、delete、create、copy、blobs…）一律 404；轉送時移除 Authorization、Origin、Cookie 與 hop-by-hop header，`Host` 改成 `127.0.0.1:11434`；串流直接轉送；body 上限 16 MB，上游 150 秒逾時。**放棄的請求會在 VM 上真的停下來**：bot 取消請求（鏈的時間上限、回合期限、關機）時會關掉連線，cloudflared 隨即關掉它到 proxy 的連線；`serve()` 以 aiohttp 的 `handler_cancellation` 執行，handler 因此被取消、關掉通往 Ollama 的連線，Ollama 就停止生成，`auth_proxy.log` 記一行 `POST /api/chat cancelled after N s: the client went away`。VM 的檔案是開機時才上傳，所以 proxy 的改動要到下一次 Colab 開機才生效。**只有** `/api/chat`、`/v1/systemone`、`/v1/chat/completions` 會更新閒置計時，health 與 probe 不算。
- **watchdog**（proxy 內，每 30 秒）：閒置超過 `IDLE_TTL + 120` 秒、存活超過 `MAX_SESSION + 120` 秒，或 Ollama 重啟一次後仍然掛掉 → `state.json` 寫 `stopping` → 簽章過的 `stopping` 事件 → `POST http://$TBE_RUNTIME_ADDR/unassign`（預期 200）→ 結束。任何一步出錯都不會讓 watchdog 停擺：單次檢查出錯只記 log、下一輪照跑；連續 3 次出錯、釋放流程本身出錯（例如磁碟滿了寫不了 `state.json`），或 watchdog 的 task 整個掛掉，都直接送出上面那個 unassign POST（不寫 `state.json`、不送事件）然後結束。**不呼叫** `google.colab.runtime.unassign()`：它在非 kernel 的 process 裡送出 POST 後會在 `eval_js` 失敗。

## 一次性設定（需要你操作）

1. **專用 Google 帳號**（強烈建議）：用一個只持有 Colab 方案、**沒有任何 GCP 權限**的帳號。gcloud 產生的 refresh token 一定帶 `cloud-platform` scope；若用 terabanana 的 Owner 帳號，存在 Secret Manager 的 token 就等於 Owner 權限。這個帳號也不要拿來做其他 gcloud 登入（見下方 token 注意事項）。
2. **產生並儲存 refresh token**：先跑 `./scripts/mint_colab_adc.sh`（dry-run，只做唯讀檢查並印出計畫），再跑 `./scripts/mint_colab_adc.sh --apply`。它在暫存的 `CLOUDSDK_CONFIG`（不會蓋掉你本機原本的 ADC 與 gcloud 登入）下執行
   `gcloud auth application-default login --no-launch-browser --disable-quota-project --scopes=openid,https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/userinfo.email,https://www.googleapis.com/auth/colaboratory`；
   用只登入**專用帳號**的無痕視窗打開網址、同意後把授權碼貼回終端機。腳本確認檔案是帶 refresh token 的 `authorized_user` 憑證，經 stdin 存成 Secret Manager `linebot-colab-adc` 的新版本（第一次會建立 secret），執行 `./scripts/provision_gcp.sh --step colab-secret --apply` 讓 `linebot-runtime` 能讀取，最後刪除暫存目錄；token 內容不會印出。
3. **掛載**：`./scripts/deploy_cloud_run.sh` 在 `COLAB_ENABLED=true` 時把最新的 enabled 版本**釘住版本號**掛載在 `/secrets/colab/adc.json`（`COLAB_ADC_PATH` 的預設值），部署本身不讀它；`COLAB_ENABLED=false` 時不掛載。**絕不**設定 `GOOGLE_APPLICATION_CREDENTIALS`（否則 Firestore、GCS、Cloud Tasks 都會改用這個個人 token）。
4. **bot 設定**：`COLAB_SHARED_SECRET`（至少 32 個隨機字元，放在 dotenv secret），`COLAB_ENABLED=true`（需要 `FIRESTORE_ENABLED=true`、讀得到的 `COLAB_ADC_PATH` 與 `INTERNAL_CRON_SECRET`（控制器只在每分鐘的 cron 執行），缺少時 Colab 層維持停用、服務照常啟動），`LINE_ADMIN_USER_IDS`（在私訊對 bot 傳 `!whoami` 取得自己的 userId）。第一次實跑時再加 `MODEL_PATH_PREFERENCE=api`，只讓 `!colab on` 開機。完整步驟見 [DEPLOY.md](../DEPLOY.md)〈Colab 層〉。
5. （選用）**named tunnel**：有 Cloudflare 網域時，建立 tunnel、把公開網址指到 `http://127.0.0.1:8081`，設定 `COLAB_TUNNEL_MODE=named`、`COLAB_TUNNEL_HOST` 與 `COLAB_TUNNEL_TOKEN`；launcher 會以 `TUNNEL_MODE=named`、`TUNNEL_TOKEN`、`TUNNEL_HOST` 傳給 VM。
6. **先跑 DRYRUN**：設定 `COLAB_DRY_RUN=true`，控制器改申請 CPU runtime（`Variant.DEFAULT`），VM 以 `DRYRUN=1` 走完整個控制流程（assign、上傳、silent env、launcher 標記、register、heartbeat、watchdog 自我釋放、孤兒對帳、`!colab on/off/status`）。DRYRUN 跳過 GPU、Ollama 與模型，改啟動一個假的上游：`/api/version`、`/api/ps`、`/api/tags`、`/api/show` 回傳能通過 readiness probe 的固定內容，推論路徑則回 Ollama 的忙碌 503，所以這段期間真的使用者訊息會立刻改走 API 層，不會收到假回覆。READY 後 `.venv312/bin/python scripts/live_gate.py --path colab` 的 `colab.probe` 應為 PASS（其他三項 SKIP）。
7. **再跑一次真正的 A100**，READY 後 `scripts/live_gate.py --path colab` 四項都要 PASS；量測 assign 延遲、冷啟動時間、`consumptionRateHourly`、VRAM（照〈VRAM〉的首次實跑檢查記下來）、524、兩則對話同時進來時的忙碌閘門；再用 `scripts/evals/routing_eval.py` 重新調整 `COLAB_ROUTER_*` 門檻；然後再把 `MODEL_PATH_PREFERENCE` 改回 `colab`，對所有人開啟。

## 成本

- A100 40 GB 約 **5.37 CU/小時**（第三方數字；Google 未公布，以 assign 後 `ccu-info` 的 `consumptionRateHourly` 為準，控制器會記錄在狀態文件上）。
- **一次啟動 ≈ 2–2.7 CU**：開機 4–8 分鐘加閒置 15 分鐘，約 0.35–0.5 小時。計費從 assign 開始，不是模型載入完成才開始。
- **絕不用 high-mem**：`--high-mem` 會拿到 80 GB 的 A100，約 6.77 CU/小時。
- 保護措施：CU 餘額低於 10 不開機、每日上限 30 CU、閒置 15 分鐘或滿 11 小時釋放、watchdog 後援、任何啟動失敗都自我釋放、`!colab off` 與 `COLAB_ENABLED=false` 可一鍵關閉。DRYRUN 的 CPU runtime 也會用掉少量 CU。

## Colab 條款風險（已接受）

Colab FAQ 禁止把 runtime 當成 web service，也禁止讓第三方使用。你在 2026-09-29/30 已接受這個風險並選擇讓**所有使用者**先走 Colab（ADR-0010）。帳號仍可能被警告或封鎖。對策：專用帳號、一鍵關閉、assign 遇到 400／401／403 或 `DENYLISTED` 立即 DISABLED 並記錄 ERROR、CU 上限、從不讓回覆等待 Colab。

## Refresh token 注意事項

- **6 個月沒用會失效**：每日的孤兒對帳會持續使用它。
- **每個 OAuth client 最多 100 個有效 refresh token**：gcloud 的 `application-default login` 與 colab-cli 的 oauth2 登入共用同一個 Cloud SDK client（`764086051850-…`）。這個帳號每登入一次就多一個 token，超過 100 個時**最舊的會被默默撤銷**，所以專用帳號不要再做其他 gcloud／colab 登入。
- 你撤銷存取權、Workspace 管理員政策（session control、限制 scope）也會讓它失效。
- 失效時 bot 會看到 `invalid_grant` 或 `Reauthentication is needed`，Colab 層進入 DISABLED 並記錄 ERROR；回覆不受影響（全部走 API 層）。token 端點只是暫時出錯（502、HTML 頁面、沒有 token 的 200）時算暫時性錯誤，下一輪重試，不會 DISABLED。進入 DISABLED 時若還持有 runtime，會先送一次 unassign 釋放它（DISABLED 期間唯一會呼叫 Colab 的動作）；token 已失效而釋放不了時，由 VM 的 watchdog 釋放。
- **重新產生**：`./scripts/mint_colab_adc.sh --apply`（新增一個 `linebot-colab-adc` 版本）→ `./scripts/deploy_cloud_run.sh`（釘住最新的 enabled 版本；smoke check 通過後 destroy 沒有 revision 釘住、也不在最新 2 個之內的舊版本，disabled 版本仍計費，只有 destroyed 才免費）→ `!colab on`。
- **撤銷**：token 外洩時，用專用帳號登入 https://myaccount.google.com/connections 移除「Google Cloud SDK」；這會撤銷這個帳號透過 gcloud 產生的**所有** token（包括 bot 正在用的），之後照上面重新產生。被新版本取代的舊 token 不需要另外處理：部署會 destroy 它的版本，沒人使用 6 個月後也會自然失效。

## 疑難排解

先看 `!colab status`（狀態、最後錯誤、冷卻時間、CU 餘額與速率）。VM 還在時，heartbeat 會印出 `state.json`；VM 上的 log 在 `/content/bootstrap.log`、`ollama.log`（DRYRUN 是 `stub_upstream.log`）、`auth_proxy.log`、`cloudflared.log`。

| 現象 | 原因 | 處理 |
| --- | --- | --- |
| assign 回 **412** | `TooManyAssignmentsError`：同時開太多 runtime，或暫時的用量／容量限制 | 自動冷卻 5→10→20→60 分鐘。檢查專用帳號上是否有瀏覽器開著的 runtime（`colab sessions`） |
| assign 回 **503** | 暫時沒有 A100 | 自動冷卻後再試；常見於尖峰時段 |
| assign 回 **400** | 方案沒有 A100 權限或額度 | DISABLED。確認 Colab 方案與 CU 餘額 |
| assign 回 **401／403**、訊息含 **DENYLISTED** 或 abusive activity | token 權限錯誤，或帳號因疑似濫用被封鎖 | DISABLED。DENYLISTED 時先停用 Colab 層（`COLAB_ENABLED=false`），不要換帳號硬闖 |
| `invalid_grant`、`invalid_client`、`Reauthentication is needed` | refresh token 失效（`invalid_client`：憑證檔裡的 client 不對） | 依上面的步驟重新產生 |
| `state.json` 停在 `gpu` | 拿到的不是 40 GB A100，或驅動 < 550 | VM 已自我釋放；若一直發生，回報並檢查 Colab 的 GPU 配置 |
| `pull`：qwen3.8 的 digest 不是釘住的值 | qwen3.8 的 tag 在 registry 上被重新發佈 | 每次開機都會在這一步失敗並自我釋放（連續 3 次後 DISABLED），回覆全部走 API 層。照〈釘住的 digest〉確認新版本後重新釘版 |
| bot log 有 ERROR `nimble:9b off for this Colab session: nimble:9b has digest …`，`/health/detail` 的 `colab.unpinned` 或 `!colab status` 列出 `nimble:9b` | nimble 的 tag 被重新發佈（Bespoke 更新得很頻繁） | 不影響開機與 qwen3.8：這個 session 的路由改由規則與 API 路由決定（每個 session 只記一筆 ERROR，含釘住與實際的 digest）。要再用 nimble，照〈釘住的 digest〉確認新版本後重新釘版、重調門檻並部署 |
| bot log 有 ERROR `nimble:9b off for this Colab session: router not fully on GPU` 或 `router not loaded`，`/health/detail` 的 `colab.off_gpu` 或 `!colab status` 列出 `nimble:9b` | Ollama 把 nimble 部分放到 CPU、nimble 沒拉到或沒載入、開機時為了 qwen3.8 卸載了 nimble，或 session 中健康檢查發現 nimble 被踢出 VRAM（log 有 WARNING `health check found nimble:9b not loaded`） | 同上，這個 session 不用 nimble 路由，qwen3.8 照常。看 `bootstrap.log` 的 ERROR 與 `ollama.log`（`evicting`／`fits`），照〈VRAM〉記下數字後處理 |
| `probe`：`qwen3.8… is not loaded` 或 `is not fully on the GPU` | qwen3.8 只有部分在 GPU，或卸載 nimble 後 qwen3.8 仍載入失敗 | 開機失敗並自我釋放。看 `state.json` 的 `vram` 與 `ollama.log`（`evicting`／`fits`），照〈VRAM〉處理 |
| `tunnel`：90 秒內沒連上 | Colab 擋 UDP（QUIC） | 已自動改 `--protocol http2` 重試；兩者都失敗時看 `cloudflared.log` |
| `register`：HTTP 403／409 | 這台 VM 已不是目前的 session（`nbh` 不符或狀態已變） | VM 會自我釋放，不需處理 |
| `register`：HTTP 503、`readiness probe failed` | bot 經 tunnel 探測失敗 | VM 會重試；持續失敗就看 `!colab status` 的 problems |
| Colab 回應 **524** | 非串流呼叫超過 Cloudflare 的 125 秒 | bot 自動改走 API 層；把 `COLAB_DEEP_THINK` 調低 |

## 釘住的 digest

`versions.json` 與 `src/providers/model_profiles.py` 各有一份兩個模型的 digest（`tests/test_colab_vm_scripts.py` 與 `tests/test_colab_registration.py` 檢查兩邊一致），要**同時**修改。Ollama 只能拉 tag，registry 也不能用 digest 取 manifest（`GET /v2/library/<模型>/manifests/sha256:…` 連目前的 digest 都回 404），所以 404 **不能**證明某個 digest 被移除，釘版也只能在拉了 tag 之後比對。

- **qwen3.8 不符**：開機在 `pull` 失敗並自我釋放；bot 的 readiness probe 也拒絕它。
- **nimble 不符**：開機照常完成，`ready` 回報實際的 digest；bot 的 probe 通過、session 照樣 READY，但記下 probe 看到的 digest，這個 session 的 nimble 路由直接停用（不送任何請求），路由改由規則與 API 路由決定。每個 session 只記一筆 ERROR（含釘住與實際的 digest），`/health/detail` 的 `colab.unpinned` 與 `!colab status` 也會顯示。READY 狀態裡沒有記錄到 nimble 的 digest 時也當成不符。`scripts/live_gate.py --path colab` 的 `colab.probe` 仍會把它列為不符。
- **只換 `MODEL_ROLES` 不行**：路由層的模型、`versions.json` 與 model profile 必須一起換。VM 沒拉的模型在 `/v1/systemone` 回 404，而 404 會把整個 Colab 層（包括 qwen3.8 對話）標成不健康，直到下一次健康檢查通過。
- **確認新版本**：以 tag 取回 manifest，它的 sha256 就是 Ollama 的 digest；再看 layers 與 config 改了什麼（模型層、params、template、SYSTEM prompt）：

  ```bash
  curl -s -H 'Accept: application/vnd.docker.distribution.manifest.v2+json' \
    https://registry.ollama.ai/v2/library/nimble/manifests/9b | tee /tmp/manifest.json | shasum -a 256
  python3 -m json.tool /tmp/manifest.json
  ```

  小的 layer（system、params、config）可用 `https://registry.ollama.ai/v2/library/nimble/blobs/sha256:<digest>` 取回來看。確認後修改兩個檔案、跑測試、部署；nimble 換了 checkpoint 時，也要用 `scripts/evals/routing_eval.py` 重新調整 `COLAB_ROUTER_*` 門檻。

## VRAM

A100 40 GB 上同時放 qwen3.8（32K context）與 nimble（Q8_0，`num_ctx` 8194），餘裕比以前小，而且**沒有實測過**：

- Ollama 載入新模型時，若預估大小超過目前空閒 VRAM 的 80% 就會踢掉已載入的模型。它對 nimble 的預估約 9.87 GiB（GGUF 9,527,501,312 bytes 加 8194 的 KV cache），所以 qwen3.8 載入後至少要剩 **約 12.3 GiB**；Ollama 自動選 batch 1024 時再加 768 MiB，約 **13.3 GiB**。
- 所以 warm-up **一定先載入 qwen3.8 再載入 nimble**：反過來的話 qwen3.8（預估約 22.4 GiB）要在 nimble 已佔用時找到約 28 GiB 空閒，幾乎放不下。Ollama 0.35.0 預估放不下時是踢掉閒置的模型、不會把新模型部分放到 CPU，所以 nimble 放不下時被踢掉的是 qwen3.8：bootstrap 會發現 `/api/ps` 沒有 qwen3.8，就卸載 nimble（`/api/generate`、`keep_alive: 0`）、單獨重新載入 qwen3.8，記一筆 ERROR 後不用 nimble 繼續開機。
- session 中 qwen3.8 單獨被重新載入（例如它的 runner 當掉）而 nimble 還在時，同樣放不下，Ollama 會踢掉閒置的 nimble；健康檢查發現 nimble 不見了，只讓 nimble 在這個 session 剩下的時間停用（WARNING `health check found nimble:9b not loaded`），qwen3.8 對話照常，也不會再呼叫 nimble（重新載入它可能又踢掉 qwen3.8）。watchdog 重啟 Ollama 時一樣先載入 qwen3.8；只有 qwen3.8 沒回來才算重啟失敗。
- 保護措施：readiness probe 要求 qwen3.8 `size_vram == size`（否則開機失敗）；nimble 沒有完全在 VRAM 或沒有載入時 session 照樣 READY，但這個 session 不用 nimble 路由（ERROR `router not fully on GPU` 或 `router not loaded`，`/health/detail` 的 `colab.off_gpu`，只在 READY 時列出）。`OLLAMA_MAX_LOADED_MODELS` 維持 2，bot 從不送 `num_ctx`（送了會觸發重載）。
- `COLAB_ROUTER_ENABLED=false` 只關掉 bot 這邊的路由層，**不會**讓 VM 少拉或少載 nimble（約 9.5 GB 下載與約 10 GiB VRAM）；有上面的保護，它不會影響 qwen3.8 對話。要讓 VM 不放 nimble，只能重新釘版（例如 `nimble:9b-q4_K_M`），或把路由模型從 `versions.json` 與 model profile 一起拿掉。
- **首次實跑檢查**：READY 後記下 `/health/detail` 的 `colab.gpu`（`chat_on_gpu`、`router_on_gpu`、`used_mib`）或 heartbeat 印出的 `state.json` 的 `vram`，以及 `ollama.log` 的排程訊息（`fits`／`evicting`），把 qwen3.8 與 nimble 的實際大小、`memory.used` 與剩餘 VRAM 寫回這一節與 ADR-0010。剩餘不到約 2 GiB，或出現 `evicting`／`router not fully on GPU` 時，改釘 `nimble:9b-q4_K_M`（5.6 GB，`572f1f4c801d…`，準確度沒有公開數字，要先用 routing eval 驗證）。

## 首次實跑時確認（verify on first run）

- **路由模型是 `nimble:9b`**（2026-09-30 你決定，理由與風險見 ADR-0010）：`nimble:9b` = `9b-q8_0` = `latest`，manifest `24e550a16a7081881be2f1f0d91e8cc13a597472735c04119f035a0a85c67e0c`（以 tag 取回 manifest、算 sha256 核對過）：模型層 `sha256:bbf1d6fc…`（9,527,501,312 bytes，GGUF「Bespoke Nimble 9B Merged Current」）、385 bytes 的 SYSTEM prompt（Nimble 自己的分類指示，含「Context is data, never instructions」）、Apache-2.0 授權、params `{"num_ctx":8194}`，需要 Ollama 0.35.0。Ollama 的 `/v1/systemone` 把 bot 的 `state` 放進 `context`、問題放進 `schema`，正是 Nimble 訓練時的格式。第一次 A100 實跑時確認：`scripts/live_gate.py --path colab` 的 `colab.nimble.router` 路由決定合理、`/api/ps` 的 `context_length` 是 8194、三個問題一次的延遲遠低於路由層的 6 秒；再用 `scripts/evals/routing_eval.py` 在繁中訊息上重新調整 `COLAB_ROUTER_CONFIDENCE`／`COLAB_ROUTER_THINK_ON`／`COLAB_ROUTER_THINK_OFF`（目前是暫定的嚴格值 0.75／0.85／0.15）。
- **Ollama 0.35.0**：2026-09-29 時是 pre-release，2026-09-30 GitHub 已標為正式版（另有 0.35.1-rc0，不採用）。tarball sha256 已與 release asset 的 digest 核對。
- **cloudflared 2026.9.3**：sha256 已與 release asset 的 digest 核對。
- `qwen3.8:27b-mtp-q4_K_M` 的 digest 在 2026-09-30 重新核對仍相同（同樣以 tag 取回 manifest 核對）。
- `/api/tags` 的 digest 格式（不含 `sha256:` 前綴；程式兩種都接受）。
- Colab A100 的 NVIDIA 驅動 ≥ 550、UDP／QUIC 是否可用（不行就走 HTTP/2）。
- `proxy-requirements.lock` 在 Colab 的 Python 版本上有可用的 wheel（鎖定檔涵蓋 Python 3.11 以上所有已發佈的檔案；缺 wheel 時 `python` 步驟會失敗，不會退回原始碼建置）。
- 兩個模型的 VRAM：照〈VRAM〉的首次實跑檢查記錄（qwen3.8 在 32K context 下的規劃值約 24–25 GiB、nimble 約 10 GiB，都還沒實測）。
- NDJSON 串流經 quick tunnel 是否即時；bot 放棄請求後，`auth_proxy.log` 是否出現 `cancelled after N s`（proxy 收到斷線後一定會中止 Ollama 的請求，測試以真正的 `serve()` 驗證；實跑要確認的是 Cloudflare 與 cloudflared 會把斷線傳到 proxy）。
- heartbeat 是否確實讓 runtime 不被回收。

## opencode 擴充

兩種做法，都需要 64K context：把 `bootstrap.py` 的 `OLLAMA_SERVE_ENV["OLLAMA_CONTEXT_LENGTH"]` 改成 `65536`。bot 從不在請求裡送 `num_ctx`，所以 qwen3.8 自然用這個值、不會觸發模型重載（nimble 有自己的 `num_ctx` 8194，不受影響）；但 qwen3.8 的 KV cache 會多約 2 GiB，依〈VRAM〉的預估 nimble 很可能就放不下了，先照首次實跑的數字確認。

1. **推薦：另開一個 session。** 用 `colab ssh --proxy-mode` 把 VM 的 11434 port-forward 到 Mac，opencode 設 `baseURL: http://localhost:11434/v1`。不需要公開網址，SSE 串流也正常；和 bot 的 session 完全分開，互不影響閒置計時與忙碌閘門。
2. **共用 bot 的 proxy。** 設 `OPENCODE_ENABLED=1`，proxy 會多開 `POST /v1/chat/completions` 與 `GET /v1/models`（一樣要 bearer token），opencode 的請求也算活動。quick tunnel 不支援 SSE，所以這個做法需要 named tunnel。注意：qwen3.8 一次只服務一個請求，opencode 使用中時 bot 的忙碌閘門會跳過 Colab 層；而 bot 控制器的閒置計時只看 LINE 的需求，opencode 的流量不會阻止控制器在 15 分鐘後釋放 session。
