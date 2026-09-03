import logging
import os

from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="A+ SMC Trading Monitor", docs_url="/docs",
              description="24/7 Smart Money Concepts monitor for 6 forex/commodity pairs. "
                          "Sends Telegram alerts when all 4 A+ rules are simultaneously satisfied.")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

_STATIC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static")
_API_KEY = os.environ.get("SCAN_API_KEY", "")


def _check_api_key(x_api_key: str = Header(default="")):
    if not _API_KEY:
        raise HTTPException(500, "SCAN_API_KEY not configured")
    if x_api_key != _API_KEY:
        raise HTTPException(403, "Invalid API key")


@app.get("/")
def root():
    return {"status": "ok", "service": "A+ SMC Trading Monitor"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/version.json")
def version():
    path = os.path.join(_STATIC_DIR, "version.json")
    if os.path.exists(path):
        return FileResponse(path, media_type="application/json",
                            headers={"Access-Control-Allow-Origin": "*"})
    return JSONResponse({}, status_code=404, headers={"Access-Control-Allow-Origin": "*"})


@app.post("/scan", summary="Run one full SMC scan cycle",
          description="Scans all 6 pairs through the 4-rule SMC engine and sends Telegram alerts "
                      "for any A+ setups found. Protected by API key.")
def run_scan(x_api_key: str = Header(default="")):
    """Execute one scan cycle across all monitored pairs."""
    _check_api_key(x_api_key)
    from .scanner import scan_all_pairs
    result = scan_all_pairs()
    return result


@app.post("/notify/startup", summary="Send startup notification",
          description="Sends the startup Telegram notification. Protected by API key.")
def startup_notification(x_api_key: str = Header(default="")):
    _check_api_key(x_api_key)
    from .telegram import send_startup_notification
    ok = send_startup_notification()
    return {"sent": ok}


@app.get("/state", summary="View current state",
         description="Returns active alerts, pending FVG setups, and session status.")
def get_state(x_api_key: str = Header(default="")):
    _check_api_key(x_api_key)
    from .state import _load, _ensure_keys, get_all_pending
    from .smc_engine import is_within_trading_session
    state = _ensure_keys(_load())
    session_ok, session_info = is_within_trading_session()
    return {
        "session_active": session_ok,
        "session_info": session_info,
        "active_alerts": state.get("alerts", {}),
        "pending_fvg_setups": get_all_pending(),
    }
