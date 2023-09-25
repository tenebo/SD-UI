import os
from pathlib import Path
from modules.paths_internal import script_path
def is_restartable():'\n    Return True if the webui is restartable (i.e. there is something watching to restart it with)\n    ';return bool(os.environ.get('SD_WEBUI_RESTART'))
def restart_program():'creates file tmp/restart and immediately stops the process, which webui.bat/webui.sh interpret as a command to start webui again';(Path(script_path)/'tmp'/'restart').touch();stop_program()
def stop_program():os._exit(0)