# Overrides the contrib hook: the module `webrtcvad` is shipped by the
# distribution `webrtcvad-wheels`, so copy_metadata('webrtcvad') fails.
from PyInstaller.utils.hooks import copy_metadata

datas = copy_metadata("webrtcvad-wheels")
