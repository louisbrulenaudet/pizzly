__version__ = "0.1.0"

VERSION_MAJOR = 0
VERSION_MINOR = 1
VERSION_PATCH = 1

VERSION_INFO = (VERSION_MAJOR, VERSION_MINOR, VERSION_PATCH)
VERSION = ".".join(map(str, VERSION_INFO))

assert __version__ == VERSION, "Version representations are not synchronized"
