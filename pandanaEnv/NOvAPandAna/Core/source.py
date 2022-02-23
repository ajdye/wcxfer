import sys
import glob

try:
    import samweb_client
    ALLOWSAMWEB = True
except ImportError:
    ALLOWSAMWEB = False

def GetFileList(query, offset, stride, limit):
    if type(query) is list:
        print("Using list souce")
        return query[offset::stride][:limit]
    elif type(query) is str and " " not in query:
        print("Using glob souce")
        return glob.glob(query)
    elif ALLOWSAMWEB:
        pass
    else:
        # Perhaps you're using a custom iterator?
        return query
