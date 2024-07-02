def fmtsize(size):
    match size:
        case _ if size < 1024:
            return str(size)
        case _ if size < 1024*1024:
            size = size / 1024
            return f"{size:.1f}k"
        case _ if size < 1024*1024*1024:
            size = size / (1024*1024)
            return f"{size:.1f}M"
        case _ if size < 1024*1024*1024*1024:
            size = size / (1024*1024*1024)
            return f"{size:.1f}G"