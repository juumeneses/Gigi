"""Unix flock compatible API, with a nonblocking Windows byte lock."""
import os

if os.name != 'nt':
    from fcntl import LOCK_EX, LOCK_NB, LOCK_UN, flock
else:
    import errno
    import msvcrt
    LOCK_EX, LOCK_NB, LOCK_UN = 2,4,8

    def flock(file,operation):
        fd=file.fileno() if hasattr(file,'fileno') else file
        if os.fstat(fd).st_size==0:os.write(fd,b'\0')
        os.lseek(fd,0,os.SEEK_SET)
        mode=msvcrt.LK_UNLCK if operation & LOCK_UN else msvcrt.LK_NBLCK
        try:msvcrt.locking(fd,mode,1)
        except OSError as exc:
            if not operation & LOCK_UN and exc.errno in (errno.EACCES,errno.EAGAIN,errno.EDEADLK):
                raise BlockingIOError('Outra tarefa ja esta usando o laboratorio') from exc
            raise
