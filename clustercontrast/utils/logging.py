from __future__ import absolute_import
import os
import sys

from .osutils import mkdir_if_missing


class Logger(object):
    """
    将 stdout 同时写到终端与 log 文件。
    注意：不得 close() 保存的 console（即原 sys.stdout），否则后续 print 会失败或静默丢输出。
    """

    def __init__(self, fpath=None):
        # 必须在执行 sys.stdout = self 之前保存当前 stdout（终端），否则 tee 会递归或丢输出
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(os.path.dirname(fpath))
            # 行缓冲 + 每次 write 后 flush：log.txt 与终端同步；errors=replace 避免个别非法字符整段写失败
            self.file = open(
                fpath, 'w', encoding='utf-8', errors='replace', buffering=1
            )

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        try:
            self.console.write(msg)
            self.console.flush()
        except Exception:
            pass
        if self.file is not None:
            try:
                self.file.write(msg)
                self.file.flush()
            except Exception:
                pass

    def flush(self):
        try:
            self.console.flush()
        except Exception:
            pass
        if self.file is not None:
            try:
                self.file.flush()
            except Exception:
                pass

    def close(self):
        # 禁止关闭 self.console（不要关 sys.__stdout__ / 原终端）
        if self.file is not None:
            try:
                self.file.flush()
                try:
                    os.fsync(self.file.fileno())
                except OSError:
                    pass
                self.file.close()
            except Exception:
                pass
            self.file = None
