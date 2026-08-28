"""Tenhou (天凤) phoenix-lobby data pipeline.

Modules:
- index_dl.py    : fetch list.cgi / list.cgi?old -> index manifest
- listing.py     : parse scc{yyyymmdd}{hh}.html.gz -> (logid, time) for 四鳳南 (lobby 00a9)
- download.py    : rate-limited, resumable mjlog XML downloader
- mjlog_parser.py: state-machine parser for mjlog XML
- shanten_tmp.py : temporary shanten/agari helpers (TODO: migrate to src/riichi/)
- extract.py     : decision-point records (docs/observation_schema.md 2/3)
- stats.py       : dataset statistics -> docs/data_report.md
- batch.py       : background batch pipeline (list -> download -> parse -> extract)
"""
__version__ = "0.1.0"
